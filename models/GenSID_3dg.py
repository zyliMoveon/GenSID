import math
from math import floor, log2
from functools import partial
import torch
from torch.optim import Adam
import torch.nn.functional as F
from kornia.filters import filter2d, filter3d # added filter3d for blurring

from ..utils.diff_augment_3dg import DiffAugment
from ..utils.tools_3dg import *
from ..utils.attention import MultiHeadCrossAttention

class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[:, None, None] * f[None, :, None] * f[None, None, :]
        f = f.unsqueeze(0)
        # Apply the 3D filter to x
        return filter3d(x, f, normalized=True)

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

# Stylegan2 classes
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class GrayscaleBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False, xy_upsample=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 1  # Since images are grayscale
        self.conv = Conv3DMod(input_channel, out_filters, 1, demod=False)  # Using 3D convolution

        # Conditional upsampling
        #if xy_upsample:
        #    self.upsample = nn.Sequential(
        #        nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=False),
        #        Blur()
        #    ) if upsample else None
        #else:
        self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                Blur()
            ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, d, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x


class Conv3DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        # Adjust the weight parameter initialization based on kernel dimensionality
        if isinstance(kernel, tuple):
            self.weight = nn.Parameter(torch.randn(out_chan, in_chan, *kernel))  # Unpack the tuple
        else:
            self.weight = nn.Parameter(torch.randn(out_chan, in_chan, kernel, kernel, kernel))
        # added another kernel for the 3rd dimension
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')


    def _get_same_padding(self, size, kernel, dilation, stride):
        if isinstance(kernel, tuple):
            # Calculate padding for each dimension separately
            padding = tuple(((s - 1) * (stride - 1) + dilation * (k - 1)) // 2 for s, k in zip(size, kernel))
            return padding
        else:
            # Single-dimensional kernel
            return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, depth, h, w = x.shape # Added depth dimension 'depth'

        w1 = y[:, None, :, None, None, None]
        w2 = self.weight[None, :, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, depth, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        # Adjust the call to _get_same_padding
        if isinstance(self.kernel, tuple):
            padding = self._get_same_padding((depth, h, w), self.kernel, self.dilation, self.stride)
        else:
            padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)


        x = F.conv3d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, depth, h, w)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False, xy_upsample=False, xy_upsample_rgb=False, cond_dim=512):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear',
                                    align_corners=False) if upsample and not xy_upsample else None

        # XY-specific 2D upsampling, Z or depth dimension remains the same
        #self.xy_upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear',
        #                               align_corners=False) if xy_upsample else None

        kernel = (3, 3, 3) if xy_upsample else 3
        #print(kernel)
        self.cond_dim = cond_dim  # MODIFIED

        #input_channels = input_channels * 2
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.cond_trans1 = nn.Linear(cond_dim, input_channels)  # MODIFIED
        #self.MHCA1 = MultiHeadCrossAttention(d_model=input_channels, num_heads=input_channels/32)  # MODIFIED
        self.MHCA1 = MultiHeadCrossAttention(d_model=input_channels, num_heads=self.get_MHCA_headnum(input_channels))  # MODIFIED
        self.conv1 = Conv3DMod(input_channels, filters, kernel)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.cond_trans2 = nn.Linear(cond_dim, filters)  # MODIFIED
        #self.MHCA2 = MultiHeadCrossAttention(d_model=filters, num_heads=filters/32)  # MODIFIED
        self.MHCA2 = MultiHeadCrossAttention(d_model=filters, num_heads=self.get_MHCA_headnum(filters))  # MODIFIED
        self.conv2 = Conv3DMod(filters, filters, kernel)

        self.activation = leaky_relu()
        self.to_grayscale = GrayscaleBlock(latent_dim, filters, upsample_rgb, rgba, xy_upsample=xy_upsample_rgb)

    def get_MHCA_headnum(self, input_dim):
        if input_dim > 128:
            return input_dim/128
        else:
            return 1

    def forward(self, x, prev_rgb, istyle, inoise, condition):
        if self.upsample is not None:
            x = self.upsample(x)

        #elif self.xy_upsample is not None:
        #   x = self.xy_upsample(x)

        a = x.shape[2]
        b = x.shape[3]
        c = x.shape[4]

        inoise = inoise[:, :a, : b, : c, :]

        noise1 = self.to_noise1(inoise).permute((0, 4, 1, 2, 3))

        noise2 = self.to_noise2(inoise).permute((0, 4, 1, 2, 3))

        style1 = self.to_style1(istyle)
        condition_trans1 = self.cond_trans1(condition)  # MODIFIED
        style1 = self.MHCA1(style1, condition_trans1)  # MODIFIED
        x = self.conv1(x, style1)

        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        condition_trans2 = self.cond_trans2(condition)  # MODIFIED
        style2 = self.MHCA2(style2, condition_trans2)  # MODIFIED
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_grayscale(x, prev_rgb, istyle)

        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True, xy_specific=False):
        super().__init__()
        #self.xy_specific = xy_specific
        stride = (1, 2, 2) if xy_specific else (2, 2, 2) if downsample else 1
        self.conv_res = nn.Conv3d(input_channels, filters, 1, stride = stride) # changed to 3d

        #'''
        self.net = nn.Sequential(
            nn.Conv3d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv3d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        if downsample:
            conv_stride = (1, 2, 2) if xy_specific else (2, 2, 2)
            self.downsample = nn.Sequential(
                Blur(),
                nn.Conv3d(filters, filters, 3, padding=1, stride=conv_stride)
            )
        else:
            self.downsample = None
        #'''
        '''
        if downsample:
            self.net = nn.Sequential(
                nn.Conv3d(input_channels, filters, 3, padding=1),
                leaky_relu(),
                nn.Conv3d(filters, filters, 3, padding=1),
                leaky_relu()
            )
            conv_stride = (1, 2, 2) if xy_specific else (2, 2, 2)
            self.downsample = nn.Sequential(
                Blur(),
                nn.Conv3d(filters, filters, 3, padding=1, stride=conv_stride)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv3d(input_channels, filters, 3, padding=1),
                leaky_relu(),
                nn.Conv3d(filters, filters, 3, padding=1),
                tanh()
            )
            self.downsample = None
        '''

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        #print(x.shape, res.shape)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512, cond_dim=512):  # 512 fmap_max
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim  # MODIFIED

        self.num_layers = int(log2(image_size) - 1)  # Layers for XY plane
        #self.num_layers_z = int(log2(32) - 2)  # Layers for Z dimension # 32 for the Paper, please adapt depth parameter here.
        #self.num_layers_z = int(log2(image_size) - 2)

        # the input parameter network-capacity is used to determine the number of filters for the generator blocks.
        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)

        filters = list(map(set_fmap_max, filters))
        filters = [i * 2 for i in filters]
        print("Generator", filters)

        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])

        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4, 4)))# added 4 for the depths dimension
            #print(self.initial_block.shape)

        self.initial_conv = nn.Conv3d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])


        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            #xy_upsample = ind > self.num_layers_z # Apply XY-specific upsampling in the later layers
            #xy_upsample_rgb = ind > self.num_layers_z  # Apply XY-s

            attn_fn = attn_and_ff(in_chan) if ind in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent,
                #xy_upsample=xy_upsample,
                #xy_upsample_rgb=xy_upsample_rgb,
                cond_dim=cond_dim
            )
            self.blocks.append(block)


    def forward(self, styles, input_noise, condition):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)
        i=0

        for style, block, attn in zip(styles, self.blocks, self.attns):

            if exists(attn):
                x = attn(x)

            x, rgb = block(x, rgb, style, input_noise, condition)  # MODIFIED
            i+=1

        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):# 512
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        #image_size_z = 32 # Layers for Z dimension # 32 for the Paper, please adapt depth parameter here.
        image_size_z = image_size
        num_layers_z = int(log2(image_size_z) - 1)

        num_init_filters = 1

        blocks = []
        # the input parameter network-capacity is used to determine the number of filters for the discriminator blocks.
        filters = [num_init_filters] + [int(network_capacity * 1) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))

        print("Discriminator", filters)
        chan_in_out = list(zip(filters[:-1], filters[1:]))
        print("Chan in out", chan_in_out)

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind != (len(chan_in_out) - 1)

            if ind <= num_layers_z:
                block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            else:
                block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last, xy_specific=True)

            blocks.append(block)

            #Attention and quantization
            attn_fn = attn_and_ff(out_chan) if ind + 1 in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if ind + 1 in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * 2 * chan_last

        self.final_conv = nn.Conv3d(chan_last, chan_last, 3, padding=1)
        #self.tanh = tanh()  # MODIFIED
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        #x = self.tanh(x)  # MODIFIED

        x = self.flatten(x)

        x = self.to_logit(x)
        return x.squeeze(), quantize_loss

class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim = 4096, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0): # fmap_max 512
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)
        self.cond_dim = 512  # MODIFIED

        #self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max, cond_dim=self.cond_dim)  # MODIFIED
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)

        #self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max )

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        #set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        #generator_params = list(self.G.parameters()) + list(self.S.parameters())
        generator_params = self.G.parameters()
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # mixed precision
        self.fp16 = fp16


    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv3d, nn.Linear}: # changed to Conv3d
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        #update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        #self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x