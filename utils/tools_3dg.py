from random import random
import multiprocessing
from contextlib import contextmanager, ExitStack
import numpy as np
import torch
from torch.backends import cudnn
from torch import nn, einsum
from torch.autograd import grad as torch_grad


# constants
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['nii.gz', 'nii'] # changed extensions from jpg or png to nii.gz or nii

# make code faster when input size is constant
cudnn.benchmark = True

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


# Additonal feature related functions are commented out
# class RandomApply(nn.Module):
#     def __init__(self, prob, fn, fn_else = lambda x: x):
#         super().__init__()
#         self.fn = fn
#         self.fn_else = fn_else
#     def forward(self, x):
#         fn = self.fn if random() < self.prob else self.fn_else
#         return fn(x)
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#     def forward(self, x):
#         return self.fn(x) + x
#
# class ChanNorm(nn.Module):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
#         self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))
#
#     def forward(self, x):
#         var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) / (var + self.eps).sqrt() * self.g + self.b
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = ChanNorm(dim)
#
#     def forward(self, x):
#         return self.fn(self.norm(x))


# # attention
# class DepthWiseConv2d(nn.Module):
#     def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias), # Changed to Conv3d
#             nn.Conv3d(dim_in, dim_out, kernel_size = 1, bias = bias) # Changed to Conv3d
#         )
#     def forward(self, x):
#         return self.net(x)
#
# class LinearAttention(nn.Module):
#     def __init__(self, dim, dim_head = 64, heads = 8):
#         super().__init__()
#         self.scale = dim_head ** -0.5
#         self.heads = heads
#         inner_dim = dim_head * heads
#
#         self.nonlin = nn.GELU()
#         self.to_q = nn.Conv3d(dim, inner_dim, 1, bias = False) # Changed to Conv3d
#         self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
#         self.to_out = nn.Conv3d(inner_dim, dim, 1) # Changed to Conv3d
#
#     def forward(self, fmap):
#         h, x, y = self.heads, *fmap.shape[-2:]
#         q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))
#
#         q = q.softmax(dim = -1)
#         k = k.softmax(dim = -2)
#
#         q = q * self.scale
#
#         context = einsum('b n d, b n e -> b d e', k, v)
#         out = einsum('b n d, b d e -> b n e', q, context)
#         out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)
#
#         out = self.nonlin(out)
#         return self.to_out(out)
#
# # one layer of self-attention and feedforward, for images
# attn_and_ff = lambda chan: nn.Sequential(*[
#     Residual(PreNorm(chan, LinearAttention(chan))),
#     Residual(PreNorm(chan, nn.Sequential(nn.Conv3d(chan, chan * 2, 1), leaky_relu(), nn.Conv3d(chan * 2, chan, 1)))) # Changed to Conv3d
# ])

# helpers
def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def parse_list(arg_value):
    return [item.strip() for item in arg_value.strip("[]").split(',')]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

# For MedicalNet feature extraction
'''
def generate_model():
    model = resnet10(sample_input_W=image_size[0],
                sample_input_H=image_size[1],
                sample_input_D=image_size[2],
                num_seg_classes=1)
    return model
'''

def loss_backwards(fp16, scaler, loss, **kwargs):
    if fp16:
        scaler.scale(loss).backward(**kwargs)
    else:
        loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    device = images.device
    num_voxels = images.shape[2] * images.shape[3] * images.shape[4]  # Adjusted for 3D
    # pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_voxels)
    pl_noise = torch.randn(images.shape, device=device) / (num_voxels ** (1/3))
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def tanh():
    return nn.Tanh()

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


# dataset
def normalize_and_unsqueeze_nifti(image):
    # Normalize the image to the range [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = np.transpose(image, (2, 0, 1))

    # Convert to a PyTorch tensor and add a channel dimension for grayscale
    image_grayscale_tensor = torch.from_numpy(image).unsqueeze(0).float()  # Shape: [1, D, H, W]
    return image_grayscale_tensor

# augmentations
def random_hflip(tensor, prob):
    if prob < random():
        return tensor

    # horizontal flipping in differentiable augmentations
    return torch.flip(tensor, dims=(3,))


