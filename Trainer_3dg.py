import os
import json
from tqdm import tqdm
from shutil import rmtree
import torch
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from pathlib import Path
import nibabel as nib
from torch.cuda.amp import autocast, GradScaler
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

from utils.MRIDataset import MRIDataset
from utils.tools_3dg import *
from utils.loss_3dg import *
from models.GenSID_3dg import *
from models.Resnet_3dg import resnet18, resnet50
from monai.losses.ssim_loss import SSIMLoss

class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = 128,
        network_capacity = 16,
        fmap_max = 512,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 1000,
        num_image_tiles = 4,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['translation', 'cutout'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        self.fp16 = fp16
        self.scaler = GradScaler()

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.L1LOSS = 0
        self.L2LOSS = 0
        self.SSIMLOSS = 0
        self.PERCEPTUALLOSS = 0
        self.MSSSIMLOSS = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None
        self.WCLoss = None
        self.CLSLoss = None
        self.WELoss = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.logger = aim.Session(experiment=name) if log else None

        # mixed precision
        self.scaler = GradScaler()

        # prepare classifier img -> wc
        self.classifier = resnet18(spatial_size=128, sample_duration=128, num_classes=2)  # MODIFIED
        empty = torch.nn.Sequential()  # MODIFIED
        self.classifier.fc1 = empty  # MODIFIED
        self.classifier.fc3 = empty  # MODIFIED
        cls_path = '/your/pretrained/classifier/path/'  # MODIFIED
        self.classifier.load_state_dict(torch.load(cls_path + '4_80_net.pth'), strict=False)  # MODIFIED
        self.classifier = self.classifier.cuda(rank)  # MODIFIED
        self.classifier.eval()  # MODIFIED
        for param in self.classifier.parameters():
            param.requires_grad = False
        print('CLASSIFIER LOADED SUCCESSFULLY!')  # MODIFIED

        # prepare classifier img -> label
        self.cls_ori = resnet18(spatial_size=128, sample_duration=128, num_classes=2)  # MODIFIED
        self.cls_ori.load_state_dict(torch.load(cls_path + '4_80_net.pth'))
        self.cls_ori = self.cls_ori.cuda(rank)  # MODIFIED
        self.cls_ori.eval()  # MODIFIED
        for param in self.cls_ori.parameters():
            param.requires_grad = False
            #print('require grad? ', param.requires_grad)
        print('ORI_CLS LOADED SUCCESSFULLY!')

        # prepare encoder img -> embedding
        self.encoder = resnet50(spatial_size=128, sample_duration=128, num_classes=2)  # MODIFIED
        empty = torch.nn.Sequential()  # MODIFIED
        #self.encoder.fc1 = nn.Linear(2048, 1024)
        self.encoder.avgpool = empty
        self.encoder.fc1 = empty
        self.encoder.fc3 = empty
        self.encoder = self.encoder.cuda(rank)
        self.E_opt = Adam(self.encoder.parameters(), lr=self.lr, betas=(0.5, 0.9))
        print('ENCODER CREATED!')

    @property
    def image_extension(self):
        return 'png' # if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, fmap_max = self.fmap_max, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            #self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder):
        self.dataset = MRIDataset(mode="labeled",fold=4)
        num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'
        apply_clear_cache = self.steps % 10 == 0
        #print("E_Lr:{}".format(self.E_opt.state_dict()['param_groups'][0]['lr']))
        #print("G_Lr:{}".format(self.GAN.G_opt.state_dict()['param_groups'][0]['lr']))
        #print("D_Lr:{}".format(self.GAN.D_opt.state_dict()['param_groups'][0]['lr']))
        if apply_clear_cache:
            torch.cuda.empty_cache()

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        self.encoder.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)
        total_L1_loss = torch.tensor(0.).cuda(self.rank)
        total_L2_loss = torch.tensor(0.).cuda(self.rank)
        total_SSIM_loss = torch.tensor(0.).cuda(self.rank)
        total_perceptual_loss = torch.tensor(0.).cuda(self.rank)
        total_MsSSIM_loss = torch.tensor(0.).cuda(self.rank)
        total_wc_loss = torch.tensor(0.).cuda(self.rank)
        total_cls_loss = torch.tensor(0.).cuda(self.rank)
        total_we_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 16 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000
        apply_cls_loss = self.steps > 180000
        apply_we_loss = self.steps > 96000
        apply_wc_loss = self.steps > 180000

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        if exists(self.GAN.D_cl):
            self.GAN.D_opt.zero_grad()
            print('enter GAN D_opt')

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
                    noise = image_noise(batch_size, image_size, device=self.rank)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader).cuda(self.rank)
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, loss_id = 0)

            self.GAN.D_opt.step()

        # setup losses
        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

        # train discriminator
        avg_pl_length = self.pl_mean
        #self.GAN.D_opt.zero_grad()
        #self.GAN.G_opt.zero_grad()  # MODIFIED
        mse = torch.nn.MSELoss(reduction='mean')  # MODIFIED
        kl = nn.KLDivLoss(reduction='batchmean')  # MODIFIED

        #for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G]):

            ### Train Discriminator ###
            #'''
            self.GAN.D_opt.zero_grad()
            self.E_opt.zero_grad()  # MODIFIED

            #get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            #style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)  # [(B, 512), 6]
            noise = image_noise(batch_size, image_size, device=self.rank)

            #w_space = latent_to_w(S, style)  # [((B, 1, 512), 6)]
            #w_styles = styles_def_to_tensor(w_space)  # (B, 6, 512)

            image_batch = next(self.loader)['x_lb'].cuda(self.rank)
            image_batch.requires_grad_()
            we = self.encoder(image_batch)  # (B, 1, 131072)
            we = we.reshape(we.shape[0], 4096, -1).mean(dim=-1)  # (B, 4096)
            w = [(we, num_layers)]  # [(B, 4096), 6]
            w = styles_def_to_tensor(w)  # (B, 6, 4096)
            #with torch.no_grad():  # MODIFIED
            wc = self.classifier(image_batch)  # MODIFIED  #(B, 512)
            wc = wc.unsqueeze(1)  # MODIFIED  #(B, 1, 512)
            real_label = self.cls_ori(image_batch)  # (B, 2)
            real_p = F.softmax(real_label, dim=-1)

            with autocast(enabled=self.fp16):
                #image_batch = next(self.loader)['x_lb'].cuda(self.rank)
                #image_batch.requires_grad_()

                #generated_images = G(w_styles, noise, wc)  # MODIFIED
                generated_images = G(w, noise, wc)  # MODIFIED
                fake_output, fake_q_loss = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)

                #image_batch = next(self.loader)['x_lb'].cuda(self.rank)
                #image_batch.requires_grad_()
                real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

                real_output_loss = real_output
                fake_output_loss = fake_output

                if self.rel_disc_loss:
                    real_output_loss = real_output_loss - fake_output.mean()
                    fake_output_loss = fake_output_loss - real_output.mean()

                divergence = D_loss_fn(real_output_loss, fake_output_loss)
                disc_loss = divergence

                if self.has_fq:
                    quantize_loss = (fake_q_loss + real_q_loss).mean()
                    self.q_loss = float(quantize_loss.detach().item())

                    disc_loss = disc_loss + quantize_loss

                if apply_gradient_penalty:
                    gp = gradient_penalty(image_batch, real_output)
                    self.last_gp_loss = gp.clone().detach().item()
                    self.track(self.last_gp_loss, 'GP')
                    disc_loss = disc_loss + gp
            #'''
            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(self.scaler, disc_loss)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

            self.d_loss = float(total_disc_loss)
            self.track(self.d_loss, 'D')

            if self.fp16:
                self.scaler.step(self.GAN.D_opt)
                self.scaler.update()
            else:
                self.GAN.D_opt.step()
            #'''
            ### Train Generator ###

            self.GAN.G_opt.zero_grad()
            self.E_opt.zero_grad()  # MODIFIED

            #style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(batch_size, image_size, device=self.rank)

            # input real images and infer wc
            with autocast(enabled=self.fp16):
                generated_images = G(w, noise, wc)  # MODIFIED

                fake_output, _ = D_aug(generated_images, **aug_kwargs)
                fake_output_loss = fake_output

                real_output = None
                if G_requires_reals:
                    image_batch = next(self.loader)['x_lb'].cuda(self.rank)
                    real_output, _ = D_aug(image_batch, detach=True, **aug_kwargs)
                    real_output = real_output.detach()

                if self.top_k_training:
                    epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                    k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                    k = math.ceil(batch_size * k_frac)

                    if k != batch_size:
                        fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

                loss = G_loss_fn(fake_output_loss, real_output)
                gen_loss = loss

                if apply_path_penalty:
                    pl_lengths = calc_pl_lengths(w, generated_images)
                    avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                    if not is_empty(self.pl_mean):
                        pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                        if not torch.isnan(pl_loss):
                            gen_loss = gen_loss + pl_loss

                #with torch.no_grad():  # MODIFIED
                if apply_we_loss:
                    weg = self.encoder(generated_images)  # (B, 1, 131072)
                    weg = weg.reshape(w.shape[0], 4096, -1).mean(dim=-1)  # (B, 4096)
                    weLoss = mse(we, weg)
                if apply_wc_loss:
                    wcg = self.classifier(generated_images)  # MODIFIED  #(B, 512)
                    wcg = wcg.unsqueeze(1)  # MODIFIED  #(B, 1, 512)
                    wcLoss = mse(wc, wcg)
                if apply_cls_loss:
                    fake_label = self.cls_ori(generated_images) # (B, 2)
                    fake_p = F.log_softmax(fake_label,dim=-1)
                    clsLoss = kl(fake_p, real_p)

                    #self.WCLoss = wcLoss.clone().detach().item()
                    #self.track(self.WCLoss, 'wcloss')

                # add conditional losses
                L1loss = L1_loss(image_batch, generated_images)  # MODIFIED
                L2loss = L2_loss(image_batch, generated_images)
                PerceptualLoss = Perceptual_Loss(3, image_batch, generated_images, rank=self.rank)  # MODIFIED
                SSIMloss = SSIM_Loss(3, image_batch, generated_images)  # MODIFIED
                MSSSIMloss = MSSSIM_Loss(3, image_batch, generated_images)

                if self.steps <= 66000:
                    gen_loss = 0.1 * gen_loss + 20 * L1loss + 2 * PerceptualLoss
                elif 66000 < self.steps <= 90000:
                    gen_loss = 0.1 * gen_loss + 10 * L1loss + 20 * L2loss + 5 * PerceptualLoss + 8 * MSSSIMloss
                elif 90000 < self.steps <= 96000:
                    gen_loss = 0.1 * gen_loss + 10 * L1loss + 20 * L2loss + 5 * PerceptualLoss + 8 * MSSSIMloss + 0.5 * weLoss + 0.5 * wcLoss
                elif self.steps > 96000:
                    gen_loss = 0.1 * gen_loss + 10 * L1loss + 20 * L2loss + 5 * PerceptualLoss + 8 * MSSSIMloss + 5 * weLoss + 5 * wcLoss + clsLoss

                gen_loss = gen_loss / self.gradient_accumulate_every
                gen_loss.register_hook(raise_if_nan)
                backwards(self.scaler, gen_loss)

                total_gen_loss += loss.detach().item() / self.gradient_accumulate_every
                total_L1_loss += L1loss.detach().item() / self.gradient_accumulate_every
                total_L2_loss += L2loss.detach().item() / self.gradient_accumulate_every
                total_SSIM_loss += SSIMloss.detach().item() / self.gradient_accumulate_every
                total_perceptual_loss += PerceptualLoss.detach().item() / self.gradient_accumulate_every
                total_MsSSIM_loss += MSSSIMloss.detach().item() / self.gradient_accumulate_every
                if apply_we_loss:
                    total_we_loss += weLoss.detach().item() / self.gradient_accumulate_every
                if apply_wc_loss:
                    total_wc_loss += wcLoss.detach().item() / self.gradient_accumulate_every
                if apply_cls_loss:
                    total_cls_loss += clsLoss.detach().item() / self.gradient_accumulate_every

                self.g_loss = float(total_gen_loss)
                self.track(self.g_loss, 'G')
                self.L1LOSS = float(total_L1_loss)
                self.track(self.g_loss, 'L1loss')
                self.PERCEPTUALLOSS = float(total_perceptual_loss)
                self.track(self.g_loss, 'Perceptualloss')

                if self.fp16:
                    self.scaler.step(self.GAN.G_opt)
                    self.scaler.step(self.E_opt)  # MODIFIED
                    self.scaler.update()
                else:
                    self.GAN.G_opt.step()
                    self.E_opt.step()

        self.L1LOSS = float(total_L1_loss)
        self.L2LOSS = float(total_L2_loss)
        self.SSIMLOSS = float(total_SSIM_loss)
        self.PERCEPTUALLOSS = float(total_perceptual_loss)
        self.MSSSIMLOSS = float(total_MsSSIM_loss)
        if apply_we_loss:
            self.WELoss = float(total_we_loss)
        if apply_wc_loss:
            self.WCLoss = float(total_wc_loss)
        if apply_cls_loss:
            self.CLSLoss = float(total_cls_loss)


        # calculate moving averages
        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors
        #print("SHOW before NaN error: ", total_gen_loss, total_disc_loss)
        #'''
        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException
        #'''

        # periodically save results
        if self.is_main:
            if self.steps % self.save_every == 0:
                print('Save checkpoint')
                self.save(self.checkpoint_num)
                torch.cuda.empty_cache()

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                torch.cuda.empty_cache()
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
                fid, medical_FID, prd_AUC = self.calculate_fid(num_batches)

                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'Step - {self.steps} -- FID: {fid} -- MD: {medical_FID} -- prd_AUC: {prd_AUC}\n')

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        print("Evaluating")
        torch.cuda.empty_cache()

        self.GAN.eval()
        self.encoder.eval()  # MODIFIED
        ext = self.image_extension
        num_rows = self.num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        with torch.no_grad():
            #latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank) # [(torch(B,512), 6)]
            n = image_noise(num_rows ** 2, image_size, device=self.rank)

            # condition
            eva_loader = MRIDataset(mode="labeled", fold=4)  # MODIFIED
            eva_cond = []
            eva_we = []
            for i in range(num_rows ** 2):
                #with torch.no_grad():  # MODIFIED
                img = eva_loader.__getitem__(i)['x_lb'].unsqueeze(0).cuda(self.rank)
                wc = self.classifier(img)  # MODIFIED
                we = self.encoder(img)
                we = we.reshape(4096, -1).mean(dim=-1)  # (B, 4096)

                eva_cond.append(wc)
                eva_we.append(we)
            eva_cond2 = torch.stack(eva_cond)  # torch(B, 1, 512)
            eva_we2 = torch.stack(eva_we).squeeze(1)  # torch(B, 512)
            latents = [(eva_we2, num_layers)]
            del eva_we2

            # regular
            torch.cuda.empty_cache()
            generated_images = self.generate_truncated(self.GAN.G, latents, n, trunc_psi = self.trunc_psi, wc = eva_cond2)
            del eva_cond2

            generated_images = generated_images[:, :, :, :, 64]
            torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
            del generated_images
            del latents

        torch.cuda.empty_cache()

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        sys.path.append(str(Path(__file__).resolve().parent))
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        real_path_3d = self.fid_dir / 'real_3d'
        fake_path_3d = self.fid_dir / 'fake_3d'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)
            rmtree(real_path_3d, ignore_errors=True)
            os.makedirs(real_path_3d)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)['x_lb']

                for k, image in enumerate(real_batch.unbind(0)):
                    # print(image.shape)
                    nifti = nib.Nifti1Image(image.squeeze(0).cpu().detach().numpy(), np.eye(4))
                    nib.save(nifti, real_path_3d / f'{str(k + batch_num * self.batch_size)}.nii.gz')

                    for i in range(image.shape[1]):
                        image_slice = image[:, i, :, :]
                        filename = str(k + batch_num * self.batch_size)+f"-{i}"
                        torchvision.utils.save_image(image_slice, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake
        rmtree(fake_path, ignore_errors=True)
        rmtree(fake_path_3d, ignore_errors=True)
        os.makedirs(fake_path)
        os.makedirs(fake_path_3d)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # Number of GPUs available. Use 0 for CPU mode.
        ngpu = 1
        # gpu number
        cuda_n = [0]

        device = torch.device("cuda:" + str(cuda_n[0]) if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                nifti = nib.Nifti1Image(image.squeeze(0).cpu().detach().numpy(), np.eye(4))#
                nib.save(nifti, fake_path_3d / f'{str(j + batch_num * self.batch_size)}.nii.gz')

                for i in range(image.shape[1]):
                    image_slice = image[:, i, :, :]
                    torchvision.utils.save_image(image_slice, str(fake_path / f'{str(j + batch_num * self.batch_size)}-f{i}-ema.{ext}'))

        pretrained_model_path = os.path.join(self.base_dir, "MedicalNet/resnet_10_23dataset.pth")
        # CUSTOM MEDICAL_NET BASED FID AND PRD CALCULATION
        # getting model
        print(f'getting pretrained model on {device} \n')
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        from MedicalNet.evaluation import generate_model
        net = generate_model()
        print('Resnet model created \n')

        # Handle multi-gpu if desired
        if (device.type == 'cuda') and (ngpu >= 1):
            net = nn.DataParallel(net, cuda_n)

        net.load_state_dict(checkpoint['state_dict'])
        print('Resnet model loaded with pretrained weights')

        # paths for real and generated features to be saved
        data_dir_real_features = os.path.join(self.base_dir, "MedicalNet", "features", "real")
        data_dir_gen_features = os.path.join(self.base_dir, "MedicalNet", "features", "fake")

        os.makedirs(data_dir_real_features, exist_ok = True)
        os.makedirs(data_dir_gen_features, exist_ok = True)

        from MedicalNet.evaluation import generate_model, process_dataset, load_features_from_npy_gz, compute_prd_from_embedding, calculate_Medical_fid

        print(f'Creating features from {fake_path_3d}')
        process_dataset(fake_path_3d, data_dir_gen_features, net)

        print(f'Creating features from {real_path_3d}')
        process_dataset(real_path_3d, data_dir_real_features, net)

        print("All features extracted with MedicalNet \n")

        real_features = load_features_from_npy_gz(data_dir_real_features)
        gen_features = load_features_from_npy_gz(data_dir_gen_features)

        # Calculate PRD
        precision, recall = compute_prd_from_embedding(gen_features, real_features, enforce_balance=False)
        sorted_indices = np.argsort(recall)
        sorted_recall = recall[sorted_indices]
        sorted_precision = precision[sorted_indices]

        # traditional FID calculation
        fid = fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

        # Medical Net based FID calculation
        fid_MedicalNET_score = calculate_Medical_fid(real_features, gen_features)

        # Calculate the AUC of PRD
        PRD_AUC = np.trapz(sorted_precision, sorted_recall)

        print("AUC of the PRD curve:", PRD_AUC)
        print(f'Medical Net based FID score: {fid_MedicalNET_score}')

        return fid, fid_MedicalNET_score, PRD_AUC

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        #S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            #tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8, wc=None):
        #w = map(lambda t: (S(t[0]), t[1]), style)
        w = map(lambda t: (t[0], t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi, wc)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise
        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 1., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for i in range(generated_images.shape[0]):  # Iterate over each volume
                single_volume = generated_images[i].squeeze().detach().cpu().numpy()  # Convert to numpy array
                nifti_img = nib.Nifti1Image(single_volume, np.eye(4))  # np.eye(4) is a placeholder for the affine matrix

                nifti_filename = os.path.join(folder_path, f'{str(num)}_{i:03d}.nii.gz')
                nib.save(nifti_img, nifti_filename)

    def print_wb(self, wbrun):
        wbrun.log({"G_loss": self.g_loss, "D_loss": self.d_loss,
                   "GP": self.last_gp_loss, "PL": self.pl_mean,
                   #"CR": self.last_cr_loss, "Q": self.q_loss,
                   #"FID": self.last_fid,
                   "L1loss": self.L1LOSS, "L2loss": self.L2LOSS, "weloss": self.WELoss, "SSIMloss": self.SSIMLOSS, "MSSSIMloss": self.MSSSIMLOSS,
                   "wcloss": self.WCLoss, "Perceptualloss": self.PERCEPTUALLOSS, "clsloss": self.CLSLoss
         })

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            #('CR', self.last_cr_loss),
            #('Q', self.q_loss),
            #('FID', self.last_fid),
            ('L1', self.L1LOSS),
            ('L2', self.L2LOSS),
            ('we', self.WELoss),
            ('wc', self.WCLoss),
            ('cls', self.CLSLoss),
            ('Per', self.PERCEPTUALLOSS),
            ('SSIM', self.SSIMLOSS),
            ('MsSSIM', self.MSSSIMLOSS)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.5f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'encoder': self.encoder.state_dict()
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()
        #torch.cuda.empty_cache()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every + 1

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
            self.encoder.load_state_dict((load_data['encoder']))
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e


class ModelLoader:
    def __init__(self, *, base_dir, name = 'default',fmap_max=512, network_capacity=16, load_from = -1):
        self.model = Trainer(name = name, base_dir = base_dir, fmap_max=fmap_max, network_capacity=network_capacity)
        self.model.load(load_from)
        fmap_max

    def noise_to_styles(self, noise, trunc_psi = None):
        noise = noise.cuda()
        w = self.model.GAN.SE(noise)

        if exists(trunc_psi):
            w = self.model.truncate_style(w, trunc_psi=trunc_psi)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device = 0)

        images = self.model.GAN.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images

    def styles_to_images_ext_noise(self, w, n):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)

        images = self.model.GAN.GE(w_tensors, n)

        images.clamp_(0., 1.)
        return images


def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)