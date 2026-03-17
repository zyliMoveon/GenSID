import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from monai.losses import PerceptualLoss, ssim_loss
import sys
sys.path.append('../..')
from ssim import MS_SSIM


#losses
def gen_hinge_loss(fake, real):
    return -fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 - real) + F.relu(1 + fake)).mean()

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

# Add loss for training
def L1_loss(real_img, gen_img):
    # between real images and generated images
    L1 = torch.nn.L1Loss(reduction='mean')
    return L1(real_img, gen_img)

def L2_loss(real_img, gen_img):
    L2 = torch.nn.MSELoss(reduction='mean')
    return L2(real_img, gen_img)

def Perceptual_Loss(spatial_dims=3, real_img=None, gen_img=None, rank=0):
    perceptual_loss = PerceptualLoss(spatial_dims=spatial_dims).cuda(rank)
    return perceptual_loss(real_img, gen_img)

def SSIM_Loss(spatial_dims=3, real_img=None, gen_img=None):
    ssimloss = ssim_loss.SSIMLoss(spatial_dims=spatial_dims)
    return ssimloss(real_img, gen_img)

def MSSSIM_Loss(spatial_dims=3, real_img=None, gen_img=None):
    msssimloss = MS_SSIM(data_range=1, win_size=7, size_average=True, channel=1, spatial_dims=spatial_dims)
    return 1 - msssimloss(real_img, gen_img)


