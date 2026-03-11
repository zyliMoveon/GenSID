import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
import numpy as np
import torch
import pickle
import monai.transforms as mtransforms
from monai.data import CacheDataset
import numpy as np

def TransformAug(mode='Weak'):
    if mode=='Weak':
        return mtransforms.OneOf(
                [
                mtransforms.Compose([]),
                # randflip
                # mtransforms.RandFlip(prob=1,spatial_axis=0),
                # mtransforms.RandFlip(prob=1,spatial_axis=1),
                # mtransforms.RandFlip(prob=1,spatial_axis=2),
                # mtransforms.RandFlip(prob=1),
                # randtranslate
                mtransforms.RandAffine(prob=1.0,translate_range=(30, 0, 0),padding_mode='border'),
                mtransforms.RandAffine(prob=1.0,translate_range=(0, 30, 0),padding_mode='border'),
                mtransforms.RandAffine(prob=1.0,translate_range=(0, 0, 30),padding_mode='border'),
                # mtransforms.RandAffine(prob=1.0,padding_mode='border'),
                ]
                , weights=(0.5,0.167,0.167,0.166)
            )
    elif mode=='Strong':
        return mtransforms.OneOf(
                [
                mtransforms.Compose([]),
                # flip
                # mtransforms.RandFlip(prob=1),
                # translate
                mtransforms.RandAffine(prob=1.0, translate_range=(30, 0, 0), padding_mode='border'),
                mtransforms.RandAffine(prob=1.0, translate_range=(0, 30, 0), padding_mode='border'),
                mtransforms.RandAffine(prob=1.0, translate_range=(0, 0, 30), padding_mode='border'),
                # mtransforms.RandAffine(prob=1.0,padding_mode='border'),
                # rotate
                # mtransforms.RandAffine(prob=1.0,rotate_range=(np.pi / 6,np.pi / 6,np.pi / 6)
                #                 ,padding_mode='border'),
                # contrast
                mtransforms.RandAdjustContrast(prob=1.0,gamma=3),
                # gaussian noise
                mtransforms.RandGaussianNoise(prob=1.0,mean=0,std=0.15),
                # gaussian smooth
                mtransforms.RandGaussianSmooth(prob=1.0,sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)),
                # sharpen
                mtransforms.RandGaussianSharpen(prob=1.0,sigma1_x=(0.25, 1.5), sigma1_y=(0.25, 1.5), sigma1_z=(0.25, 1.5), sigma2_x=1, sigma2_y=1, sigma2_z=1, alpha=(5.0, 10.0)),
                # Gibbs
                mtransforms.RandGibbsNoise(prob=1.0,alpha=(0.5,0.7)),
                # Rissan
                mtransforms.RandRicianNoise(prob=1.0,mean=0,std=0.2),
                # Shift
                # mtransforms.RandShiftIntensity(prob=1.0,offsets=0.2),
                # std shift
                # mtransforms.RandStdShiftIntensity(prob=1.0,factors=(0.8,1.2)),
                ],
                weights=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1)
            )


class MRIDataset(torch.utils.data.Dataset):
    """
    Dataset class

    Parameters:
        root - - : (str) Path of the root folder
        mode - - : (str) {'train' or 'test'} Part of the dataset that is loaded
        seed - - : (int) random seed
    """

    def __init__(self,
                 mode='labeled',
                 fold=4,
                 ):
        # basic initialize
        self.mode = mode

        self.datapath='/your/data/path'
        self.nc_pairs= pickle.load(open(self.datapath+'/your_he.pkl', 'rb'))
        self.ad_pairs= pickle.load(open(self.datapath+'/your_di.pkl', 'rb'))
        self.w_transforms = TransformAug(mode='Weak')
        self.s_transforms = TransformAug(mode='Strong')
        self.basic_transform= mtransforms.Compose(
            [mtransforms.LoadImage(image_only=True),
            mtransforms.EnsureChannelFirst(),
            mtransforms.SqueezeDim(),
            mtransforms.EnsureChannelFirst(),
            mtransforms.EnsureType(),
            mtransforms.Orientation(axcodes="RAI"),
            mtransforms.ScaleIntensityRangePercentiles(lower=0, upper=99, b_min=0.0, b_max=1.0,clip=True,relative=False),
            # mtransforms.CenterSpatialCrop(roi_size=(188, 224, 188)),
            # mtransforms.SpatialPad(spatial_size=(188, 188, 188), mode='minimum'),
            mtransforms.Resize(spatial_size=(128, 128, 128)),
            ]
            )
        self.imgs = []
        self.labels=[]

        if mode=='labeled':
            for key in list(set([0,1,2,3,4])-set([fold])):
                # nc data
                self.imgs += self.nc_pairs[key]
                self.labels += [0] * len(self.nc_pairs[key])
                self.imgs += self.ad_pairs[key]
                self.labels += [1] * len(self.ad_pairs[key])
        elif mode=='test':
            self.imgs += self.nc_pairs[fold]
            self.labels += [0] * len(self.nc_pairs[fold])
            self.imgs += self.ad_pairs[fold]
            self.labels += [1] * len(self.ad_pairs[fold])
            # self.imgs += self.ad_pairs[fold]
        elif mode=='labeled_test':
            for key in list(set([0,1,2,3,4])-set([fold])):
                # nc data
                self.imgs += self.nc_pairs[key]
                self.labels += [0] * len(self.nc_pairs[key])
                self.imgs += self.ad_pairs[key]
                self.labels += [1] * len(self.ad_pairs[key])


    def __getitem__(self, index):
        # extract specific data <important part>
        data_path = self.imgs[index]
        value = self.labels[index]
        A = self.basic_transform(data_path)

        if self.mode=='labeled':
            return {'idx_lb': index, 'x_lb': self.w_transforms(A), 'y_lb': value}
        elif self.mode=='test':
            return {'idx_lb': index, 'x_lb': A, 'y_lb': value}
        elif self.mode=='labeled_test':
            return {'idx_lb': index, 'x_lb': A, 'y_lb': value}

    def __len__(self):
        #  the length of dataset
        return len(self.imgs)
