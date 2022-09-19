import torch
import torchvision.transforms as T


def log_norm(x):
    """according to supplementary paper, the data is right skewd. Thus we use log transform to normalize the data.
    A constant is added in order to push all values to be > 1"""
    if ((x < 1).sum().sum()) > 0:
        c = abs(x.min()) + 1
        x = x + c
    x = torch.log(x) / torch.log(x).max()
    return x


def augmentations():
    transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(hue=.05, saturation=.05),
        T.RandomRotation(90)
    ])
    return transforms
