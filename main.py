import train
from data import augmentations

import torch
import torchvision

import matplotlib.pyplot as plt
import argparse

# Ignore Warning
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2, help='batch size for training phase')
    parser.add_argument('--batch-size-test', type=int, default=1, help='batch size for test phase (always 1)')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID (0, 1, 2, ...)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--is-skip-connect', type=bool, default=True, help='Number of epochs for training')
    parser.add_argument('--is-attention', type=bool, default=True, help='Number of epochs for training')
    opt = parser.parse_args()

    # hyp params
    TRAIN_BATCH_SIZE = opt.batch_size
    TEST_BATCH_SIZE = opt.batch_size_test
    num_classes = 21
    ignore_index = 255

    gpu_id = opt.gpu_id
    print_freq = 1
    epoch_print = 1

    save = True
    epochs = opt.epochs

    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    train_tf = augmentations.Mask_Aug(transforms=[augmentations.ToTensor(), augmentations.PILToTensor(), 
                                                augmentations.Resize((256, 256)), augmentations.RandomCrop((224, 224)), 
                                                augmentations.RandomHorizontalFlip(),
                                                augmentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    val_tf = augmentations.Mask_Aug(transforms=[augmentations.ToTensor(), augmentations.PILToTensor(), 
                                            augmentations.Resize((256, 256)), 
                                            augmentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.VOCSegmentation(root='./', year='2012', image_set='train', download=True, transforms=train_tf)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = torchvision.datasets.VOCSegmentation(root='./', year='2012', image_set='val', download=True, transforms=val_tf)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)

    deconvnet_v2 = train.DeconvNetv2(num_classes=num_classes, ignore_index=ignore_index, 
                                gpu_id=gpu_id, print_freq=print_freq, epoch_print=epoch_print, is_sk=opt.is_skip_connect, is_attent=opt.is_attention)

    deconvnet_v2.train(train_loader, val_loader, save=save, epochs=epochs, lr=lr, momentum=momentum, weight_decay=weight_decay)