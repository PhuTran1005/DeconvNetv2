import os
import torch
from PIL import Image

    
def data_augmented(voc_root):
    if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
    
    splits_dir = os.path.join(voc_root, "ImageSets", "Segmentation")

    split_f_train = os.path.join(splits_dir, "trainaug" + ".txt")
    split_f_val = os.path.join(splits_dir, "val" + ".txt")

    with open(os.path.join(split_f_train)) as f:
            file_names_train = [x.strip() for x in f.readlines()]

    with open(os.path.join(split_f_val)) as f:
            file_names_val = [x.strip() for x in f.readlines()]
    
    image_dir = os.path.join(voc_root, "JPEGImages")
    images_train = [os.path.join(image_dir, x + ".jpg") for x in file_names_train]
    images_val = [os.path.join(image_dir, x + ".jpg") for x in file_names_val]

    target_dir = os.path.join(voc_root, "SegmentationClassAugRaw")
    targets_train = [os.path.join(target_dir, x + ".png") for x in file_names_train]
    targets_val = [os.path.join(target_dir, x + ".png") for x in file_names_val]

    assert len(images_train) == len(targets_train)
    assert len(images_val) == len(targets_val)


    return images_train, images_val, targets_train, targets_val


class VOCSegmentation(torch.utils.data.Dataset):
    def __init__(self, images, masks, transforms = None):
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target