import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.transform = transform
        self.image_path = image_path
        self.mask_path = mask_path
        self.images = sorted(os.listdir(image_path))
        self.masks = sorted(os.listdir(mask_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_path, self.images[index])
        mask_path = os.path.join(self.mask_path, self.masks[index])
        #print(image_path, mask_path) #Images should be paired with corresponding mask, uncomment the line and run train.py to confirm in the terminal
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augments = self.transform(image=image, mask=mask)
            image = augments['image']
            mask = augments['mask']

        return image, mask