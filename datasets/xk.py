from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2

class XKDataset(Dataset):
    def __init__(self, root_path, domain='train', image_size=(256, 256)):
        super(XKDataset, self).__init__()
        self.root_path = root_path
        self.image_size = image_size
        self.domain = domain
        self.image_dir = os.path.join(self.root_path, 'images', domain)
        self.label_dir = os.path.join(self.root_path, 'masks', domain)
        self.name_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        batch_input = {}
        image = cv2.imread(os.path.join(self.image_dir, self.name_list[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = Image.open(os.path.join(self.label_dir, self.name_list[index]))
        mask = np.array(mask)

        if self.domain == 'train':
            image, mask = self._flip(image, mask)
            image, mask = self._rotate(image, mask)

        image = Image.fromarray(np.uint8(image))
        if self.domain == 'train':
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=[0.5, 1.4], contrast=[0.6, 2.2], saturation=0, hue=0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)
        else:
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image)
        mask = torch.tensor(mask)
        # resize mask
        mask = mask.unsqueeze(0)
        mask = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.NEAREST),
        ])(mask)

        mask = mask.squeeze()
        mask = mask.long()

        batch_input['image'] = image
        batch_input['label'] = mask
        batch_input['name'] = self.name_list[index]

        return batch_input

    def _flip(self, image, mask):
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        return image, mask

    def _rotate(self, image, mask):
        if np.random.rand() > 0.5:
            angle = np.random.randint(-180, 180)
            image = Image.fromarray(image)
            mask = Image.fromarray(mask)
            image = image.rotate(angle)
            mask = mask.rotate(angle)
            image = np.array(image)
            mask = np.array(mask)
        return image, mask

    def pixel_num_per_class(self):
        pixel_num_per_class = {}
        for i in range(len(self.name_list)):
            mask = cv2.imread(os.path.join(self.label_dir, self.name_list[i]), 0)
            for j in range(0, 6):
                pixel_num_per_class[j] = pixel_num_per_class.get(j, 0) + np.sum(mask == j)
        return pixel_num_per_class


if __name__ == '__main__':
    dataset = XKDataset(root_path='./data', domain='train', image_size=(512,512))
    print(dataset.pixel_num_per_class())