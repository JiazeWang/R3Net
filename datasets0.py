import os
import os.path

import torch.utils.data as data
from PIL import Image

filename_input = "sense_train3.txt"
with open(filename_input) as f:
    lines = f.readlines()
folders = []
for line in lines:
    line = line.rstrip()
    folders.append(line)


def make_dataset():
    img = []
    ground = []
    for line in folders:
        img_list = [os.path.splitext(f)[0] for f in os.listdir(line) if f.startswith('frame_')]
        line_train = [(os.path.join(line, img_name + '.png')) for img_name in img_list]
        line_ground = [(os.path.join(line[0:-6], 'mask','mask_'+ img_name[6:] + '.png')) for  img_name in img_list]
        img = img + line_train
        ground = ground + line_ground
    return img, ground


class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root = root
        self.imgs, self.grounds = make_dataset()
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        gt_path = self.grounds[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
