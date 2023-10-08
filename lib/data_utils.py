import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFilter 
import os, cv2
import numpy as np
from numpy import cumsum
import random
from scipy.stats import norm
from math import floor
import copy

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

def random_translate(image, target):
    if random.random() > 0.5:
        image_height, image_width = image.size
        a = 1
        b = 0
        c = int((random.random()-0.5) * 60)
        d = 0
        e = 1
        f = int((random.random()-0.5) * 60)
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        target_translate = target.copy()
        target_translate = target_translate.reshape(-1, 2)
        target_translate[:, 0] -= 1.*c/image_width
        target_translate[:, 1] -= 1.*f/image_height
        target_translate = target_translate.flatten()
        return image, target_translate
    else:
        return image, target

def random_blur(image):
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.random()*2))
    return image

def random_occlusion(image):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:,:,::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height*0.4*random.random())
        occ_width = int(image_width*0.4*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np[:,:,::-1].astype('uint8'), 'RGB')
        return image_pil
    else:
        return image

def random_flip(image, target, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target = np.array(target).reshape(-1, 2)
        target = target[points_flip, :]
        target[:,0] = 1-target[:,0]
        target = target.flatten()
        return image, target
    else:
        return image, target

def random_rotate(image, target, angle_max):
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_num= int(len(target) / 2)
        target_center = np.array(target) - np.array([center_x, center_y]*landmark_num)
        target_center = target_center.reshape(landmark_num, 2)
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c,-s), (s, c)))
        target_center_rot = np.matmul(target_center, rot)
        target_rot = target_center_rot.reshape(landmark_num*2) + np.array([center_x, center_y]*landmark_num)
        return image, target_rot
    else:
        return image, target

def gen_target_map(target, target_map, sigma):
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]
    tmp_size = sigma * 3
    for i in range(map_channel):
        mu_x = int(target[i][0] * map_width - 0.5)
        mu_y = int(target[i][1] * map_height - 0.5)
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width-1)
        mu_y = min(mu_y, map_height-1)
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size), int(mu_y + tmp_size)]
        ul[0] = max(0, ul[0])
        ul[1] = max(0, ul[1])
        br[0] = min(br[0], map_width-1)
        br[1] = min(br[0], map_height-1)
        margin_left = int(min(mu_x, tmp_size))
        margin_right = int(min(map_width-1-mu_x, tmp_size))
        margin_top = int(min(mu_y, tmp_size))
        margin_bottom = int(min(map_height-1-mu_y, tmp_size))
        assert margin_right >= -margin_left
        assert margin_bottom >= -margin_top

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = int(size // 2)
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        target_map[i, (mu_y-margin_top):(mu_y+margin_bottom+1), (mu_x-margin_left):(mu_x+margin_right+1)] = g[(y0-margin_top):(y0+margin_bottom+1), (x0-margin_left):(x0+margin_right+1)]
    return target_map

class ImageFolder_map(data.Dataset):
    def __init__(self, cfg, phase, root, labels, map_size, points_flip, transform=None):
        self.cfg = cfg
        self.phase = phase
        self.root = root
        self.labels = labels
        self.map_size = map_size
        self.points_flip = points_flip
        self.transform = transform
        self.num_lms = cfg.num_lms
        self.net_stride = cfg.net_stride
        self.input_size = cfg.input_size
        self.gt_sigma = cfg.gt_sigma

    def __getitem__(self, index):
        img_name, data_type, cur_iter, target = self.labels[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.phase == 'train':
            img, target = random_translate(img, target)
            img = random_occlusion(img)
            img, target = random_flip(img, target, self.points_flip)
            img, target = random_rotate(img, target, 30)
            img = random_blur(img)

            target_map = np.zeros((self.num_lms, self.map_size[2], self.map_size[3]))
            target_map2 = np.zeros((self.num_lms, self.map_size[2], self.map_size[3]))

            mask_map = np.ones_like(target_map)
            mask_map2 = np.ones_like(target_map2)
            target_map = gen_target_map(target, target_map, self.gt_sigma)

            if data_type == 'gt':
                mask_map2 = np.zeros_like(target_map2)
                if self.cfg.shrink_curri:
                    sigma_cur = self.cfg.shrink_curri[cur_iter]
                    if sigma_cur < self.gt_sigma:
                        target_map = np.zeros((self.num_lms, self.map_size[2], self.map_size[3]))
                        target_map = gen_target_map(target, target_map, sigma_cur)
            elif data_type == 'pseudo':
                sigma_cur = self.cfg.shrink_curri[cur_iter]
                if sigma_cur < self.gt_sigma: 
                    target_map = np.zeros((self.num_lms, self.map_size[2], self.map_size[3]))
                    target_map = gen_target_map(target, target_map, sigma_cur)
                    mask_map2 = np.zeros_like(target_map2)
                else:
                    target_map2 = gen_target_map(target, target_map2, sigma_cur)
                    mask_map = np.zeros_like(target_map)
            else:
                print('No such data type:', data_type)
                exit(0)

            target_map = torch.from_numpy(target_map).float()
            target_map2 = torch.from_numpy(target_map2).float()

            mask_map = torch.from_numpy(mask_map).float()
            mask_map2 = torch.from_numpy(mask_map2).float()

            if self.transform is not None:
                img = self.transform(img)

            return img, target_map, target_map2, mask_map, mask_map2 
        # for val
        else:
            target = torch.from_numpy(target).float()

            if self.transform is not None:
                img = self.transform(img)

            return img, target

    def __len__(self):
        return len(self.labels)

class ImageFolder_tf(data.Dataset):
    def __init__(self, cfg, phase, root, labels, map_size, points_flip, transform=None):
        self.cfg = cfg
        self.phase = phase
        self.root = root
        self.labels = labels
        self.points_flip = points_flip
        self.transform = transform
        self.num_lms = cfg.num_lms
        self.net_stride = cfg.net_stride
        self.input_size = cfg.input_size

    def __getitem__(self, index):
        img_name, data_type, cur_iter, target = self.labels[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')
        if self.phase == 'train':
            img, target = random_translate(img, target)
            img = random_occlusion(img)
            img, target = random_flip(img, target, self.points_flip)
            img, target = random_rotate(img, target, 30)
            img = random_blur(img)

            exponents = np.ones(self.num_lms*2)
            exponents2 = np.ones(self.num_lms*2)

            mask = np.ones_like(target)
            mask2 = np.ones_like(target)

            if data_type == 'gt':
                mask2 = np.zeros_like(target)
            elif data_type == 'pseudo':
                exp_cur = self.cfg.shrink_curri[cur_iter]
                if exp_cur == 1:
                    mask2 = np.zeros_like(target)
                else:
                    exponents2 *= exp_cur
                    mask = np.zeros_like(target)

            target = torch.from_numpy(target).float().view(-1, 2)
            exponents = torch.from_numpy(exponents).float().view(-1, 2)
            exponents2 = torch.from_numpy(exponents2).float().view(-1, 2)
            mask = torch.from_numpy(mask).float().view(-1, 2)
            mask2 = torch.from_numpy(mask2).float().view(-1, 2)

            if self.transform is not None:
                img = self.transform(img)

            return img, target, exponents, exponents2, mask, mask2
        # for val
        else:
            target = torch.from_numpy(target).float().view(-1, 2)

            if self.transform is not None:
                img = self.transform(img)

            return img, target

    def __len__(self):
        return len(self.labels)

# loading unlabeled train data
class ImageFolder_u(data.Dataset):
    def __init__(self, cfg, phase, root, labels, map_size, transform=None):
        self.cfg = cfg  
        self.phase = phase  
        self.root = root
        self.labels = labels
        self.map_size = map_size
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_name = self.labels[index][0]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, img_name

    def __len__(self):
        return len(self.labels)

