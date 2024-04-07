import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image
import logging
import copy
import importlib
import json
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks import *
import data_utils
from functions import * 
from hrnetv2_config import _C   
from hrnetv2 import HighResolutionNet

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if not len(sys.argv) == 2:
    print('Format:')
    print('python lib/train_semi_str.py config_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

my_config = importlib.import_module(config_path, package='STLD')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists('./snapshots'):
    os.mkdir('./snapshots')
if not os.path.exists(os.path.join('./snapshots', cfg.data_name)):
    os.mkdir(os.path.join('./snapshots', cfg.data_name))
save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists('./logs'):
    os.mkdir('./logs')
if not os.path.exists(os.path.join('./logs', cfg.data_name)):
    os.mkdir(os.path.join('./logs', cfg.data_name))
log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

print('###########################################')
print('experiment_name:', cfg.experiment_name)
print('data_name:', cfg.data_name)
print('labeled_num:', cfg.labeled_num)
print('os_num:', cfg.os_num)
print('semi_iter:', cfg.semi_iter)
print('det_head:', cfg.det_head)
print('net_stride:', cfg.net_stride)
print('input_size:', cfg.input_size)
print('batch_size:', cfg.batch_size)
print('init_lr:', cfg.init_lr)
print('num_epochs:', cfg.num_epochs)
print('decay_steps:', cfg.decay_steps)
print('backbone:', cfg.backbone)
print('num_lms:', cfg.num_lms)
print('use_gpu:', cfg.use_gpu)
print('gpu_id:', cfg.gpu_id)
print('###########################################')
logging.info('###########################################')
logging.info('experiment_name: {}'.format(cfg.experiment_name))
logging.info('data_name: {}'.format(cfg.data_name))
logging.info('labeled_num: {}'.format(cfg.labeled_num))
logging.info('semi_iter: {}'.format(cfg.semi_iter))
logging.info('det_head: {}'.format(cfg.det_head))
logging.info('net_stride: {}'.format(cfg.net_stride))
logging.info('input_size: {}'.format(cfg.input_size))
logging.info('batch_size: {}'.format(cfg.batch_size))
logging.info('init_lr: {}'.format(cfg.init_lr))
logging.info('num_epochs: {}'.format(cfg.num_epochs))
logging.info('decay_steps: {}'.format(cfg.decay_steps))
logging.info('backbone: {}'.format(cfg.backbone))
logging.info('num_lms: {}'.format(cfg.num_lms))
logging.info('use_gpu: {}'.format(cfg.use_gpu))
logging.info('gpu_id: {}'.format(cfg.gpu_id))
logging.info('###########################################')

if cfg.det_head == 'map':
    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=True)
        net = Map_resnet(resnet18, cfg)
    elif cfg.backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=True)
        net = Map_resnet(resnet50, cfg)
    elif cfg.backbone == 'resnet101':
        resnet101 = models.resnet101(pretrained=True)
        net = Map_resnet(resnet101, cfg)
    elif cfg.backbone == 'hrnet':
        _C.MODEL.NUM_JOINTS = cfg.num_lms
        hrnet = HighResolutionNet(_C)
        hrnet.init_weights('./models/pytorch/imagenet/hrnetv2_w18_imagenet_pretrained.pth')
        net = Map_hrnet(hrnet, cfg)
    else:
        print('No such backbone!')
        exit(0)
elif cfg.det_head == 'tf':
    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=True)
        net = TF_resnet(resnet18, cfg)
    elif cfg.backbone == 'resnet50':
        resnet50 = models.resnet50(pretrained=True)
        net = TF_resnet(resnet50, cfg)
    elif cfg.backbone == 'resnet101':
        resnet101 = models.resnet101(pretrained=True)
        net = TF_resnet(resnet101, cfg)
    elif cfg.backbone == 'hrnet':
        _C.MODEL.NUM_JOINTS = cfg.num_lms
        hrnet = HighResolutionNet(_C)
        hrnet.init_weights('./models/pytorch/imagenet/hrnetv2_w18_imagenet_pretrained.pth')
        net = TF_hrnet(hrnet, cfg)
    else:
        print('No such backbone!')
        exit(0)
else:
    print('No such head:', cfg.det_head)
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

criterion_cls = nn.MSELoss(reduction='sum')
criterion_reg = nn.L1Loss(reduction='sum')

points_flip = None
if cfg.data_name == 'data_300W':
    points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 68
elif cfg.data_name == 'WFLW':
    points_flip = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
    assert len(points_flip) == 98
elif cfg.data_name == 'AFLW':
    points_flip = [6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 15, 14, 13, 18, 17, 16, 19]
    points_flip = (np.array(points_flip)-1).tolist()
    assert len(points_flip) == 19
else:
    print('No such data!')
    exit(0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

labeled_num_str = str(cfg.labeled_num)
file_name_l = 'train_semi_l_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'
file_name_u = 'train_semi_u_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'
labels_train_l = get_label(cfg.data_name, file_name_l, 'gt')
labels_train_u = get_label(cfg.data_name, file_name_u)
labels_val = get_label(cfg.data_name, 'test.txt', 'gt')

labels_train_l_origin = copy.deepcopy(labels_train_l)
# if labeled number less than os_num, oversample it to os_num for better performance
sample_num = cfg.os_num
sample_thresh = cfg.os_num
if len(labels_train_l) < sample_thresh:
    sample_times = ceil(1. * sample_num / len(labels_train_l))
    labels_train_l = labels_train_l * sample_times
    labels_train_l = labels_train_l[:sample_num]
    print('oversampled to {}'.format(sample_num))
    logging.info('oversampled to {}'.format(sample_num))

if cfg.det_head == 'tf':
    map_size = None
else:
    map_size = get_map_size(net, cfg.input_size, device)

data_train_u = data_utils.ImageFolder_u(cfg, 'train', 
                                        os.path.join('data', cfg.data_name, 'images_train'), 
                                        labels_train_u, 
                                        map_size,
                                        transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize]))

if cfg.det_head == 'map':
    data_val = data_utils.ImageFolder_map(cfg, 'val',
                                          os.path.join('data', cfg.data_name, 'images_test'),
                                          labels_val, 
                                          map_size,
                                          points_flip,
                                          transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize]))
elif cfg.det_head == 'tf':
    data_val = data_utils.ImageFolder_tf(cfg, 'val',
                                         os.path.join('data', cfg.data_name, 'images_test'), 
                                         labels_val, 
                                         map_size,
                                         points_flip,
                                         transforms.Compose([
                                         transforms.ToTensor(),
                                         normalize]))
else:
    print('No such head:', cfg.det_head)
    exit(0)

loader_train_u = torch.utils.data.DataLoader(data_train_u, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
loader_val = torch.utils.data.DataLoader(data_val, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)

###############
# the norm for test
norm_indices = None
if cfg.data_name == 'data_300W':
    norm_indices = [36, 45]
elif cfg.data_name == 'WFLW':
    norm_indices = [60, 72]
elif cfg.data_name == 'AFLW':
    pass
else:
    print('No such data!')
    exit(0)

pseudo_labels_pp = []
pseudo_labels_sl = []
for ti in range(cfg.semi_iter):
    print('###################################################')
    print('Starting iter {}'.format(ti))
    logging.info('Starting iter {}'.format(ti))

    if cfg.det_head == 'map':
        if cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=True)
            net = Map_resnet(resnet18, cfg)
        elif cfg.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)
            net = Map_resnet(resnet50, cfg)
        elif cfg.backbone == 'resnet101':
            resnet101 = models.resnet101(pretrained=True)
            net = Map_resnet(resnet101, cfg)
        elif cfg.backbone == 'hrnet':
            _C.MODEL.NUM_JOINTS = cfg.num_lms
            hrnet = HighResolutionNet(_C)
            hrnet.init_weights('./models/pytorch/imagenet/hrnetv2_w18_imagenet_pretrained.pth')
            net = Map_hrnet(hrnet, cfg)
        else:
            print('No such backbone!')
            exit(0)
    elif cfg.det_head == 'tf':
        if cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=True)
            net = TF_resnet(resnet18, cfg)
        elif cfg.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)
            net = TF_resnet(resnet50, cfg)
        elif cfg.backbone == 'resnet101':
            resnet101 = models.resnet101(pretrained=True)
            net = TF_resnet(resnet101, cfg)
        elif cfg.backbone == 'hrnet':
            _C.MODEL.NUM_JOINTS = cfg.num_lms
            hrnet = HighResolutionNet(_C)
            hrnet.init_weights('./models/pytorch/imagenet/hrnetv2_w18_imagenet_pretrained.pth')
            net = TF_hrnet(hrnet, cfg)
        else:
            print('No such backbone!')
            exit(0)
    else:
        print('No such head:', cfg.det_head)
        exit(0)
    net = net.to(device)

    #####################################
    if len(pseudo_labels_pp) > 0:
        print('pseudo pretraining...')
        logging.info('pseudo pretraining...')

        if cfg.det_head == 'map':
            data_train_pp = data_utils.ImageFolder_map(cfg, 'train',
                                                       os.path.join('data', cfg.data_name, 'images_train'), 
                                                       pseudo_labels_pp, 
                                                       map_size,
                                                       points_flip,
                                                       transforms.Compose([
                                                       transforms.RandomGrayscale(0.2),
                                                       transforms.ToTensor(),
                                                       normalize]))
        elif cfg.det_head == 'tf':
            data_train_pp = data_utils.ImageFolder_tf(cfg, 'train',
                                                      os.path.join('data', cfg.data_name, 'images_train'), 
                                                      pseudo_labels_pp, 
                                                      map_size,
                                                      points_flip,
                                                      transforms.Compose([
                                                      transforms.RandomGrayscale(0.2),
                                                      transforms.ToTensor(),
                                                      normalize]))
        else:
            print('No such head:', cfg.det_head)
            exit(0)

        loader_train_pp = torch.utils.data.DataLoader(data_train_pp, batch_size=cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)
        optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
        if ti <= 1:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)
            # train with pseudo labels
            net = train_model(cfg, cfg.num_epochs, net, loader_train_pp, loader_val, criterion_cls, criterion_reg, optimizer, scheduler, os.path.join(save_dir, 'last_pseudo_pre.pth'), norm_indices, ti, device)
        else:
            # only 1/5 epoch for pseudo pretraining
            state_dict = torch.load(os.path.join(save_dir, 'last_pseudo_pre.pth'))
            net.load_state_dict(state_dict)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[s//5 for s in cfg.decay_steps], gamma=0.1)
            # train with pseudo labels
            net = train_model(cfg, cfg.num_epochs//5, net, loader_train_pp, loader_val, criterion_cls, criterion_reg, optimizer, scheduler, os.path.join(save_dir, 'last_pseudo_pre.pth'), norm_indices, ti, device)

    if ti <= 1:
        # no shrink regression
        labels_train = labels_train_l  
    elif ti <= 3:
        # with shrink regression
        labels_train = labels_train_l + pseudo_labels_sl
    else:
        # with shrink regression
        labels_train = labels_train_l_origin + pseudo_labels_sl

    
    if cfg.det_head == 'map':
        data_train = data_utils.ImageFolder_map(cfg, 'train',
                                                os.path.join('data', cfg.data_name, 'images_train'), 
                                                labels_train, 
                                                map_size,
                                                points_flip,
                                                transforms.Compose([
                                                transforms.RandomGrayscale(0.2),
                                                transforms.ToTensor(),
                                                normalize]))
    elif cfg.det_head == 'tf':
        data_train = data_utils.ImageFolder_tf(cfg, 'train',
                                               os.path.join('data', cfg.data_name, 'images_train'), 
                                               labels_train, 
                                               map_size,
                                               points_flip,
                                               transforms.Compose([
                                               transforms.RandomGrayscale(0.2),
                                               transforms.ToTensor(),
                                               normalize]))
    else:
        print('No such head:', cfg.det_head)
        exit(0)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=False)

    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)
    # train with gt+pseudo data
    net = train_model(cfg, cfg.num_epochs, net, loader_train, loader_val, criterion_cls, criterion_reg, optimizer, scheduler, os.path.join(save_dir, 'last.pth'), norm_indices, ti, device)
    #####################################

    ###############
    # test
    print('After iter {}'.format(ti))
    logging.info('After iter {}'.format(ti))
    nme_mean, fr, auc = val_model(cfg, net, loader_val, norm_indices, device)
    print('nme: {}, fr: {}, auc: {}'.format(nme_mean, fr, auc))
    logging.info('nme: {}, fr: {}, auc: {}'.format(nme_mean, fr, auc))
    
    ###############
    # estimate pseudo labels
    print('Estimating pseudo-labels...')
    logging.info('Estimating pseudo-labels...')
    pseudo_labels_sl = gen_pseudo(cfg, net, loader_train_u, 'pseudo', ti, device)
    pseudo_labels_pp = [[p[0], 'gt', p[2], p[3]] for p in pseudo_labels_sl]

