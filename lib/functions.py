import os, cv2
import numpy as np
from numpy import cumsum
from PIL import Image, ImageFilter
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import time
import math
from math import floor
from scipy.io import loadmat, savemat
from collections import OrderedDict
import json
from scipy.integrate import simps

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

def get_label(data_name, label_file, data_type=None, cur_iter=-1):
    label_path = os.path.join('data', data_name, label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0])==1:
        return labels

    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if data_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, data_type, cur_iter, target])
    return labels_new

def get_map_size(net, input_size, device):
    net.eval()
    with torch.no_grad():
        data_tmp = torch.rand(1, 3, input_size[0], input_size[1])
        data_tmp = data_tmp.to(device)
        out = net(data_tmp)
    return list(out[0].size())

def compute_loss_tf(outputs, outputs2, labels, criterion, exponents, exponents2, masks, masks2, cfg):
    loss = torch.mean(torch.pow(torch.abs(outputs-labels)*masks, exponents)*1./exponents)
    loss2 = torch.mean(torch.pow(torch.abs(outputs2-labels)*masks2, exponents2)*1./exponents2)
    return loss, loss2

def compute_loss_map(outputs_map, outputs_map2, labels_map, labels_map2, masks_map, masks_map2, criterion_cls):
    loss_map = criterion_cls(outputs_map*masks_map, labels_map*masks_map)
    if not masks_map.sum() == 0:
        loss_map /= masks_map.sum()

    loss_map2 = criterion_cls(outputs_map2*masks_map2, labels_map2*masks_map2)
    if not masks_map2.sum() == 0:
        loss_map2 /= masks_map2.sum()

    return loss_map, loss_map2

def train_model(cfg, num_epochs, net, loader_train, loader_val, criterion_cls, criterion_reg, optimizer, scheduler, save_path, norm_indices, cur_iter, device):
    for epoch in range(num_epochs):
        net.train()
        for i, data in enumerate(loader_train):
            if cfg.det_head == 'map':
                inputs, labels_map, labels_map2, masks_map, masks_map2 = data
                inputs = inputs.to(device)
                labels_map = labels_map.to(device)
                labels_map2 = labels_map2.to(device)
                masks_map = masks_map.to(device)
                masks_map2 = masks_map2.to(device)
                outputs_map, outputs_map2 = net(inputs)
                loss_map, loss_map2 = compute_loss_map(outputs_map, outputs_map2, labels_map, labels_map2, masks_map, masks_map2, criterion_cls)
                loss = loss_map + cfg.shrink_weight_curri[cur_iter]*loss_map2 
            elif cfg.det_head == 'tf':
                inputs, labels, exponents, exponents2, masks, masks2 = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                exponents = exponents.to(device)
                exponents2 = exponents2.to(device)
                masks = masks.to(device)
                masks2 = masks2.to(device)
                outputs, outputs2, atten_weights_list, self_atten_weights_list = net(inputs)
                loss1, loss2 = compute_loss_tf(outputs, outputs2, labels, criterion_reg, exponents, exponents2, masks, masks2, cfg)
                loss = loss1 + cfg.shrink_weight_curri[cur_iter]*loss2
            else:
                print('No such head:', det_head)
                exit(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                if cfg.det_head == 'map':
                    print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <map2 loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(loader_train)-1, loss.item(), loss_map.item(), cfg.shrink_weight_curri[cur_iter]*loss_map2.item()))
                    logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <map loss: {:.6f}> <map2 loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(loader_train)-1, loss.item(), loss_map.item(), cfg.shrink_weight_curri[cur_iter]*loss_map2.item()))
                elif cfg.det_head == 'tf':
                    print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <loss1: {:.6f}> <loss2: {:.6f}>'.format(epoch, num_epochs-1, i, len(loader_train)-1, loss1.item(), cfg.shrink_weight_curri[cur_iter]*loss2.item()))
                    logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <loss1: {:.6f}> <loss2: {:.6f}>'.format(epoch, num_epochs-1, i, len(loader_train)-1, loss1.item(), cfg.shrink_weight_curri[cur_iter]*loss2.item()))
                else:
                    print('No such head:', cfg.det_head)
                    exit(0)
        
        scheduler.step()
    torch.save(net.state_dict(), save_path)
    print('saving model to {}'.format(save_path))
    logging.info('saving model to {}'.format(save_path))
    return net

def val_model(cfg, net, loader_val, norm_indices, device):
    nme_list = []
    for i, data in enumerate(loader_val):
        if cfg.det_head == 'map':
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            tmp_x, tmp_y, outputs, max_cls = forward_map(net, inputs, cfg.input_size, cfg.net_stride)
            tmp_batch, tmp_channel, tmp_height, tmp_width = outputs.size()
            tmp_x = tmp_x.view(tmp_batch, tmp_channel, 1)
            tmp_y = tmp_y.view(tmp_batch, tmp_channel, 1)
            outputs = torch.cat((tmp_x, tmp_y), 2)

            labels = labels.view(tmp_batch, tmp_channel, 2)

            nmes = compute_nme(outputs, labels, norm_indices)
            nme_list.extend(nmes)
        elif cfg.det_head == 'tf':
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _, _, _ = forward_tf(net, inputs)
            nmes = compute_nme(outputs, labels, norm_indices)
            nme_list.extend(nmes)
        else:
            print('No such head:', cfg.det_head)
            exit(0)
    nme_mean = np.mean(nme_list)
    fr, auc = compute_fr_and_auc(nme_list)
    return nme_mean, fr, auc
    
def forward_map(net, inputs, input_size, net_stride):
    net.eval()
    with torch.no_grad():
        outputs, _ = net(inputs)
        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs.size()

        outputs = outputs.reshape(tmp_batch*tmp_channel, -1)
        max_ids = torch.argmax(outputs, 1)
        max_cls = torch.max(outputs, 1)[0].view(tmp_batch, tmp_channel)
        max_ids = max_ids.view(-1, 1)

        outputs = outputs.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
        outputs_sub_r = outputs[:,:,:,2:]
        outputs_sub_l = outputs[:,:,:,:-2]
        outputs_sub_b = outputs[:,:,2:,:]
        outputs_sub_t = outputs[:,:,:-2,:]
        pad_x = nn.ZeroPad2d((1, 1, 0, 0))
        pad_y = nn.ZeroPad2d((0, 0, 1, 1))

        outputs_shift_x = pad_x(outputs_sub_r - outputs_sub_l)
        outputs_shift_x = outputs_shift_x.view(tmp_batch*tmp_channel, -1)
        outputs_shift_x_select = torch.gather(outputs_shift_x, 1, max_ids)
        outputs_shift_x_select = outputs_shift_x_select.squeeze(1)
        outputs_shift_x_select = outputs_shift_x_select.sign() * 0.25

        outputs_shift_y = pad_y(outputs_sub_b - outputs_sub_t)
        outputs_shift_y = outputs_shift_y.view(tmp_batch*tmp_channel, -1)
        outputs_shift_y_select = torch.gather(outputs_shift_y, 1, max_ids)
        outputs_shift_y_select = outputs_shift_y_select.squeeze(1)
        outputs_shift_y_select = outputs_shift_y_select.sign() * 0.25

        tmp_x = (max_ids%tmp_width).view(-1,1).float()+1+outputs_shift_x_select.view(-1,1)
        tmp_y = (max_ids//tmp_width).view(-1,1).float()+1+outputs_shift_y_select.view(-1,1)
        tmp_x /= 1.0 * input_size[1] / net_stride
        tmp_y /= 1.0 * input_size[0] / net_stride

    return tmp_x, tmp_y, outputs, max_cls

def forward_tf(net, inputs):
    net.eval()
    with torch.no_grad():
        outputs, outputs2, atten_weights_list, self_atten_weights_list = net(inputs)
    return outputs, outputs2, atten_weights_list, self_atten_weights_list

def compute_nme(lms_pred, lms_gt, norm_indices):
    if norm_indices is not None:
        norm = torch.norm(lms_gt[:,norm_indices[0],:] - lms_gt[:,norm_indices[1],:], dim=1)
    else:
        norm = 1
    nmes = (torch.mean(torch.norm(lms_pred - lms_gt, dim=2), dim=1) / norm).cpu().numpy().tolist()
    return nmes 

def compute_fr_and_auc(nmes, thres=0.1, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc

def gen_pseudo(cfg, net, loader_train_u, data_type, cur_iter, device):
    img_names_list = []
    outputs_list = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_train_u):
            if cfg.det_head == 'map':
                inputs, img_names = data
                inputs = inputs.to(device)

                tmp_x, tmp_y, outputs, max_cls = forward_map(net, inputs, cfg.input_size, cfg.net_stride)
                tmp_batch, tmp_channel, tmp_height, tmp_width = outputs.size()
                tmp_x = tmp_x.view(tmp_batch, tmp_channel, 1)
                tmp_y = tmp_y.view(tmp_batch, tmp_channel, 1)
                outputs = torch.cat((tmp_x, tmp_y), 2)
                img_names_list += img_names
                outputs_list.append(outputs)
            elif cfg.det_head == 'tf':
                inputs, img_names = data
                inputs = inputs.to(device)
                outputs, _, _, _ = forward_tf(net, inputs)
                img_names_list += img_names
                outputs_list.append(outputs)
            elif cfg.det_head == 'pip':
                inputs, img_names = data
                inputs = inputs.to(device)

                tmp_x, tmp_y, outputs_cls, max_cls = forward_pip(net, inputs, cfg.input_size, cfg.net_stride)
                tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
                tmp_x = tmp_x.view(tmp_batch, tmp_channel, 1)
                tmp_y = tmp_y.view(tmp_batch, tmp_channel, 1)
                outputs = torch.cat((tmp_x, tmp_y), 2)
                img_names_list += img_names
                outputs_list.append(outputs)
            else:
                print('No such head:', cfg.det_head)
                exit(0)
    outputs_list = torch.cat(outputs_list, dim=0)
    pseudo_labels = [[img_names_list[i], data_type, cur_iter, outputs_list[i].flatten().cpu().numpy()] for i in range(len(img_names_list))]
    return pseudo_labels


