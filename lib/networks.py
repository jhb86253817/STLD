import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import time
from transformer import Transformer, PositionEmbeddingSine, MLP_SL

cudnn.deterministic = True
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

class Map_resnet(nn.Module):
    def __init__(self, resnet, cfg):
        super(Map_resnet, self).__init__()
        self.num_lms = cfg.num_lms
        self.input_size = cfg.input_size
        self.net_stride = cfg.net_stride
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.feat_dim = resnet.fc.weight.size()[1]

        self.deconv1 = nn.ConvTranspose2d(self.feat_dim, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_deconv1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_deconv2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_deconv3 = nn.BatchNorm2d(256)

        self.final_layer = nn.Conv2d(256, self.num_lms, kernel_size=1, stride=1, padding=0)
        self.final_layer2 = nn.Conv2d(256, self.num_lms, kernel_size=1, stride=1, padding=0)

        nn.init.normal_(self.deconv1.weight, std=0.001)
        if self.deconv1.bias is not None:
            nn.init.constant_(self.deconv1.bias, 0)
        nn.init.constant_(self.bn_deconv1.weight, 1)
        nn.init.constant_(self.bn_deconv1.bias, 0)

        nn.init.normal_(self.deconv2.weight, std=0.001)
        if self.deconv2.bias is not None:
            nn.init.constant_(self.deconv2.bias, 0)
        nn.init.constant_(self.bn_deconv2.weight, 1)
        nn.init.constant_(self.bn_deconv2.bias, 0)

        nn.init.normal_(self.deconv3.weight, std=0.001)
        if self.deconv3.bias is not None:
            nn.init.constant_(self.deconv3.bias, 0)
        nn.init.constant_(self.bn_deconv3.weight, 1)
        nn.init.constant_(self.bn_deconv3.bias, 0)

        nn.init.normal_(self.final_layer.weight, std=0.001)
        if self.final_layer.bias is not None:
            nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv1(x)
        x = self.bn_deconv1(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.bn_deconv2(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = self.bn_deconv3(x)
        x = F.relu(x)
        x1 = self.final_layer(x)
        x2 = self.final_layer2(x)
        return x1, x2

class TF_resnet(nn.Module):
    def __init__(self, resnet, cfg):
        super(TF_resnet, self).__init__()
        self.dynamic_query = cfg.dynamic_query
        self.dq_pos = cfg.dq_pos
        self.qa_attn = cfg.qa_attn
        self.qa_group = cfg.qa_group
        self.sigmoid = cfg.sigmoid
        self.tf_dim = cfg.tf_dim
        self.tf_en_num = cfg.tf_en_num 
        self.tf_de_num = cfg.tf_de_num 
        self.num_lms = cfg.num_lms
        self.input_size = cfg.input_size
        self.net_stride = cfg.net_stride
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.feat_dim = resnet.fc.weight.size()[1]

        ###########################################
        if self.dynamic_query:
            self.fmap_size = (int(self.input_size[0]/self.net_stride), int(self.input_size[1]/self.net_stride))
            self.fc1 = nn.Linear(self.tf_dim, self.num_lms*self.tf_dim)

            nn.init.normal_(self.fc1.weight, std=0.001)
            if self.fc1.bias is not None:
                nn.init.constant_(self.fc1.bias, 0)

        self.pos_layer = PositionEmbeddingSine(self.tf_dim//2)
        self.transformer = Transformer(d_model=self.tf_dim,
                                       num_encoder_layers=self.tf_en_num,
                                       num_decoder_layers=self.tf_de_num,
                                       num_queries=self.num_lms,
                                       qa_attn=self.qa_attn,
                                       qa_group=self.qa_group)
        self.query_embed = nn.Embedding(self.num_lms, self.tf_dim)
        self.bbox_embed = MLP_SL(self.tf_dim, 512, 2, 3)
        self.input_proj = nn.Conv2d(self.feat_dim, self.tf_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.input_proj(x)

        pos_embed = self.pos_layer(x)
        if self.dynamic_query:
            if self.dq_pos:
                x_pool = F.avg_pool2d(x+pos_embed, self.fmap_size).squeeze(2).squeeze(2)
            else:
                x_pool = F.avg_pool2d(x, self.fmap_size).squeeze(2).squeeze(2)
            dq = self.fc1(x_pool)
            dq = dq.view(-1, self.num_lms, self.tf_dim).permute(1,0,2)
            hs, _, atten_weights_list, self_atten_weights_list = self.transformer(x, None, self.query_embed.weight, pos_embed, dq)
        else:
            hs, _, atten_weights_list, self_atten_weights_list = self.transformer(x, None, self.query_embed.weight, pos_embed)
        #########################################

        # bs x num_lms x 2
        if self.sigmoid:
            outputs_coord, outputs_coord2 = self.bbox_embed(hs.squeeze(0)).sigmoid()
        else:
            outputs_coord, outputs_coord2 = self.bbox_embed(hs.squeeze(0))
        return outputs_coord, outputs_coord2, atten_weights_list, self_atten_weights_list

class Map_hrnet(nn.Module):
    def __init__(self, hrnet, cfg):
        super(Map_hrnet, self).__init__()
        self.num_lms = cfg.num_lms
        self.input_size = cfg.input_size
        self.net_stride = cfg.net_stride
        self.hrnet = hrnet

    def forward(self, x):
        x1, x2 = self.hrnet(x)
        return x1, x2

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
                            
    def forward(self, x):
        return x

class TF_hrnet(nn.Module):
    def __init__(self, hrnet, cfg):
        super(TF_hrnet, self).__init__()
        self.dynamic_query = cfg.dynamic_query
        self.dq_pos = cfg.dq_pos
        self.qa_attn = cfg.qa_attn
        self.qa_group = cfg.qa_group
        self.sigmoid = cfg.sigmoid
        self.tf_dim = cfg.tf_dim
        self.tf_en_num = cfg.tf_en_num 
        self.tf_de_num = cfg.tf_de_num 
        self.num_lms = cfg.num_lms
        self.input_size = cfg.input_size
        self.net_stride = cfg.net_stride
        self.hrnet = hrnet
        self.hrnet.head = Identity()

        ###########################################
        if self.dynamic_query:
            self.fmap_size = (int(self.input_size[0]/self.net_stride), int(self.input_size[1]/self.net_stride))
            self.fc1 = nn.Linear(self.tf_dim, self.num_lms*self.tf_dim)

            nn.init.normal_(self.fc1.weight, std=0.001)
            if self.fc1.bias is not None:
                nn.init.constant_(self.fc1.bias, 0)
        ###########################################

        self.pos_layer = PositionEmbeddingSine(self.tf_dim//2)
        self.transformer = Transformer(d_model=self.tf_dim,
                                       num_encoder_layers=self.tf_en_num,
                                       num_decoder_layers=self.tf_de_num,
                                       num_queries=self.num_lms,
                                       qa_attn=self.qa_attn,
                                       qa_group=self.qa_group)
        self.query_embed = nn.Embedding(self.num_lms, self.tf_dim)
        self.bbox_embed = MLP_SL(self.tf_dim, 512, 2, 3)
        self.input_proj = nn.Conv2d(270, self.tf_dim, kernel_size=1)

    def forward(self, x):
        x, _ = self.hrnet(x)
        x = self.input_proj(x)

        pos_embed = self.pos_layer(x)
        #########################################
        if self.dynamic_query:
            if self.dq_pos:
                x_pool = F.avg_pool2d(x+pos_embed, self.fmap_size).squeeze(2).squeeze(2)
            else:
                x_pool = F.avg_pool2d(x, self.fmap_size).squeeze(2).squeeze(2)
            dq = self.fc1(x_pool)
            dq = dq.view(-1, self.num_lms, self.tf_dim).permute(1,0,2)
            hs, _, atten_weights_list, self_atten_weights_list = self.transformer(x, None, self.query_embed.weight, pos_embed, dq)
        else:
            hs, _, atten_weights_list, self_atten_weights_list = self.transformer(x, None, self.query_embed.weight, pos_embed)
        #########################################

        # bs x num_lms x 2
        if self.sigmoid:
            outputs_coord, outputs_coord2 = self.bbox_embed(hs.squeeze(0)).sigmoid()
        else:
            outputs_coord, outputs_coord2 = self.bbox_embed(hs.squeeze(0))
        return outputs_coord, outputs_coord2, atten_weights_list, self_atten_weights_list

