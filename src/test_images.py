#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Data, Config, DataImage
from PGNet import PGNet

class Test(object):
    def __init__(self, Network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Config(datapath=path, snapshot=model, mode='test')
        self.data   = DataImage(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=2)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        head = os.path.join('../result', self.model[3:], self.cfg.datapath.split(os.sep)[-1])
        if not os.path.exists(head):
            os.makedirs(head)
        print(f'Saving results at {head}')

        with torch.no_grad():
            for image, mask, shape, name in self.loader:

                image = image.cuda().float()
                mask  = mask.cuda().float()
                p = self.net(image, shape=None)
                out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
                pred   = torch.sigmoid(out_resize[0,0])
                pred  = (pred*255).cpu().numpy()

                name = os.path.basename(name[0])
                out = os.path.join(head, name.split('.')[0]+'_mask.png')
                print(out)
                cv2.imwrite(out, np.round(pred))

if __name__=='__main__':
    img_root = sys.argv[1]
    for model in ['model-31']:
        t = Test(PGNet, img_root, os.path.join('../model', 'PGNet_DUT+HR', model))
        t.save()
