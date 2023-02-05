import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Data, Config, DataImage
from PGNet import PGNet


class Test(object):
    def __init__(self, network, path, model):
        ## dataset
        self.model  = model
        self.cfg    = Config(datapath=path, snapshot=model, mode='test')
        self.data   = DataImage(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=2)
        ## network
        self.net    = network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self, path, forwards=1):
        print(f'Saving results in {path}')
        os.makedirs(path, exist_ok=True)
        
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                mask  = mask.cuda().float()

                # Successive iteration of forwards on previous results
                for i in range(forwards):
                    p = self.net(image, shape=None)
                    # Replicate 1 channel mask into 3 channels
                    image = image.expand(-1, 3,-1,-1)
                
                # Resize and save
                out_resize   = F.interpolate(p[0],size=shape, mode='bilinear')
                pred   = torch.sigmoid(out_resize[0,0])
                pred  = (pred*255).cpu().numpy()
                name = os.path.basename(name[0])
                out = os.path.join(path, name)
                cv2.imwrite(out, np.round(pred))


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'Saliency detection on cropped BigBird images')
    parser.add_argument('--model', help='path to model')
    parser.add_argument('--root', help='root folder containing cropped bigbird object instances', required=True)
    parser.add_argument('--objects', help='only crop specific objects', nargs='*', default=None)
    parser.add_argument('--in-folder', help='name of folder where cropped images are stored', required=True)
    parser.add_argument('--out', help='output root path (default=root)', default=None)
    parser.add_argument('--out-folder', help='name of folder where output saliency maps are to be stored', required=True)
    parser.add_argument('--forwards', help='number of forward passes (default=1)', type=int, default=1)
    args = parser.parse_args()
    
    for k,v in args.__dict__.items():
        print(f'{k:->20} : {v}')
    
    # Load object names
    objects = args.objects
    if objects is None:
        objects = [i for i in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, i))]
        print(f'Inferencing on {len(objects)} objects')
    
    # Get output root
    out = args.out
    if out is None:
        out = args.root

    # Iterate over each object
    for obj_idx, obj in enumerate(objects, 1):
        print(f'\nInferencing on \t[{obj_idx:>3d}/{len(objects)}] : \t{obj}')
        obj_img_path = os.path.join(args.root, obj, args.in_folder)
        obj_out_path = os.path.join(out, obj, args.out_folder)
        t = Test(PGNet, obj_img_path, args.model)
        t.save(obj_out_path, args.forwards)
        print('-'*70)
