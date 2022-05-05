#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import argparse

from train import train_full
from model.model import UNet

from dataloader.dataloader import InriaDataset
import pandas as pd 

var= pd.read_json('variables.json')

# Root 
#root = '/home/ign.fr/ttea/Code_IGN/AerialImageDataset'
#train_dir = os.path.join(root,'train/images')
#gt_dir = os.path.join(root,'train/gt')
#test_dir = os.path.join(root,'test/images')


# Main 
def main(args):
  
    # Training Model with Full Args 
    model = UNet(args.n_channel, args.conv_width, args.n_class, cuda=args.cuda)
    trained_model = train_full(args, model)
     
if __name__ == "__main__":
    
    print('Training')
    parser = argparse.ArgumentParser()
    
    # Hyperparameter
    parser.add_argument('--n_epoch', default = 40)
    parser.add_argument('--n_epoch_test',type = int ,default = int(5)) #periodicity of evaluation on test set
    parser.add_argument('--batch_size',type = int, default = 16)
    parser.add_argument('--conv_width',default = [16,32,64,128,256,128,64,32,16])
    parser.add_argument('--cuda',default = 1)
    parser.add_argument('--lr', default = 0.0001)
    
    parser.add_argument('--n_class',type = int, default = 2)
    parser.add_argument('--n_channel',type=int, default = 3)
    parser.add_argument('--class_names' , default= ['None','Batiment'])
    
    parser.add_argument('--save_model', default= True)
    parser.add_argument('--save_model_name ', default = "unet_test.pth") 
    
    
    tile_size = (512,512)
    parser.add_argument('--train_dataset', default = InriaDataset(var['variables']['root'], tile_size, 'train', None, False))
    parser.add_argument('--val_dataset', default = InriaDataset(var['variables']['root'], tile_size, 'validation', None, False))
    
    args = parser.parse_args()

    main(args)