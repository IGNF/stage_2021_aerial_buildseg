#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import argparse
import torch 
import pandas as pd


from model.model import UNet
from inference import inference_roi,predictor_arg

# Root -> json file
#root = '/home/ign.fr/ttea/Code_IGN/AerialImageDataset'
#train_dir = os.path.join(root,'train/images')
#gt_dir = os.path.join(root,'train/gt')
#test_dir = os.path.join(root,'test/images')

var= pd.read_json('variables.json')

list_test_img = os.listdir(var['variables']['test_dir'])
path_image = os.path.join(var['variables']['test_dir'],list_test_img[2])
output_dir = r'/home/ign.fr/ttea/Code_IGN/Data/output'
model_dir = r'/home/ign.fr/ttea/Code_IGN/Data/model_save'





# Main 
def main(config):
        
    # Loading Model with Full Args 
    model = UNet(config.n_channel, config.conv_width, config.n_class, cuda=config.cuda)
    model.load_state_dict(torch.load(os.path.join(model_dir,"unet_crossentropy.pth")))
    
    # Inference : Prediction on test set 
    predictor = predictor_arg
    roi_size = (256,256)
    
    inference_roi(path_image,roi_size,predictor,output_dir,model)
    #for img in list_test_img:
    #    path_image = os.path.join(test_dir,img)
    #    inference_roi(path_image,roi_size,predictor,output_dir)
    

if __name__ == "__main__":
    
    print('Inference')
    parser = argparse.ArgumentParser()
    
    # Hyperparameter
    parser.add_argument('--n_epoch', default = 10)
    parser.add_argument('--n_epoch_test',type = int ,default = int(5)) #periodicity of evaluation on test set
    parser.add_argument('--batch_size',type = int, default = 16)
    parser.add_argument('--conv_width',default = [16,32,64,128,256,128,64,32,16])
    parser.add_argument('--cuda',default = 1)
    parser.add_argument('--lr', default = 0.001)
    
    parser.add_argument('--n_class',type = int, default = 2)
    parser.add_argument('--n_channel',type=int, default = 3)
    parser.add_argument('--class_names' , default= ['None','Batiment'])
    
    parser.add_argument('--save_model', default= True)
    parser.add_argument('--save_model_name ', default = "unet_bin_crossentropy.pth") 
    
    config = parser.parse_args()

    main(config)

# Modifier partie encoder -> EfficientNet puis comparer les 2 modèles / alternative : Resnet18
# Checker les paramètres 

# https://gitlab.inria.fr/naudeber/DeepHyperX/-/blob/master/main.py
# https://github.com/LeeJunHyun/Image_Segmentation/blob/master/dataset.py

# Next Step : https://pytorch-lightning.readthedocs.io/en/latest/
# Pytorch Lightning : https://github.com/PyTorchLightning/pytorch-lightning