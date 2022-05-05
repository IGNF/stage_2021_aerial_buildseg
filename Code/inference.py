#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os 
import numpy as np

import torch

import rasterio
import rasterio.mask
from rasterio.windows import Window
from rasterio.plot import show

# Inference 
def predictor_arg(tensor,model):
    """
    args: 
        - tensor : tensor of size N,C,W,H
        - model : Model 
    
    Return : argmax of the model with the tensor as input 
    """
    return model(tensor).argmax(1) 

def inference_roi(path_image,roi_size,predictor,output_dir,model):
    """
    path_image : path for the image 
    roi_size : inference size 
    predictor : model predictor
    output_dir : output directory for the prediction 
    model : Model 
    
    Show the prediction 
    """
    # open image with rasterio 
    img =  rasterio.open(os.path.join(path_image))
    height = img.height
    width = img.width
    
    nb_col = width // roi_size[0]
    nb_row = height // roi_size[1]
    
    base=os.path.basename(path_image)
    base_without_ex = os.path.splitext(base)[0]

    profile = img.profile.copy()

    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        driver = "GTiff",
        height = height,
        width = width,
        compress='lzw')

    img_transform = img.transform
    
    # Initialisation 
    mask =  np.zeros((1,width, height))
    #print('mask shape',np.shape(mask))
    
    shp_width = np.shape(mask)[1]
    shp_height = np.shape(mask)[2]
    
    with torch.no_grad(): 
        for col in range(0,nb_col):
            for row in range(0,nb_row):
                
                tile = img.read(window=Window(col*roi_size[0],row*roi_size[1],roi_size[0],roi_size[1]))
                tile_tensor = torch.from_numpy(tile).float()
                pred = predictor(tile_tensor.unsqueeze(dim=0),model)
                pred_cm = pred.cpu().detach().numpy()
                
                # Affiche Rvb & Mask 
                show(tile)
                show(pred_cm)
                
                mask[:,row*roi_size[1]:(row+1)*roi_size[1],col*roi_size[0]:(col+1)*roi_size[0]] = pred_cm.astype(np.uint8)
                
                # Cas unique dernière tile en diagonale
                if (col == nb_col -1) and (row == nb_row -1):
                    tile = img.read(window=Window( shp_width - roi_size[0], shp_height - roi_size[1],roi_size[0],roi_size[1]))
                    tile_tensor = torch.from_numpy(tile).float()
                    pred = predictor(tile_tensor.unsqueeze(dim=0),model)
                    pred_cm = pred.cpu().detach().numpy()
                    mask[:,shp_height - roi_size[0] :,shp_width - roi_size[1]:] = pred_cm.astype(np.uint8)
                    
                
                # Dernière Row -> Recouvrement
                if row == nb_row -1:
                    # window argument : taille height, width image
                    tile = img.read(window=Window(col*roi_size[0],shp_height - roi_size[0],roi_size[0],roi_size[1]))
                    tile_tensor = torch.from_numpy(tile).float()
                    pred = predictor(tile_tensor.unsqueeze(dim=0),model)
                    pred_cm = pred.cpu().detach().numpy()
                    mask[:,shp_height - roi_size[1]:,col*roi_size[0]:(col+1)*roi_size[0]] = pred_cm.astype(np.uint8)
                
                # Dernière Col -> Recouvrement 
                if col == nb_col -1:
                    # window argument : taille height, width image
                    tile = img.read(window=Window( shp_width - roi_size[0], row*roi_size[1] ,roi_size[0],roi_size[1]))
                    tile_tensor = torch.from_numpy(tile).float()
                    pred = predictor(tile_tensor.unsqueeze(dim=0),model)
                    pred_cm = pred.cpu().detach().numpy()
                    mask[:,row*roi_size[1]:(row+1)*roi_size[1],shp_height - roi_size[0]:] = pred_cm.astype(np.uint8)
                    
        # Profile update (transformation)
        x,y = rasterio.transform.xy(img_transform, nb_col*roi_size[0],nb_row*roi_size[1])
        out_transform = rasterio.transform.from_origin(x,y,nb_col*roi_size[0],nb_row *roi_size[1]) 
        out_tile_name = os.path.join(output_dir,f'{base_without_ex}_{nb_col:02}_{nb_row:02}_predfinal.tif')
        profile.update(transform = out_transform)
        
        # Plot mask 
        mask = mask.astype(np.uint8)
        show(mask)
        
        with rasterio.open(out_tile_name,"w",**profile) as dst : 
            dst.write(mask)