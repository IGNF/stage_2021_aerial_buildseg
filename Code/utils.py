#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Utils 

import os 
import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.windows import Window

def read_image(root,filename):
    """
    read image with rasterio and return an array [C, W, H]
    
    no schema/georef returned yet.
    root : root directory 
    filename : image filename as string
    
    Returns: raster as an array 
    """
    img = rasterio.open(os.path.join(root,filename))
    img_array = img.read()
    img.close()
    return img_array

def get_tile(root,image_file,tile_size,idx):
    """
    image_file : image filename as string
    tile_size : tuple of the dimension for the tile (width, height)
    idx : index of the tile, int
    
    Returns: tile of the image file [channel,width,height] -> [nb tile , channel, width, height]
    """

    # Read Image 
    image =rasterio.open(os.path.join(root,image_file))
    
    image_shape = np.shape(image)
    width = image_shape[0] 
    #height= image_shape[1]
    
    tile_width = tile_size[0]
    tile_height = tile_size[1]
    
    # Number of tile 
    nb_tile_w = width // tile_width
    #nb_tile_h = height // tile_height
    
    row,col = divmod(idx,nb_tile_w)
    
    tile = image.read(window=Window(col*tile_height,row*tile_width,tile_size[0],tile_size[1]))
    return tile    


# Visualisation

def view(dataset, idx):
    """
    dataset: dataset contains tile & mask 
    idx : index 
    
    Returns : plot tile & mask  
    """
    
    item = dataset[idx]
    
    raster_tile = reshape_as_image(np.array(item[0]).astype(np.uint8))
    raster_gt = reshape_as_image(np.array(item[1][None,:,:]))
    
    figure, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,6))
    
    ax[0].imshow(raster_tile)
    ax[0].set_title('Raster Tile')
    ax[0].set_axis_off()
    
    ax[1].imshow(raster_gt)
    ax[1].set_title('Raster Gt')
    ax[1].set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
def view_batch(tiles, gt , pred = None, size = None, ncols = None):
    
    batch_size = tiles.shape[0]
    ncols = batch_size
    if size is not None :
        ncols = size
        
    if pred is None :
        figure, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(20, 8))    
    else : 
        figure, ax = plt.subplots(nrows=3, ncols=ncols, figsize=(20, 12))    
        
    for idx in range(ncols):
        
        item_tile = tiles[idx]
        item_gt = gt[idx]

        raster_tile = reshape_as_image(np.array(item_tile).astype(np.uint8))
        raster_gt = reshape_as_image(np.array(item_gt[None,:,:]))

        ax[0][idx].imshow(raster_tile)
        ax[0][idx].set_axis_off()

        ax[1][idx].imshow(raster_gt)
        ax[1][idx].set_axis_off()
        
        if pred is not None :
            item_pred = pred[idx]
            raster_pred = reshape_as_image(np.array(item_pred[None,:,:]))
            ax[2][idx].imshow(raster_pred)
            ax[2][idx].set_axis_off()

    plt.tight_layout()
    plt.show()    


