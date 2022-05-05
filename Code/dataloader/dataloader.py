# -*- coding: utf-8 -*-

import os 
import pandas as pd 

var= pd.read_json('variables.json')

# Import Libraries 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import rasterio
import rasterio.mask
from rasterio.windows import Window


# Dataset
class InriaDataset(Dataset):

    def __init__(self, root, tile_size, mode, transform, filtered,part):
        
        self.root = root
        # self.get_tile = tile_loader  # not external
        self.tile_size= tile_size
        self.mode = mode
        self.transform = transform
        self.filtered = filtered   
        self.part = part
        
        # Build image list
        self.train_dir = os.path.join(root,'train/images')
        self.gt_dir = os.path.join(root,'train/gt')
        self.test_dir = os.path.join(root,'test/images')

        self.train_images = os.listdir(self.train_dir)
        self.test_images = os.listdir(self.test_dir)
        self.gt_images = os.listdir(self.gt_dir)
        
        # Datalist -> all tiles, only_bat -> only batiment tiles 
        self.datalist = []
        self.only_bat = []
        self.used_tiles = []
        self.tiles = []
        
        score = 0
        label= 'None'
        
        # Nb tuiles par images 
        # all images on INRIA Dataset have the same shape/size so we use the first image shape
        with rasterio.open(os.path.join(self.train_dir, self.train_images[0])) as first_img :
            # shape dimension is [C, W, H ]
            images_width = first_img.width
            images_heigth = first_img.height
        
        tile_width = self.tile_size[0]
        tile_heigth = self.tile_size[1]
        # round to int number we coud lost some data on image border
        nb_tile_col = images_width // tile_width
        nb_tile_row = images_heigth // tile_heigth
        self.nb_tile_by_image = nb_tile_col*nb_tile_row

        if filtered == True :
            for id_image in range(0, len(self.train_images)):
                train_image = self.train_images[id_image]
                gt_image = self.gt_images[id_image]
                for id_tile in range(0, self.nb_tile_by_image):

                    mask = self._get_tile(self.gt_dir, train_image, id_tile)
                    mask[np.where(mask==255)] = 1

                    # Determine the score & label of the mask 
                    score = (mask==1).sum() / np.shape(mask)[1]

                    if score >0:
                        label = 'Batiment'
                    else:
                        label = 'None'
                    self.datalist.append((id_tile, train_image, gt_image, score, label))
        else: 
            for id_image in range(0, len(self.train_images)):
                train_image = self.train_images[id_image]
                gt_image = self.gt_images[id_image]
                for id_tile in range(0, self.nb_tile_by_image):
                    self.datalist.append((id_tile, train_image, gt_image, None, None))
                    
        self.used_tiles = self.datalist
        if filtered == True:
            for data_tuple in self.datalist:
                if data_tuple[4] == 'Batiment':
                    self.only_bat.append(data_tuple)
            self.used_tiles = self.only_bat

        
        # Split Train/Dev 80/20%
        slice_20= int(len(self.used_tiles)*(0.2))
        
        # 1/5 -> validation is 1/5
        if self.part == 1 :
            if self.mode == 'train':
                del self.used_tiles[:slice_20]
                self.tiles = self.used_tiles
            elif self.mode == 'validation':
                self.tiles = self.used_tiles[:slice_20]
       
        # 2/5
        if self.part == 2:
            if self.mode == 'train':
                del self.used_tiles[slice_20:2*slice_20]
                self.tiles = self.used_tiles
            elif self.mode == 'validation':
                self.tiles = self.used_tiles[slice_20:2*slice_20]
        
        # 3/5
        if self.part == 3:
            if self.mode == 'train':
                del self.used_tiles[2*slice_20:3*slice_20]
                self.tiles = self.used_tiles
            elif self.mode == 'validation':
                self.tiles = self.used_tiles[2*slice_20:3*slice_20]
        # 4/5
        if self.part == 4:
            if self.mode == 'train':
                del self.used_tiles[3*slice_20:4*slice_20]
                self.tiles = self.used_tiles
            elif self.mode == 'validation':
                self.tiles = self.used_tiles[3*slice_20:4*slice_20]
        # 5/5
        if self.part == 5:
            if self.mode == 'train':
                del self.used_tiles[4*slice_20:]
                self.tiles = self.used_tiles
            elif self.mode == 'validation':
                self.tiles = self.used_tiles[4*slice_20:]
            
        # test mode 
        else: 
            self.images = self.test_images
            
    def __len__(self):
        
        return len(self.tiles)

    def __getitem__(self, idx):
        
        if self.mode =='test':
            root = self.test_dir
        else:
            root = self.train_dir
        
        tile_data = self.tiles[idx]
        id_image, id_tile = tile_data[1], tile_data[0]
                    
        image = self._get_tile(self.train_dir, id_image, id_tile)
        mask = self._get_tile(self.gt_dir, id_image, id_tile)   
        mask[np.where(mask==255)] = 1
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask)
        
        return image_tensor, mask_tensor[0,:,:]
    
    
    def _get_tile(self, root, image_file, idx):
        # Read Image 
        with rasterio.open(os.path.join(root, image_file)) as dataset :

            width = dataset.width
            height= dataset.height

            tile_width = self.tile_size[0]
            tile_height = self.tile_size[1]

            # Number of tile 
            nb_tile_w = width // tile_width
            nb_tile_h = height // tile_height

            row, col = divmod(idx, nb_tile_w)

            tile = dataset.read(window=Window(col*tile_height,row*tile_width,self.tile_size[0],self.tile_size[1]))
            return tile  