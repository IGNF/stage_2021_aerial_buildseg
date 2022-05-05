# Ex 1 - Passer d'un couple d'image RVB + IRC à un tenseur R,V,B,IR de même taille & compatible avec pytorch 

#%%
rvb_path = './IMAGES/A1_RVB/zone_1.tif'
irc_path = './IMAGES/A2_IRC/zone_1.tif'

#%%
## Import librairies 

import rasterio
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import torch
import torchvision
import torchvision.transforms as transforms
from rasterio.windows import Window
from rasterio.plot import show

#%%

# Reading rvb & irc file 
src_rvb = rasterio.open(rvb_path)
src_irc = rasterio.open(irc_path) 

print("rvb profile :",src_rvb.profile)

# Geo-Trasnform 
print("rvb geometric transformation:",src_rvb.profile['transform'])

# Extract to numpy array 
# Extract a smaller window 256 row x 512 columns 
def extract(src,row,column):
    """
    Take as input the open raster file, the parameters rows & columns
    
    Output: 
        - src.read : extract to numpy array with a smaller window 
    """
    return src.read(1,window=Window(0,0,column,row))
    
array_rvb = extract(src_rvb,256,512)
array_irc = extract(src_irc,256,512)

print("rvb array :",array_rvb)
print("rvb shape :",array_rvb.shape)

#%%
# Concatenate RGB & IRC array 
def concatenate(array1,sub_array2,idx):
    """
    Take as input the first array & the sub array we want to concatenate, the indice of which sub array we want
    
    Output:
        - concatenate array 
    """
    return np.append(array1,np.array([sub_array2[idx]]),axis=0)

concat_array = concatenate(rvb_array,irc_array,0)

print("concatenate array shape :",concat_array.shape)

#%%
# Transform to tensor 

tensor_rvb_irc = torch.from_numpy(concat_array)
print("tensor rvb irc :",tensor_rvb_irc)
print("tensor rvb irc shape:",tensor_rvb_irc.shape)