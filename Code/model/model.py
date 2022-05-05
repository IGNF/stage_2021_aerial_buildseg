#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import argparse
#from dataloader.dataloader import InriaDataset
from dataloader.dataloader import InriaDataset


# Model 

#Double Conv2D
def conv_block(in_channel, out_channel):
    """
    in_channel : number of input channel, int 
    out_channel : number of output channel, int
    
    Returns : Conv Block of 2x Conv2D with ReLU 
    """
    
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3,padding=1),
        nn.ReLU(inplace= True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3,padding=1),
        nn.ReLU(inplace= True),
    )
    return conv


# crop the image(tensor) to equal size, half left side image concatenate with right side image
def crop(target_tensor, tensor): # x,c
    """
    target_tensor : target the tensor to crop  
    tensor: tensor 
    
    Returns : tensor cropped by half left side image concatenate with right side image
    
    """
    
    target_size = target_tensor.size()[2] 
    tensor_size = tensor.size()[2]        
    delta = tensor_size - target_size     
    delta = delta // 2                    

    if (tensor_size - 2*delta)%2 == 0:
      tens = tensor[:, :, delta:tensor_size- delta , delta:tensor_size-delta]

    elif (tensor_size -2*delta)%2 ==1:
      tens = tensor[:, :, delta:tensor_size- delta -1  , delta:tensor_size-delta -1]
    return tens

class UNet(nn.Module):
  """
  UNet network for semantic segmentation
  """
  
  def __init__(self, n_channels, conv_width,  n_class, cuda = 1):
    """
    initialization function
    n_channels, int, number of input channel
    conv_width, int list, depth of the convs
    n_class = int,  the number of classes
    """
    super(UNet, self).__init__() #necessary for all classes extending the module class
    self.is_cuda = cuda
    self.n_class = n_class
    
    ## Encoder 
    
    # Conv2D (input channel, outputchannel, kernel size)
    self.c1 = conv_block(3,16)
    self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.c2 = conv_block(16,32)
    self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.c3 = conv_block(32,64)
    self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)  
    
    self.c4 = conv_block(64,128)
    self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)      
    
    self.c5 = conv_block(128,256)

    ## Decoder 
    
    # Transpose & UpSampling Convblock   
    self.t6 = nn.ConvTranspose2d(256,128, kernel_size= 2, stride=2)
    self.c6 = conv_block(256,128)

    self.t7 = nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
    self.c7 = conv_block(128,64)

    self.t8 = nn.ConvTranspose2d(64,32, kernel_size=2, stride=2)
    self.c8 = conv_block(64,32)

    self.t9 = nn.ConvTranspose2d(32,16, kernel_size=2, stride=2)
    self.c9 = conv_block(32,16)
    
    # Final Classifyer layer 
    self.outputs = nn.Conv2d(16, n_class, kernel_size= 1)
    
    #weight initialization
    self.c1[0].apply(self.init_weights)
    self.c2[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.c4[0].apply(self.init_weights)
    self.c5[0].apply(self.init_weights)
    self.c6[0].apply(self.init_weights)
    self.c7[0].apply(self.init_weights)
    self.c8[0].apply(self.init_weights)
    self.c9[0].apply(self.init_weights)
    
    if cuda: #put the model on the GPU memory
      self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
  def forward(self, input):
    """
    the function called to run inference
    """  
    if self.is_cuda: #put data on GPU
        input = input.cuda()

    # Encoder (Left Side)
    c1=self.c1(input)
    #print('input size', input.size())
    p1=self.p1(c1)
    c2=self.c2(p1)
    p2=self.p2(c2)
    c3=self.c3(p2)
    p3=self.p3(c3)
    c4=self.c4(p3)
    p4=self.p4(c4)
    c5=self.c5(p4)

    # Decoder (Right Side)
    u6=self.t6(c5)
    y4 = crop(u6,c4)
    concat4 = torch.cat([u6,y4],1)
    x6=self.c6(concat4)
    
    u7=self.t7(x6)
    y3 = crop(u7,c3)
    x7=self.c7(torch.cat([u7,y3],1))
    
    u8=self.t8(x7)
    y2 = crop(u8,c2)
    x8=self.c8(torch.cat([u8,y2],1))
    
    u9=self.t9(x8)
    y1=crop(u9,c1)
    
    x9=self.c9(torch.cat([u9,y1],1))
    
    # Final Output Layer
    out = self.outputs(x9)
    return out


class UNetDropout(nn.Module):
  """
  UNet network for semantic segmentation
  """
  
  def __init__(self, n_channels, conv_width,  n_class, cuda = 1):
    """
    initialization function
    n_channels, int, number of input channel
    conv_width, int list, depth of the convs
    n_class = int,  the number of classes
    """
    super(UNetDropout, self).__init__() #necessary for all classes extending the module class
    self.is_cuda = cuda
    self.n_class = n_class
    
    ## Encoder 
    
    # Conv2D (input channel, outputchannel, kernel size)
    self.c1 = conv_block(3,16)
    self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
    

    self.c2 = conv_block(16,32)
    self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.c3 = conv_block(32,64)
    self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)  
    
    self.c4 = conv_block(64,128)
    self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)      
    
    self.c5 = conv_block(128,256)
    
    self.d1 = nn.Dropout2d(0.1)
    self.d2 = nn.Dropout2d(0.1)
    self.d3 = nn.Dropout2d(0.1)
    self.d4 = nn.Dropout2d(0.1)

    ## Decoder 
    
    self.d5 = nn.Dropout2d(0.1)
    self.d6 = nn.Dropout2d(0.1)
    self.d7 = nn.Dropout2d(0.1)
    self.d8 = nn.Dropout2d(0.1)
    
    # Transpose & UpSampling Convblock   
    self.t6 = nn.ConvTranspose2d(256,128, kernel_size= 2, stride=2)
    self.c6 = conv_block(256,128)

    self.t7 = nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
    self.c7 = conv_block(128,64)

    self.t8 = nn.ConvTranspose2d(64,32, kernel_size=2, stride=2)
    self.c8 = conv_block(64,32)

    self.t9 = nn.ConvTranspose2d(32,16, kernel_size=2, stride=2)
    self.c9 = conv_block(32,16)
    
    # Final Classifyer layer 
    self.outputs = nn.Conv2d(16, n_class, kernel_size= 1)
    
    #weight initialization
    self.c1[0].apply(self.init_weights)
    self.c2[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.c4[0].apply(self.init_weights)
    self.c5[0].apply(self.init_weights)
    self.c6[0].apply(self.init_weights)
    self.c7[0].apply(self.init_weights)
    self.c8[0].apply(self.init_weights)
    self.c9[0].apply(self.init_weights)
    
    if cuda: #put the model on the GPU memory
      self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
  def forward(self, input):
    """
    the function called to run inference
    """  
    if self.is_cuda: #put data on GPU
        input = input.cuda()

    # Encoder (Left Side)
    c1=self.c1(input)
    #print('input size', input.size())
    p1 = self.p1(c1)
    p1 = self.d1(p1)
    c2 = self.c2(p1)
    p2 = self.p2(c2)
    p2 = self.d2(p2)
    c3 = self.c3(p2)
    p3 = self.p3(c3)
    p3 = self.d3(p3)
    c4 = self.c4(p3)
    p4 = self.p4(c4)
    p4 = self.d4(p4)
    c5 = self.c5(p4)


    # Decoder (Right Side)
    u6=self.t6(c5)
    y4 = crop(u6,c4)
    concat4 = torch.cat([u6,y4],1)
    x6=self.c6(concat4)
    x6= self.d5(x6)
    
    u7=self.t7(x6)
    y3 = crop(u7,c3)
    x7=self.c7(torch.cat([u7,y3],1))
    x7= self.d6(x7)
    
    u8=self.t8(x7)
    y2 = crop(u8,c2)
    x8=self.c8(torch.cat([u8,y2],1))
    x8= self.d7(x8)
    
    u9=self.t9(x8)
    y1=crop(u9,c1)
    
    x9=self.c9(torch.cat([u9,y1],1))
    x9= self.d8(x9)
    
    # Final Output Layer
    out = self.outputs(x9)
    return out


