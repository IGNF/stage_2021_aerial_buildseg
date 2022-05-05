#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import argparse
from dataloader.dataloader import InriaDataset


# UNet Model 

# UNet Fonctions ----------------------------------------------------------------------------------

# Double Conv2D
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

class EncoderBlock(nn.Module):
    def __init__(self,input_channel, output_channel,depth,n_block):
        super(EncoderBlock,self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel 
        self.depth = depth 
        self.n_block = n_block 
        
        self.conv  = conv_block(self.input_channel, self.output_channel)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # weight initialization 
        self.conv[0].apply(self.init_weights)
            
    def init_weights(self,layer): #gaussian init for the conv layers
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self,x):
        
        c = self.conv(x)
        if self.depth != self.n_block : 
            y = self.pool(c)
        else : 
            y = self.conv(x)
        
        return y,c 

    
class DecoderBlock(nn.Module):
    def __init__(self,input_channel, output_channel):
        super(DecoderBlock,self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel 
        
        self.conv_t  = nn.ConvTranspose2d(self.input_channel,self.output_channel, kernel_size= 2, stride=2)
        self.conv = conv_block(self.input_channel,self.output_channel)
        
        self.conv[0].apply(self.init_weights)
            
    def init_weights(self,layer): #gaussian init for the conv layers
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self,x,skip):
        u = self.conv_t(x)
        concat =torch.cat([u,skip],1)
        x = self.conv(concat)
        
        return x

# UNet Fonctions END ----------------------------------------------------------------------------------    
    
# Original UNet ---------------------------------------------------------------------------------------
class OriginalUNet(nn.Module):
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
    super(OriginalUNet, self).__init__() #necessary for all classes extending the module class
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

#--------------------------------------------------------------------------------------------------------

# Generic UNet : 
#- Choix possible Block 
#- Nombres d'étapes 
#- Utilisation Batchnormes & Dropout 

class GenericUNet(nn.Module):
  """
  UNet network for semantic segmentation
  """
  
  def __init__(self, n_channels, conv_width,  n_class, n_block, cuda = 1):
    """
    initialization function
    n_channels, int, number of input channel
    conv_width, int list, depth of the convs
    n_class = int,  the number of classes
    n_block = int, the number of blocks 
    """
    super(GenericUNet, self).__init__() #necessary for all classes extending the module class
    self.is_cuda = cuda
    
    self.n_class = n_class
    self.n_block = n_block 
    self.conv_width = conv_width 
    
    self.enc = []
    self.dec = []
    
    #-------------------------------------------------------------
    
    ## Encoder 
    
    # Conv2D (input channel, outputchannel, kernel size)
    
    for i in range(self.n_block):
        self.enc.append(EncoderBlock(self.conv_width[i],self.conv_width[i+1],i+1,self.n_block))
    
    #--------------------------------------------------------------

    self.enc = nn.ModuleList(self.enc)
    
    ## Decoder     
    
    # Transpose & UpSampling Convblock   
    
    for i in range(self.n_block-1):
        self.dec.append(DecoderBlock(self.conv_width[self.n_block+i],self.conv_width[self.n_block+i+1]))
    
    self.dec = nn.ModuleList(self.dec)
    
    # Final Classifyer layer 
    
    self.outputs = nn.Conv2d(self.conv_width[-1], self.n_class, kernel_size= 1)
    
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

    #-------------------------------------------------
    # Encoder (Left Side)
    
    enc = []
    skip = [] 
    
    for i in range(self.n_block):
        
        if i == 0:
            enc.append(self.enc[i](input)[0])
            skip.append(self.enc[i](input)[1])
            
        else : 
            enc.append(self.enc[i](enc[i-1])[0])
            skip.append(self.enc[i](enc[i-1])[1])
        
    #--------------------------------------------------
    # Decoder (Right Side)
    
    dec = []
    
    for i in range(self.n_block-1):
        if i==0:
            dec.append(self.dec[i](skip[self.n_block -1 -i],skip[self.n_block -2 -i]))
            
        else :
            dec.append(self.dec[i](dec[i-1],skip[self.n_block -2 -i]))
            
    # Final Output Layer 
    out = self.outputs(dec[-1])
    
    return out

# Generic UNet with encodeur decoder class -------------------------------------------------------------------------

#-------------------------------------------------------------
# Encodeur
class GenericUNetEncoder(nn.Module):
  """
  UNet network for semantic segmentation
  """
  
  def __init__(self, n_channels, conv_width,  n_class, n_block, cuda = 1):
    """
    initialization function
    n_channels, int, number of input channel
    conv_width, int list, depth of the convs
    n_class = int,  the number of classes
    n_block = int, the number of blocks 
    """
    super(GenericUNetEncoder, self).__init__() #necessary for all classes extending the module class
    self.is_cuda = cuda
    
    self.n_class = n_class
    self.n_block = n_block 
    self.conv_width = conv_width 
    
    self.enc = []
    
    # Conv2D (input channel, outputchannel, kernel size)
    
    for i in range(self.n_block):
        self.enc.append(EncoderBlock(self.conv_width[i],self.conv_width[i+1],i+1,self.n_block))
    
    self.enc = nn.ModuleList(self.enc)
    
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
    
    enc = []
    skip = [] 
    
    for i in range(self.n_block):
        
        if i == 0:
            enc.append(self.enc[i](input)[0])
            skip.append(self.enc[i](input)[1])
            
        else : 
            enc.append(self.enc[i](enc[i-1])[0])
            skip.append(self.enc[i](enc[i-1])[1])
    
    return enc, skip 

#-------------------------------------------------------------
# Decoder 
class GenericUNetDecoder(nn.Module):
  """
  UNet network for semantic segmentation
  """
  
  def __init__(self, n_channels, conv_width,  n_class, n_block,encoder, cuda = 1):
    """
    initialization function
    n_channels, int, number of input channel
    conv_width, int list, depth of the convs
    n_class = int,  the number of classes
    n_block = int, the number of blocks 
    """
    super(GenericUNetDecoder, self).__init__() #necessary for all classes extending the module class
    self.is_cuda = cuda
    
    self.n_class = n_class
    self.n_block = n_block 
    self.conv_width = conv_width 
    
    self.skip = encoder[1]
    self.dec= []
    
    ## Decoder     
    
    # Transpose & UpSampling Convblock   
    for i in range(self.n_block-1):
        self.dec.append(DecoderBlock(self.conv_width[self.n_block+i],self.conv_width[self.n_block+i+1]))
    
    self.dec = nn.ModuleList(self.dec)
    
    # Final Classifyer layer 
    
    self.outputs = nn.Conv2d(self.conv_width[-1], self.n_class, kernel_size= 1)
    
    if cuda: #put the model on the GPU memory
      self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
  def forward(self, input):
    """
    the function called to run inference
    """  
    
    dec = []
    
    for i in range(self.n_block-1):
        if i==0:
            dec.append(self.dec[i](self.skip[self.n_block -1 -i],self.skip[self.n_block -2 -i]))
        else :
            dec.append(self.dec[i](dec[i-1],self.skip[self.n_block -2 -i]))
            
    # Final Output Layer 
    out = self.outputs(dec[-1])
    
    return out

class GenericUNetClass(nn.Module):
    """
    UNet network for semantic segmentation
    """

    def __init__(self, n_channels, conv_width,  n_class, n_block,encoder, decoder,cuda = 1):
        """
        initialization function
        n_channels, int, number of input channel
        conv_width, int list, depth of the convs
        n_class = int,  the number of classes
        n_block = int, the number of blocks 
        """
        super(GenericUNetClass, self).__init__() #necessary for all classes extending the module class
        self.is_cuda = cuda

        self.n_class = n_class
        self.n_block = n_block 
        self.conv_width = conv_width 

        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self,layer): #gaussian init for the conv layers
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        """
        the function called to run inference
        """  

        pred_encoder = self.encoder(input)
        decoder = self.decoder 
        pred = decoder(pred_encoder)

        return pred
