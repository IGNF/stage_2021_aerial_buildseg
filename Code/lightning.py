#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import argparse
import rasterio
import torch
import numpy as np 
from train import train_full
from model.model import UNet
from torch.utils.data import Dataset, DataLoader
import torchnet as tnt
from tqdm.notebook import tqdm as tqdm_nb
import torch.nn as nn
import torch.optim as optim

from dataloader.dataloader import InriaDataset
from utils import  view_batch
from loss.loss import ConfusionMatrixTorch

import rasterio.mask
from rasterio.windows import Window
import pytorch_lightning as pl 
from sklearn.metrics import confusion_matrix

from pytorch_lightning.callbacks import Callback

# Root 
root = '/home/ign.fr/ttea/Code_IGN/AerialImageDataset'
train_dir = os.path.join(root,'train/images')
gt_dir = os.path.join(root,'train/gt')
test_dir = os.path.join(root,'test/images')

list_test_img = os.listdir(test_dir)
path_image = os.path.join(test_dir,list_test_img[2])
output_dir = r'/home/ign.fr/ttea/stage_segmentation_2021/U-Net-Model/output'


# Dataset
class InriaDataset(Dataset):

    def __init__(self, root, tile_size, mode, transform, filtered):
        
        self.root = root
        # self.get_tile = tile_loader  # not external
        self.tile_size= tile_size
        self.mode = mode
        self.transform = transform
        self.filtered = filtered   
        
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
        
        # Shuffle Training Data 
        # self.train_images = np.random.RandomState(0).permutation(self.train_images)
        # self.gt_images = np.random.RandomState(0).permutation(self.gt_images)
        # ND : done by Pytorhc dataloader not necessary here
        
        # Nb tuiles par images 
        # all images on INRIA Dataset have the same shape/size so we use the first image shape
        with rasterio.open(os.path.join(self.train_dir, self.train_images[0])) as first_img :
            # shape dimension is [C, W, H ]
            images_width = first_img.width
            images_heigth = first_img.height
        
        tile_width = tile_size[0]
        tile_heigth = tile_size[1]
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
        slice_80 = int(len(self.used_tiles)*(0.8))
       
        if self.mode == 'train':
            self.tiles = self.used_tiles[:slice_80]
        elif self.mode == 'validation':
            self.tiles = self.used_tiles[slice_80:]
            
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

        #torchvision.transforms.RandomHorizontalFlip(p=0.5)
        mask[np.where(mask==255)] = 1
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask)
        
        return image_tensor, mask_tensor[0,:,:]
    
    
    def _get_tile(self, root, image_file, idx):
        # Read Image 
        with rasterio.open(os.path.join(root, image_file)) as dataset :

            width = dataset.width
            #height= dataset.height

            tile_width = self.tile_size[0]
            tile_height = self.tile_size[1]

            # Number of tile 
            nb_tile_w = width // tile_width
            #nb_tile_h = height // tile_height

            row, col = divmod(idx, nb_tile_w)

            tile = dataset.read(window=Window(col*tile_height,row*tile_width,self.tile_size[0],self.tile_size[1]))
            return tile  


# Pytorch Lightning
# Faire une class contenant model, crossentropy loss, training step, validation step, optimizer
# 1 class pour dataset, dataloader 

# https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09


# Model

class UNet_lightning(pl.LightningModule):
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
    super(UNet_lightning, self).__init__() #necessary for all classes extending the module class
    #self.is_cuda = cuda
    self.n_class = n_class
    
    ## Encoder 
    
    # Conv2D (input channel, outputchannel, kernel size)
    self.c1 = self.conv_block(3,16)
    self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.c2 = self.conv_block(16,32)
    self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.c3 = self.conv_block(32,64)
    self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)  
    
    self.c4 = self.conv_block(64,128)
    self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)      
    
    self.c5 = self.conv_block(128,256)

    ## Decoder 
    
    # Transpose & UpSampling Convblock   
    self.t6 = nn.ConvTranspose2d(256,128, kernel_size= 2, stride=2)
    self.c6 = self.conv_block(256,128)

    self.t7 = nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
    self.c7 = self.conv_block(128,64)

    self.t8 = nn.ConvTranspose2d(64,32, kernel_size=2, stride=2)
    self.c8 = self.conv_block(64,32)

    self.t9 = nn.ConvTranspose2d(32,16, kernel_size=2, stride=2)
    self.c9 = self.conv_block(32,16)
    
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
    
    
  # Séparer Encoder - Decoder en 2 fonctions ?  
  def forward(self, input):
    """
    the function called to run inference
    """  
    #if self.is_cuda: #put data on GPU
    #    input = input.cuda()

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
    y4 = self.crop(u6,c4)
    concat4 = torch.cat([u6,y4],1)
    x6=self.c6(concat4)
    
    u7=self.t7(x6)
    y3 = self.crop(u7,c3)
    x7=self.c7(torch.cat([u7,y3],1))
    
    u8=self.t8(x7)
    y2 = self.crop(u8,c2)
    x8=self.c8(torch.cat([u8,y2],1))
    
    u9=self.t9(x8)
    y1=self.crop(u9,c1)
    
    x9=self.c9(torch.cat([u9,y1],1))
    
    # Final Output Layer
    out = self.outputs(x9)
    return out


  def conv_block(self,in_channel, out_channel):
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
      
  def crop(self,target_tensor, tensor): # x,c
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

  # cross entropy loss 
  def nn_loss(self,pred, gt):
      loss = nn.CrossEntropyLoss()
      return loss(pred,gt)
      #return nn.CrossEntropyLoss(pred,gt)
    
  # training step 
#  def training_step(self,tiles,gt,batch_id):
  def training_step(self,train_batch,batch_id):
      tiles,gt = train_batch
      #pred = model(tiles)    
      pred = self.forward(tiles)
      loss = self.nn_loss(pred,gt.long())
      self.log('train_loss', loss)
      
      #red_cm = pred.argmax(1)      
      #cm = confusion_matrix(gt.view(-1), pred_cm.view(-1))

      return loss 
      
  # validation step 
  #def validation_step(self,tiles,gt,batch_id):
  def validation_step(self,val_batch,batch_id):
      tiles,gt= val_batch
      pred = self.forward(tiles)
      loss = nn.functional.binary_cross_entropy_with_logits(pred[:,0],gt.float())
      self.log('val_loss', loss)
  
      #pred_cm = pred.argmax(1)
      #cm = confusion_matrix(gt.view(-1), pred_cm.view(-1))

  # configure optimizer 
  def configure_optimizers(self):
     optimizer = optim.Adam(self.parameters(), lr=0.001)
     return optimizer


root = "/home/ign.fr/ttea/Code_IGN/AerialImageDataset"

class InriaDataModule(pl.LightningDataModule):
    
    # set up 
    def setup(self,stage):
        self.train_dataset = InriaDataset(root, args.tile_size, 'train', None, False) 
        self.val_dataset = InriaDataset(root, args.tile_size, 'validation', None, False)
    
    # train dataloader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = 16, num_workers=6,  drop_last=True, shuffle=True)
    
    # val dataloader 
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = 16, num_workers=6,  drop_last=True)
    

#data_module = InriaDataModule()
# train
#model = UNet_lightning()
#trainer = pl.Trainer()

#trainer.fit(model, data_module)

class MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')

    def on_fit_start(self,trainer,pl_module):
        print('fit begin')
        
    def on_fit_end(self,trainer,pl_module):
        print('fit end')
        
    def on_sanity_check_start(self,trainer,pl_module):
        print('validation sanity check start')
    
    def on_sanity_check_end(self,trainer,pl_module):
        print('validation sanity check end')
        
    def on_train_batch_start(self,trainer, pl_module, batch, batch_idx, dataloader_idx):
        print('train batch begin')
    
    def on_train_batch_end(self,trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        print('train batch end')
        
    def on_train_epoch_start(self,trainer,pl_module):
        print('train epoch start')
        
    def on_train_epoch_end(self,trainer,pl_module,unused=None):
        print('train epoch end')
        
    def on_validation_epoch_start(self, trainer,pl_module):
        print('validation epoch start')
        
    def on_validation_epoch_end(self,trainer, pl_module):
        print('validation epoch end')
        
    def on_before_zero_grad(self,trainer,pl_module,optimizer):
        print('before zero grad')
        
    
        

# Training 
def train(model, optimizer, args):
  """
  model : Unet model to train 
  optimizer: optimizer 
  args: arguments defined by Mock
  
  Returns : 
  - cm : confusion matrix  
  - loss_meter : average value meter 
  """  
    
  """train for one epoch"""
  model.train() #switch the model in training mode
  
  #the loader function will take care of the batching
  loader = DataLoader(args.train_dataset, args.batch_size,  drop_last=True, num_workers=4, shuffle=True)

  #tqdm will provide some nice progress bars
  loader = tqdm_nb(loader, ncols=500)
  
  #will keep track of the loss
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrixTorch(args.n_class, class_names = args.class_names, cuda =  model.is_cuda)
  nn_loss = nn.CrossEntropyLoss(reduction="mean")

  for index, (tiles,gt) in enumerate(loader):

    if model.is_cuda:
        gt = gt.cuda()
        nn_loss = nn_loss.cuda()

    optimizer.zero_grad() #put gradient to zero

    #compute the prediction
    pred = model(tiles) 
    
    ## loss = nn.functional.cross_entropy(pred, gt.long())
    loss = nn_loss(pred, gt.long())

    #compute gradients
    loss.backward() 
    
    optimizer.step() #one SGD step

    # if model.is_cuda:
    #    gt = gt.cpu() #back to CPU
    
    loss_meter.add(loss.item())
    
    #need to put the prediction back on the cpu and convert to numpy
    with torch.no_grad():
        cm.add_batch(gt.view(-1), pred.argmax(1).view(-1))
    
  return cm,loss_meter.value()[0]

def eval(model, args):
  """
  model : Unet model to evaluate
  args : arguments defined by Mock
  
  Returns : 
  - cm : Confusion Matrix 
  - loss_meter : average meter value 
  
  """  

    
  """eval on test/validation set"""
  
  model.eval() #switch in eval mode
  
  loader = DataLoader(args.val_dataset, args.batch_size,  num_workers=4, drop_last=True)
  
  loader = tqdm_nb(loader, ncols=500)
  
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrixTorch(args.n_class, class_names = args.class_names,  cuda =  model.is_cuda)

  with torch.no_grad():
    display_idx = [1,]
    # display_idx = random.sample(range(0, len(val_dataset/rgs.batch_size) ), 2)
    for index, (tiles, gt) in enumerate(loader):

      if model.is_cuda:
          gt = gt.cuda()
      
      pred = model(tiles) #compute the prediction
    
      #loss = nn.functional.cross_entropy(pred, gt.long())
      loss = nn.functional.binary_cross_entropy_with_logits(pred[:,0],gt.float())  
      loss_meter.add(loss.item())
      
      # if model.is_cuda:
      # gt = gt.cpu() #back to CPU
      pred_cm = pred.argmax(1)
        
      # Afficher ici Img, Mask, Pred 
      if index in display_idx :
          view_batch(tiles, gt.cpu(), pred = pred_cm.cpu().detach(), size = 4)
        
      #need to put the prediction back on the cpu and convert to numpy
      cm.add_batch(gt.view(-1), pred_cm.view(-1))
    
  return  cm, loss_meter.value()[0]


def train_full(args, model):
  """
  args : arguments defined by Mock
  
  returns : model train & evaluate on epochs 

  """
    
  """The full training loop"""

  #initialize the model
  model = UNet(args.n_channel, args.conv_width, args.n_class, cuda=args.cuda)

  print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
  
  #define the optimizer
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  
  #TESTCOLOR = '\033[104m'
  #TRAINCOLOR = '\033[100m'
  #NORMALCOLOR = '\033[0m'
  
  for i_epoch in range(args.n_epoch):
    #train one epoch
    cm_train, loss_train = train(model, optimizer, args)
    print('Epoch %3d -> Train Overall Accuracy: %3.2f%% Train mIoU : %3.2f%% Train Loss: %1.4f' % (i_epoch, cm_train.overall_accuracy(), cm_train.class_IoU(), loss_train))
    # print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss_train))
    
    if (i_epoch == args.n_epoch - 1) or (args.n_epoch_test != 0 and i_epoch % args.n_epoch_test == 0 and i_epoch > 0):
      #periodic testing
      cm_test, loss_test = eval(model, args)
      print('Test Overall Accuracy: %3.2f%% Test mIoU : %3.2f%%  Test Loss: %1.4f' % (cm_test.overall_accuracy(), cm_test.class_IoU(), loss_test) )
      # print('Epoch %3d -> Test Loss: %1.4f' % (i_epoch, loss_test))
  
  if args.save_model:
    torch.save(model.state_dict(), args.save_model_name)      
  
  return model


# Main 
def main(args):
  
    # Training Model with Full Args 
    model = UNet(args.n_channel, args.conv_width, args.n_class, cuda=args.cuda)
    trained_model = train_full(args, model)
     
if __name__ == "__main__":
    
    print('Training')
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
    
    parser.add_argument('--tile_size',type=int, default = (256,256))
  
    parser.add_argument('--save_model', default= True)
    parser.add_argument('--save_model_name ', default = "unet_bin_crossentropy.pth") 
    
    
    tile_size = (256,256)
    parser.add_argument('--train_dataset', default = InriaDataset(root, tile_size, 'train', None, False))
    parser.add_argument('--val_dataset', default = InriaDataset(root, tile_size, 'validation', None, False))
    
    args = parser.parse_args()

    data_module = InriaDataModule()
    model = UNet_lightning(args.n_channel,args.conv_width,args.n_class)
    trainer = pl.Trainer(callbacks=[MyPrintingCallback()],gpus=1,max_epochs=40)

    trainer.fit(model, data_module)


# configurer git ignore -> enlever cache https://gist.github.com/octocat/9257657
# callback lightning 
