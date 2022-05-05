#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Import Librairies 

import torch
import torch.nn as nn
import torch.optim as optim

import torchnet as tnt
from tqdm.notebook import tqdm as tqdm_nb
from torch.utils.data import DataLoader

import argparse

from utils import view_batch
from loss.loss import ConfusionMatrixTorch, ConfusionMatrix, confusion_matrix_torch, BinaryDiceLoss
from dataloader.dataloader import InriaDataset
from model.model import UNet

import pandas as pd 


var= pd.read_json('variables.json')

def parse_args():
    
    parser = argparse.ArgumentParser()
    tile_size = (512,512)
    
    # Hyperparameter
    parser.add_argument('--n_epoch', default = 40)
    parser.add_argument('--n_epoch_test',type = int ,default = int(5)) #periodicity of evaluation on test set
    parser.add_argument('--batch_size',type = int, default = 8)
    parser.add_argument('--conv_width',default = [16,32,64,128,256,128,64,32,16])
    parser.add_argument('--cuda',default = 1)
    parser.add_argument('--lr', default = 0.0001)
    
    #parser.add_argument('--nn_loss',default = nn.BCEWithLogitsLoss(reduction="mean"))
    parser.add_argument('--nn_loss',default = nn.CrossEntropyLoss(reduction="mean"))
    parser.add_argument('--threshold', default=0.5) 
    
    parser.add_argument('--n_class',type = int, default = 2)
    parser.add_argument('--n_channel',type=int, default = 3)
    parser.add_argument('--class_names' , default= ['None','Batiment'])
    
    parser.add_argument('--save_model', default= True)
    parser.add_argument('--save_model_name ', default = "unet_fcrossentropy_code.pth") 
   
    args = parser.parse_args("")
    return args

#args = parse_args()

def train(model, optimizer, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel):
  """
  model : Unet model to train 
  optimizer: optimizer 
  args: arguments defined by Mock
  
  Returns : 
  - cm : confusion matrix  
  - loss_meter : average value meter 
  """  
    
  cm_value = {"TP":[],"FP":[],"FN":[],"TN":[],"TPR":[],"FPR":[]}  
    
  """train for one epoch"""
  model.train() #switch the model in training mode
  
  #the loader function will take care of the batching
  loader = DataLoader(args['train_dataset'], batch_size,  drop_last=True, num_workers=4, shuffle=True)

  #tqdm will provide some nice progress bars
  loader = tqdm_nb(loader, ncols=500)
  
  #will keep track of the loss
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrixTorch(2, class_names = args['class_names'], cuda =  model.is_cuda)

  for index, (tiles,gt) in enumerate(loader):

    if model.is_cuda:
        gt = gt.cuda()
        
        if args['loss_name'] == 'BinaryDiceLoss':
            pass
        elif args['loss_name'] == 'BCEDiceLoss':
            pass
        else: 
            nn_loss = args['nn_loss'].cuda()

    optimizer.zero_grad() #put gradient to zero

    # Compute the prediction : Tiles (N_Batch, N_class, Width , Height)
    pred = model(tiles) 
    
    if n_class == 2:
        # Crossentropy
        loss = args['nn_loss'](pred,gt.long())
    else:
        # BCE
        if args['loss_name'] == 'BinaryCrossentropy':
            loss = args['nn_loss'](pred, torch.unsqueeze(gt.float(),1))
            
        elif args['loss_name'] == 'BCEDiceLoss':
            loss = args['nn_loss'](pred,gt)
        # Dice
        else: 
            loss = args['nn_loss'](pred,gt.float())
     
    # Compute gradients
    loss.backward() 
    
    optimizer.step() #one SGD step

    # if model.is_cuda:
    #    gt = gt.cpu() #back to CPU
    
    loss_meter.add(loss.item())
    
    if n_class == 1:
        pred_bin = (pred>args['threshold']).to(torch.float32)
        
    with torch.no_grad():
        
        if n_class ==1: 
            cm.add_batch(gt.view(-1),pred_bin.view(-1).long())
            
        # 2 classes 
        else: 
            cm.add_batch(gt.view(-1), pred.argmax(1).view(-1))
    
  return cm,loss_meter.value()[0], cm_value

def eval(model, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel):
  """
  model : Unet model to evaluate
  args : arguments defined by Mock
  
  Returns : 
  - cm : Confusion Matrix 
  - loss_meter : average meter value 
  
  """  

  """eval on test/validation set"""

  cm_value = {"TP":[],"FP":[],"FN":[],"TN":[],"TPR":[],"FPR":[]} 
  
  model.eval() #switch in eval mode
  
  loader = DataLoader(args['val_dataset'], batch_size,  num_workers=4, drop_last=True)
  
  loader = tqdm_nb(loader, ncols=500)
  
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrixTorch(2, class_names = args['class_names'],  cuda =  model.is_cuda)

  with torch.no_grad():
    display_idx = [1,]
    # display_idx = random.sample(range(0, len(val_dataset/rgs.batch_size) ), 2)
    for index, (tiles, gt) in enumerate(loader):

      if model.is_cuda:
          gt = gt.cuda()
      
      pred = model(tiles) #compute the prediction
        
      if n_class == 2 :
          loss = args['nn_loss'](pred, gt.long())
      else: 
          if args['loss_name'] == 'BinaryCrossentropy':
              loss = args['nn_loss'](pred,torch.unsqueeze(gt.float(),1))
            
          elif args['loss_name'] == 'BCEDiceLoss':
              loss = args['nn_loss'](pred,gt)
          else : 
              loss = args['nn_loss'](pred,gt.float())
        
      loss_meter.add(loss.item())
      
      # if model.is_cuda:
      # gt = gt.cpu() #back to CPU
      
        
      if n_class == 2 : 
          pred_cm = pred.argmax(1)
          cm.add_batch(gt.view(-1), pred_cm.view(-1))
        
      else : 
          pred_bin = (pred>args['threshold']).to(torch.float32)
          cm.add_batch(gt.view(-1),pred_bin.view(-1).long())  
            
      # Afficher ici Img, Mask, Pred 
      #if index in display_idx :
      #    view_batch(tiles, gt.cpu(), pred = pred_cm.cpu().detach(), size = 4)
      
      #need to put the prediction back on the cpu and convert to numpy
      #cm.add_batch(gt.view(-1), pred_cm.view(-1))
      #cm.add_batch(gt.view(-1),pred_bin.view(-1).long())  

  return  cm, loss_meter.value()[0],cm_value


def train_full(args, model,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel):
  """
  args : arguments defined by Mock
  
  returns : model train & evaluate on epochs 

  """
    
  """The full training loop"""

  metrics_train = {"accuracy":[],"mIoU":[],"loss":[]}
  metrics_test = {"accuracy":[],"mIoU":[],"loss":[]}

  #model = UNet(n_channel, conv_width, n_class, cuda=args['cuda'])
  
    
  print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
  
  #define the optimizer
  optimizer = optim.Adam(model.parameters(), lr=lr)
  
  for i_epoch in range(n_epoch):
    #train one epoch
    cm_train, loss_train,cm_value = train(model, optimizer, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel)
    print('Epoch %3d -> Train Overall Accuracy: %3.2f%% Train mIoU : %3.2f%% Train Loss: %1.4f' % (i_epoch, cm_train.overall_accuracy(), cm_train.class_IoU(), loss_train))
    # print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss_train))
    
    metrics_train['accuracy'].append(cm_train.overall_accuracy())  
    metrics_train['mIoU'].append(cm_train.class_IoU())
    metrics_train['loss'].append(loss_train)
    
    if (i_epoch == n_epoch - 1) or (n_epoch_test != 0 and i_epoch % n_epoch_test == 0 and i_epoch > 0):
      #periodic testing
      cm_test, loss_test,cm_value = eval(model, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel)
      print('Test Overall Accuracy: %3.2f%% Test mIoU : %3.2f%%  Test Loss: %1.4f' % (cm_test.overall_accuracy(), cm_test.class_IoU(), loss_test) )
      metrics_test['accuracy'].append(cm_test.overall_accuracy())  
      metrics_test['mIoU'].append(cm_test.class_IoU())
      metrics_test['loss'].append(loss_test)
        
  if args['save_model']:
    torch.save(model.state_dict(), args['save_model_name'])      
  
  return model,metrics_train,metrics_test


#-------------------------------------------------------------------------------------------------------

# Training with dropout 

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()
        
def eval_dropout(model, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel):
  """
  model : Unet model to evaluate
  args : arguments defined by Mock
  
  Returns : 
  - cm : Confusion Matrix 
  - loss_meter : average meter value 
  
  """  

  """eval on test/validation set"""

  cm_value = {"TP":[],"FP":[],"FN":[],"TN":[],"TPR":[],"FPR":[]} 
  
  model.eval() #switch in eval mode
  model.apply(apply_dropout)
    
  loader = DataLoader(args['val_dataset'], batch_size,  num_workers=4, drop_last=True)
  
  loader = tqdm_nb(loader, ncols=500)
  
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrixTorch(2, class_names = args['class_names'],  cuda =  model.is_cuda)

  with torch.no_grad():
    display_idx = [1,]
    # display_idx = random.sample(range(0, len(val_dataset/rgs.batch_size) ), 2)
    for index, (tiles, gt) in enumerate(loader):

      if model.is_cuda:
          gt = gt.cuda()
      
      pred = model(tiles) #compute the prediction
        
      if n_class == 2 :
          loss = args['nn_loss'](pred, gt.long())
      else: 
          if args['loss_name'] == 'BinaryCrossentropy':
              loss = args['nn_loss'](pred,torch.unsqueeze(gt.float(),1))
            
          elif args['loss_name'] == 'BCEDiceLoss':
              loss = args['nn_loss'](pred,gt)
          else : 
              loss = args['nn_loss'](pred,gt.float())
        
      loss_meter.add(loss.item())
        
      if n_class == 2 : 
          pred_cm = pred.argmax(1)
          cm.add_batch(gt.view(-1), pred_cm.view(-1))
        
      else : 
          pred_bin = (pred>args['threshold']).to(torch.float32)
          cm.add_batch(gt.view(-1),pred_bin.view(-1).long())  
            
      # Afficher ici Img, Mask, Pred 
      #if index in display_idx :
      #    view_batch(tiles, gt.cpu(), pred = pred_cm.cpu().detach(), size = 4)
      
      #need to put the prediction back on the cpu and convert to numpy
      #cm.add_batch(gt.view(-1), pred_cm.view(-1))
      #cm.add_batch(gt.view(-1),pred_bin.view(-1).long())  

  return  cm, loss_meter.value()[0],cm_value

#%%

#---------------------------------------------------------------------------------------------------

# Training for Resnet18 & EfficientNet

def train_segmentation(model, optimizer, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel):
  """
  model : Unet model to train 
  optimizer: optimizer 
  args: arguments defined by Mock
  
  Returns : 
  - cm : confusion matrix  
  - loss_meter : average value meter 
  """  
    
  cm_value = {"TP":[],"FP":[],"FN":[],"TN":[],"TPR":[],"FPR":[]}  
    
  """train for one epoch"""
  model.train() #switch the model in training mode
  
  #the loader function will take care of the batching
  loader = DataLoader(args['train_dataset'], batch_size,  drop_last=True, num_workers=4, shuffle=True)

  #tqdm will provide some nice progress bars
  loader = tqdm_nb(loader, ncols=500)
  
  #will keep track of the loss
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrixTorch(2, class_names = args['class_names'], cuda = 1)

  for index, (tiles,gt) in enumerate(loader):
        
    if args['cuda']==1 :
        gt = gt.cuda()
        
        if args['loss_name'] == 'BinaryDiceLoss':
            pass
        elif args['loss_name'] == 'BCEDiceLoss':
            pass
        else: 
            nn_loss = args['nn_loss'].cuda()

    optimizer.zero_grad() #put gradient to zero

    # Compute the prediction : Tiles (N_Batch, N_class, Width , Height)
    
    model = model.cuda()
    pred = model(tiles) 
    
    #if args['model_name'] != 'UNet':
    #    model.cuda()
    
    pred = pred.cuda()
    
    if n_class == 2:
        # Crossentropy
        loss = args['nn_loss'](pred,gt.long())
    else:
        # BCE
        if args['loss_name'] == 'BinaryCrossentropy':
            loss = args['nn_loss'](pred, torch.unsqueeze(gt.float(),1))
            
        elif args['loss_name'] == 'BCEDiceLoss':
            loss = args['nn_loss'](pred,gt)
        # Dice
        else: 
            loss = args['nn_loss'](pred,gt.float())
     
    # Compute gradients
    loss.backward() 
    
    optimizer.step() #one SGD step
    
    loss_meter.add(loss.item())
    
    if n_class == 1:
        pred_bin = (pred>args['threshold']).to(torch.float32)
        
    with torch.no_grad():
        
        if n_class ==1: 
            cm.add_batch(gt.view(-1),pred_bin.view(-1).long())
            
        # 2 classes 
        else: 
            cm.add_batch(gt.view(-1), pred.argmax(1).view(-1))
    
  return cm,loss_meter.value()[0], cm_value

def eval_segmentation(model, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel):
  """
  model : Unet model to evaluate
  args : arguments defined by Mock
  
  Returns : 
  - cm : Confusion Matrix 
  - loss_meter : average meter value 
  
  """  

  """eval on test/validation set"""

  cm_value = {"TP":[],"FP":[],"FN":[],"TN":[],"TPR":[],"FPR":[]} 
  
  model.eval() #switch in eval mode
  
  loader = DataLoader(args['val_dataset'], batch_size,  num_workers=4, drop_last=True)
  
  loader = tqdm_nb(loader, ncols=500)
  
  loss_meter = tnt.meter.AverageValueMeter()
  cm = ConfusionMatrixTorch(2, class_names = args['class_names'],  cuda =  1)

  with torch.no_grad():
    display_idx = [1,]
    # display_idx = random.sample(range(0, len(val_dataset/rgs.batch_size) ), 2)
    for index, (tiles, gt) in enumerate(loader):

      if args['model_name'] != 'UNet':
          model.cuda()
        
      if args['cuda']==1:
          gt = gt.cuda()
          
      
      pred = model(tiles) #compute the prediction
      pred = pred.cuda()  
       
      if n_class == 2 :
          loss = args['nn_loss'](pred, gt.long())
      else: 
          if args['loss_name'] == 'BinaryCrossentropy':
              loss = args['nn_loss'](pred,torch.unsqueeze(gt.float(),1))
            
          elif args['loss_name'] == 'BCEDiceLoss':
              loss = args['nn_loss'](pred,gt)
          else : 
              loss = args['nn_loss'](pred,gt.float())
        
      loss_meter.add(loss.item())
        
      if n_class == 2 : 
          pred_cm = pred.argmax(1)
          cm.add_batch(gt.view(-1), pred_cm.view(-1))
        
      else : 
          pred_bin = (pred>args['threshold']).to(torch.float32)
          cm.add_batch(gt.view(-1),pred_bin.view(-1).long())  
            
  return  cm, loss_meter.value()[0],cm_value


def train_full_segmentation(args, model,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel):
  """
  args : arguments defined by Mock
  
  returns : model train & evaluate on epochs 

  """
    
  """The full training loop"""

  metrics_train = {"accuracy":[],"mIoU":[],"loss":[]}
  metrics_test = {"accuracy":[],"mIoU":[],"loss":[]}

  print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
  
  #define the optimizer
  optimizer = optim.Adam(model.parameters(), lr=lr)
  
  for i_epoch in range(n_epoch):
    #train one epoch
    cm_train, loss_train,cm_value = train_segmentation(model, optimizer, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel)
    print('Epoch %3d -> Train Overall Accuracy: %3.2f%% Train mIoU : %3.2f%% Train Loss: %1.4f' % (i_epoch, cm_train.overall_accuracy(), cm_train.class_IoU(), loss_train))
    # print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss_train))
    
    metrics_train['accuracy'].append(cm_train.overall_accuracy())  
    metrics_train['mIoU'].append(cm_train.class_IoU())
    metrics_train['loss'].append(loss_train)
    
    if (i_epoch == n_epoch - 1) or (n_epoch_test != 0 and i_epoch % n_epoch_test == 0 and i_epoch > 0):
      #periodic testing
      cm_test, loss_test,cm_value = eval_segmentation(model, args,lr,n_epoch,n_epoch_test,batch_size,n_class,n_channel)
      print('Test Overall Accuracy: %3.2f%% Test mIoU : %3.2f%%  Test Loss: %1.4f' % (cm_test.overall_accuracy(), cm_test.class_IoU(), loss_test) )
      metrics_test['accuracy'].append(cm_test.overall_accuracy())  
      metrics_test['mIoU'].append(cm_test.class_IoU())
      metrics_test['loss'].append(loss_test)
        
  if args['save_model']:
    torch.save(model.state_dict(), args['save_model_name'])      
  
  return model,metrics_train,metrics_test
