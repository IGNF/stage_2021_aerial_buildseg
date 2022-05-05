#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

# Metrics
class ConfusionMatrix:
  def __init__(self, n_class, class_names):
    self.CM = np.zeros((n_class, n_class))
    self.n_class = n_class
    self.class_names = class_names
  
  def clear(self):
    self.CM = np.zeros((self.n_class, self.n_class))
    
  def add_batch(self, gt, pred):
    self.CM +=  confusion_matrix(gt, pred, labels = list(range(self.n_class)))
    
  def overall_accuracy(self):#percentage of correct classification
    return 100*self.CM.trace() / self.CM.sum()

  def class_IoU(self, show = 1):
    ious = np.diag(self.CM)/ (self.CM.sum(0) + self.CM.sum(1) - np.diag(self.CM))
    if show:
      print('  |  '.join('{} : {:3.2f}%'.format(name, 100*iou) for name, iou in zip(self.class_names,ious)))
    #do not count classes that are not present in the dataset in the mean IoU
    return 100*np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()


# weird trick with bincount
def confusion_matrix_torch(y_true, y_pred, n_class, cuda= False):
    N = n_class
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y_comp = torch.zeros(N * N - len(y), dtype=torch.long)
        if cuda :
          y_comp = y_comp.cuda()
        y = torch.cat([y, y_comp])
    y = y.reshape(N, N)
    return y

class ConfusionMatrixTorch:
  def __init__(self, n_class, class_names, cuda = 0):
    self.CM = torch.zeros((n_class, n_class))
    self.cuda = cuda
    if cuda :
        self.CM = self.CM.cuda()
        
    self.n_class = n_class
    self.class_names = class_names
  
  def clear(self):
    self.CM = torch.zeros((self.n_class, self.n_class))
    
  def add_batch(self, gt, pred):
    self.CM += confusion_matrix_torch(gt, pred, self.n_class, self.cuda)
    
  def overall_accuracy(self):#percentage of correct classification
    return 100*self.CM.trace() / self.CM.sum()

  def class_IoU(self, show = 1):
    if self.cuda :
        cm_array = self.CM.cpu().detach().numpy()
    else :
         cm_array = self.CM.numpy()
    ious = np.diag(cm_array)/ (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
    if show:
      print('  |  '.join('{} : {:3.2f}%'.format(name, 100*iou) for name, iou in zip(self.class_names,ious)))
    #do not count classes that are not present in the dataset in the mean IoU
    return 100*np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()

def BinaryDiceLoss(predict, target):
    """
    predict : prediction -> Float Tensor (nb_batch,width, height)
    target : ground truth -> Float Tensor (nb_batch, width, height)
    
    Returns: Binary Dice loss -> Float Tensor 
    """
    
    smooth = 1
    
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1) + smooth
    den = torch.sum(predict.pow(2) + target.pow(2), dim=1) + smooth

    dice = 2*num / den
    
    loss = 1 - dice
    
    loss = loss.mean() # reduction by mean 
    return loss 