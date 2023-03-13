import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mydataloader.self_dataloader import DataLoader


from utils.dataloader import DeeplabDataset, deeplab_dataset_collate

VOCdevkit_path  = 'VOCdevkit'
input_shape=[480,640]
num_workers=4
num_classes=2
batch_size=4
train_lines=[]
val_lines=[]
train_lines=[]
val_lines=[]
if __name__ == '__main__':

        with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
                train_lines = f.readlines()
        with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
                val_lines = f.readlines()
        n_train   = len(train_lines)
        n_val     = len(val_lines)
        t_line=train_lines
        v_line=val_lines
        train_lines=[]
        val_lines=[]
        for i in range(n_train-batch_size):
                for j in range(batch_size):
                        train_lines.append(t_line[i+j])
        for i in range(n_val-batch_size):
                for j in range(batch_size):
                        val_lines.append(v_line[i+j])
        num_train=len(train_lines)
        num_val=len(val_lines)


        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        gen             = DataLoader(train_dataset, shuffle =False, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                        drop_last = True, collate_fn = deeplab_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle =False, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                        drop_last = True, collate_fn = deeplab_dataset_collate)
        imgpre=torch.zeros([4,3,480,640])
        j=0
        for iteration, batch in enumerate(gen):
                imgs, pngs, labels = batch
                print(imgs[0,1,1,1],imgs[1,1,1,1],imgs[2,1,1,1],imgs[3,1,1,1])


                

                