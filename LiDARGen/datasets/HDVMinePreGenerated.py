from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
import h5py
from .lidar_utils import point_cloud_to_range_image

class HDVMinePreGen(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.random_roll = True
        self.full_list = glob('/data/PreGenFinal/Depth/*')
        # print("FIRST FILEs ARE")
        # print(self.full_list[0])
        # print(self.full_list[1])
        # print(self.full_list[2])
        #No test:train split yet, just make it fucking work
        if split == "train":
            self.full_list = self.full_list[:len(self.full_list)*9//10]
        else:
            self.full_list = self.full_list[len(self.full_list)*9//10:]
        self.length = len(self.full_list)
        # pointArray = []
        # LabelArray = []
        # intensityArray = []
        # for file in self.full_list:
        #     openedFile = h5py.File(file, 'r')
        #     pointArray.append(openedFile['Input'][:,:3])
        #     colour=(openedFile['Input'][:,3:6])
        #     intensityArray.append(0.3*colour[:,0] + 0.6*colour[:,0] + 0.11*colour[:,2])
        #     # LabelArray.append(openedFile['Input'])
        # print(pointArray[0].shape)
        # x = a/0
        # self.points = np.concatenate(pointArray,axis=0)
        # intensity = np.concatenate(intensityArray,axis=0)/255
        # self.points = np.concatenate((self.points, np.expand_dims(intensity,axis=1)),axis=1)
        # self.labels = None
        # self.labels = np.concatenate(LabelArray,axis=0)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # filename = self.full_list[idx] + "/Numpy/"
        filename = self.full_list[idx].split('/')[-1]
        maxRange = 2057
        if self.return_remission:
            # real, intensity, mask, sky = point_cloud_to_range_image(self.points, True, self.return_remission)
            real = np.load('/data/PreGenFinal/Depth/' + filename)
            mask = np.load('/data/PreGenFinal/Mask/' + filename)
            intensity = np.load('/data/PreGenFinal/Intensity/' + filename)
        else:
            #If not using intensity, can use overlapped scans without issue
            # real = np.load(filename + 'depth_0.npy')
            real = np.load('/data/PreGenFinal/Depth/' + filename)
            mask = np.load('/data/PreGenFinal/Mask/' + filename)
            # intensity = np.load(filename + 'intensity_0.npy')
        mask = np.where(real>=maxRange, 1, mask)
        #Other works set Sky to 0, so I'll do the same for now
        real = np.where(real>=maxRange, 0, real) + 0.0001
        #Apply log. As my max is now 2057... let's treat (2047+1) as the max. That would result in 11 after the log.
        real = ((np.log2(real+1)) / 11)
        #We are no longer logging because my 3D stuff HATES it
        # real = real / 2000
        # real = real / 2056
        # real = ((np.log2(real+1)) / 6)
        #Make negatives 0 and the rare few over 2047, be 1
        real = np.clip(real, 0, 1)
        #Random roll lets me have range image always generated pointing north, then randomly rotate the 1800 pixel width by 0...1800 pixels
        random_roll = np.random.randint(1800)



        if self.random_roll:
            real = np.roll(real, random_roll, axis = 1)
            mask = np.roll(mask, random_roll, axis = 1)
            # label = np.roll(label, random_roll, axis = 1)
        real = np.expand_dims(real, axis = 0)
        mask = np.expand_dims(mask, axis = 0)
        # mask = np.zeros_like(real).astype(int)
        # mask[:, 0:640:4, :] = 1
        #mask = np.logical_not(mask)
        # sky = np.expand_dims(sky, axis = 0)

        if self.return_remission:
            # intensity = ((np.log2(intensity+1)) / 11)
            intensity = intensity / 2000
            #Any point with an intensity over 5000 is an error, ditch it
            mask = np.where(intensity>=1, 1, mask)
            intensity = np.where(intensity>=1, 0, intensity) + 0.0001
            intensity = np.clip(intensity, 0, 1.0)
            # colour = colour / 255
            # colour = np.clip(colour,0,1)
            if self.random_roll:
                intensity = np.roll(intensity, random_roll, axis = 1)
                # intensity = intensity[:self.rowMax,:self.colMax]
                # colour = np.roll(colour, random_roll, axis = 1)

            # real = real[:,:self.rowMax,:self.colMax]
            # mask = mask[:,:self.rowMax,:self.colMax]
            intensity = np.expand_dims(intensity, axis = 0)
            # real = np.concatenate((real, intensity,colour), axis = 0)
            real = np.concatenate((real, intensity), axis = 0)
            mask = np.concatenate((mask,mask), axis = 0)

        #logical not the mask, as I want mask to retain only points which are visible
        #Now let's just take the first half because hahahaha I'm GPU poor
        # real = real[:,:,:225]
        # mask = mask[:,:,:225]
        sky = np.zeros(1)
        return real, np.logical_not(mask), sky


