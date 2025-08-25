from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
import h5py
from .lidar_utils import point_cloud_to_range_image

class HDVMineGen(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.random_roll = False
        self.full_list = glob(os.path.join('HDVMineData', 'Penrice*.h5'))
        self.batchSize = config.sampling.batch_size
        #No test:train split yet, just make it fucking work
        # if split == "train":
        #     self.full_list = list(filter(lambda file: '0000_sync' not in file and '0001_sync' not in file, full_list))
        # else:
        #     self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        self.length = 20#len(self.full_list)
        pointArray = []
        # LabelArray = []
        intensityArray = []
        for file in self.full_list:
            openedFile = h5py.File(file, 'r')
            pointArray.append(openedFile['Input'][:,:3])
            colour=(openedFile['Input'][:,3:6])
            intensityArray.append(0.3*colour[:,0] + 0.6*colour[:,0] + 0.11*colour[:,2])
            # LabelArray.append(openedFile['Input'])
        # print(pointArray[0].shape)
        # x = a/0
        self.points = np.concatenate(pointArray,axis=0)
        intensity = np.concatenate(intensityArray,axis=0)/255
        self.points = np.concatenate((self.points, np.expand_dims(intensity,axis=1)),axis=1)
        # self.labels = None
        # self.labels = np.concatenate(LabelArray,axis=0)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # filename = self.full_list[idx]
        maxRange = 2057.701
        trueIdx = idx % self.batchSize
        folderIdx = idx // self.batchSize   
        origin = self.origins[trueIdx]
        if self.return_remission:
            real, intensity, mask, sky, index = point_cloud_to_range_image(self.points, True, self.return_remission, self.origins)
        else:
            real, mask, sky, index = point_cloud_to_range_image(self.points, False, self.return_remission, self.origins)
        #Make empty cells 0
        real = np.where(real>=maxRange, 0, real) + 0.0001
        #Apply log. As my max is now 2057... let's treat (2047+1) as the max. That would result in 11 after the log.
        real = ((np.log2(real+1)) / 11)
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
        sky = np.expand_dims(sky, axis = 0)
        index = np.expand_dims(index, axis = 0)

        if self.return_remission:
            intensity = np.clip(intensity, 0, 1.0)
            if self.random_roll:
                intensity = np.roll(intensity, random_roll, axis = 1)
            intensity = np.expand_dims(intensity, axis = 0)
            real = np.concatenate((real, intensity), axis = 0)

        #logical not the mask, as I want mask to retain only points which are visible
        #Now let's just take the first half because hahahaha I'm GPU poor
        # real = real[:,:,:225]
        # mask = mask[:,:,:225]
        return real, np.logical_not(mask), np.logical_not(sky), index

