from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
import h5py
from .lidar_utils import point_cloud_to_range_image

class HDVMineGenerateFromInvidivualScans(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.random_roll = config.data.random_roll
        self.points = []
        self.origins = []
        self.modifications = np.array(config.data.modifications)
        self.batchSize = config.sampling.batch_size
        self.rowMax = config.data.image_size
        self.colMax = config.data.image_width
        # self.originMultiplierArray = [1,2,3,4,5,6,7,8]
        self.originMultiplierArray = [1,1,1,1,1,1,1,1]
        self.full_list = glob(os.path.join('RawScans', 'Scans/*.npy'))
        if split == "train":
            self.saveNum = 1
            self.full_list = self.full_list[:len(self.full_list)*6//10]
        else:
            self.saveNum = -1
            # self.full_list = self.full_list[len(self.full_list)*6//10:len(self.full_list)*8//10]
            self.full_list = self.full_list[len(self.full_list)*6//10:]
        # self.length = len(self.full_list)
        #No test:train split yet, just make it fucking work
        # if split == "train":
        #     self.full_list = list(filter(lambda file: '0000_sync' not in file and '0001_sync' not in file, full_list))
        # else:
        #     self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        # self.length = len(self.full_list)
        # self.length = 5000
        pointArray = []
        # LabelArray = []
        intensityArray = []
        print("the file order")
        for file in self.full_list:
            filename = file.split('/')[-1]
            if(filename == "20100603 penrice_stn15_nth_C8L_02.npy"):
                print("skipped it")
                continue
            print(filename)
            origin = np.load("RawScans/Origins/" + filename)
            #for h5py
            # openedFile = h5py.File(file, 'r')
            # pointArray.append(openedFile['Input'][:,:3])
            # colour=(openedFile['Input'][:,3:6])
            # intensityArray.append(0.3*colour[:,0] + 0.6*colour[:,0] + 0.11*colour[:,2])
            #for npy
            data = np.load(file)
            self.points.append(data)
            self.origins.append(origin)

        self.length = len(self.points) * self.batchSize
            # LabelArray.append(openedFile['Input'])
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

        #THis currently DOES NOT WORK
        #Firstly the indices I return are fucked - for method 1 that's fine I guess
        #But I need to edit this for method 2 to also return the currntcount and oldindices I guess?

        # filename = self.full_list[idx]
        scanOrigin = idx // self.batchSize
        scanOrigin = scanOrigin % (len(self.points))
        #Just make it always scan 2
        modScale = scanOrigin + 1
        scanOrigin = 2

        maxRange = 2057.701
        modIdx = idx % self.batchSize
        origin = self.origins[scanOrigin] + (self.modifications[modIdx] * (modScale))
        print(modScale)
        if self.return_remission:
            real, intensity, mask, saveNum, sky, index  = point_cloud_to_range_image(self.points[scanOrigin], origin, self.return_remission, rowMax = self.rowMax, colMax = self.colMax, saveNum = self.saveNum)
        else:
            real, mask, saveNum, sky, index = point_cloud_to_range_image(self.points[scanOrigin], origin, self.return_remission, rowMax = self.rowMax, colMax = self.colMax, saveNum = self.saveNum)
        self.saveNum = saveNum
        #Ok so new plan for training - add the ENTIRE sky to the mask
        #Fuck sky points, I'll deal with that later
        #Because otherwise due to distance difference being so much larger, they get a stupidly high loss and the network predicts sky for 99% of points
        mask = np.where(real>=maxRange, 1, mask)
        #Other works set Sky to 0, so I'll do the same for now
        real = np.where(real>=maxRange, 0, real) + 0.0001
        #Apply log. As my max is now 2057... let's treat (2047+1) as the max. That would result in 11 after the log.
        real = ((np.log2(real+1)) / 11)
        # real = real / 2000
        # real = ((np.log2(real+1)) / 6)
        #Make negatives 0 and the rare few over 2047, be 1
        real = np.clip(real, 0, 1)
        #Random roll lets me have range image always generated pointing north, then randomly rotate the 1800 pixel width by 0...1800 pixels
        random_roll = np.random.randint(self.colMax)



        if self.random_roll:
            real = np.roll(real, random_roll, axis = 1)
            mask = np.roll(mask, random_roll, axis = 1)
            sky = np.roll(sky, random_roll, axis = 1)
            # label = np.roll(label, random_roll, axis = 1)
        real = np.expand_dims(real, axis = 0)
        mask = np.expand_dims(mask, axis = 0)
        maskTwo = np.zeros_like(real).astype(int)
        maskThree = np.zeros_like(real).astype(int)
        maskTwo[:, 0:self.rowMax:2, :] = 1
        maskThree[:, :, 0:self.colMax:2] = 1
        maskTwo = np.logical_and(maskTwo,maskThree)
        maskTwo = np.logical_not(maskTwo)
        #This line makes it densification
        #VERY IMPORTANT LINE
        #COMMENT OUT IF DONT WANT DENSIFICATION
        # mask = np.logical_or(mask,maskTwo)

        #Ok so sky needs to have the an extra row added because the current heuristic falls JUST shy
        #Soooo just shift it down one
        sky[1:] = sky[:-1]
        #let's do this two more times for good measure
        sky[1:] = sky[:-1]
        sky[1:] = sky[:-1]

        sky = np.expand_dims(sky, axis = 0)
        index = np.expand_dims(index, axis = 0)

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
                intensity = intensity[:self.rowMax,:self.colMax]
                # colour = np.roll(colour, random_roll, axis = 1)

            real = real[:,:self.rowMax,:self.colMax]
            mask = mask[:,:self.rowMax,:self.colMax]
            sky = sky[:,:self.rowMax,:self.colMax]
            intensity = np.expand_dims(intensity, axis = 0)
            # real = np.concatenate((real, intensity,colour), axis = 0)
            real = np.concatenate((real, intensity), axis = 0)
            mask = np.concatenate((mask,mask), axis = 0)

        #logical not the mask, as I want mask to retain only points which are visible
        #Now let's just take the first half because hahahaha I'm GPU poor
        # real = real[:,:,:225]
        # mask = mask[:,:,:225]
        # sky = np.zeros(1)
        returnedScale = np.expand_dims(modScale,axis=0) 
        return real, np.logical_not(mask), np.logical_not(sky), index, returnedScale
        # return real, mask, sky

