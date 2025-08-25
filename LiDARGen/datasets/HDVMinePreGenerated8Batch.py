from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
import h5py
from .lidar_utils import point_cloud_to_range_image

class HDVMinePreGenerated8Batch(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.batchSize = config.sampling.batch_size
        self.random_roll = False
        self.rowMax = config.data.image_size
        self.split = split
        self.colMax = config.data.image_width
        self.split = split
        #No test:train split yet, just make it fucking work
        if split == "train":
            self.full_list = glob('/data/PreGenFinal/PreGenFinal/Depth/*')
            print("FIRST FILEs ARE")
            print(self.full_list[0])
            print(self.full_list[1])
            print(self.full_list[2])
            # self.full_list = self.full_list[:len(self.full_list)*6//10]
        else:
            self.full_list = glob('/data/PreGenFinalVal/Depth/*')
            print("FIRST FILEs ARE")
            print(self.full_list[0])
            print(self.full_list[1])
            print(self.full_list[2])
            # self.full_list = self.full_list[len(self.full_list)*6//10:len(self.full_list)*8//10]
        self.length = len(self.full_list) * self.batchSize #multiply by 8 because there are 8 viewpoints for each one
        print("actual length is")
        print(self.length / self.batchSize)
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
        trueIdx = idx % self.batchSize
        folderIdx = idx // self.batchSize
        # print(idx)
        # print(folderIdx)
        # print(len(self.full_list))
        # print(self.full_list[folderIdx])
        folderName = self.full_list[folderIdx].split('/')[-1]
        filename = folderName
        maxRange = 2057
        if self.split == "train":
            if self.return_remission:
                # real, intensity, mask, sky = point_cloud_to_range_image(self.points, True, self.return_remission)
                real = np.load('/data/PreGenFinal/PreGenFinal/Depth/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                mask = np.load('/data/PreGenFinal/PreGenFinal/Mask/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                intensity = np.load('/data/PreGenFinal/PreGenFinal/Intensity/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                # mask = np.load(filename + 'GTmask_0.npy')
            else:
                #If not using intensity, can use overlapped scans without issue
                # real = np.load(filename + 'depth_0.npy')
                real = np.load('/data/PreGenFinal/PreGenFinal/Depth/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                mask = np.load('/data/PreGenFinal/PreGenFinal/Mask/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                intensity = np.load('/data/PreGenFinal/PreGenFinal/Intensity/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
        else:
            if self.return_remission:
                # real, intensity, mask, sky = point_cloud_to_range_image(self.points, True, self.return_remission)
                real = np.load('/data/PreGenFinalVal/Depth/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                mask = np.load('/data/PreGenFinalVal/Mask/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                intensity = np.load('/data/PreGenFinalVal/Intensity/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                # mask = np.load(filename + 'GTmask_0.npy')
            else:
                #If not using intensity, can use overlapped scans without issue
                # real = np.load(filename + 'depth_0.npy')
                real = np.load('/data/PreGenFinalVal/Depth/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                mask = np.load('/data/PreGenFinalVal/Mask/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                intensity = np.load('/data/PreGenFinalVal/Intensity/' + folderName + '/' + str(trueIdx) + '.npy')[:self.rowMax,:self.colMax]
                # mask = np.load(filename + 'mask_0.npy')
        
        # while(np.sum(real[np.logical_not(mask)] < 2057) < real.shape[0] * real.shape[1] // 6):
        #     idx = np.random.randint(len(self.full_list))
        #     print("too few points, skipped: " + str(np.sum(real[np.logical_not(mask)] < 2057)))
        #     filename = self.full_list[-idx].split('/')[-1]
        #     real = np.load('PreGenImages/Depth/' + filename)
        #     mask = np.load('PreGenImages/Mask/' + filename)
        #     if (self.return_remission):
        #         intensity = np.load('PreGenImages/Intensity/' + filename)
        #Make empty cells 0
        # print("total mask is")
        # print(np.sum(mask))
        # print("total pixels are")
        # print(mask.shape[0] * mask.shape[1])

        # # Comment this out if no need to edit resolution -------------------------------------------------------
        # for halveTimes in range(2):
        #     newReal = np.zeros((real.shape[0]//2,real.shape[1]//2))
        #     newIntensity = np.zeros((real.shape[0]//2,real.shape[1]//2))
        #     newMask = np.zeros((real.shape[0]//2,real.shape[1]//2))
        #     for row in range(real.shape[0]//2): #800 -> 200
        #         for column in range(real.shape[1]//2):
        #             newIdx = np.argmin([real[row*2,column*2],real[row*2+1,column*2],real[row*2,column*2+1],real[row*2+1,column*2+1]])
        #             if(newIdx == 0):
        #                 newReal[row,column] = real[row*2,column*2]
        #                 newMask[row,column] = mask[row*2,column*2]
        #                 if(self.return_remission):
        #                     newIntensity[row,column] = intensity[row*2,column*2]
        #             elif(newIdx == 1):
        #                 newReal[row,column] = real[row*2+1,column*2]
        #                 newMask[row,column] = mask[row*2+1,column*2]
        #                 if(self.return_remission):
        #                     newIntensity[row,column] = intensity[row*2+1,column*2]
        #             elif(newIdx == 2):
        #                 newReal[row,column] = real[row*2,column*2+1]
        #                 newMask[row,column] = mask[row*2,column*2+1]
        #                 if(self.return_remission):
        #                     newIntensity[row,column] = intensity[row*2,column*2+1]
        #             elif(newIdx == 3):
        #                 newReal[row,column] = real[row*2+1,column*2+1]
        #                 newMask[row,column] = mask[row*2+1,column*2+1]
        #                 if(self.return_remission):
        #                     newIntensity[row,column] = intensity[row*2+1,column*2+1]
        #             else:
        #                 print("what the fuck")
        #                 print(newIdx)
        #     real = newReal
        #     mask = newMask
        #     if(self.return_remission):
        #         intensity = newIntensity

        # newReal = np.zeros((real.shape[0]//2,real.shape[1]))
        # newIntensity = np.zeros((real.shape[0]//2,real.shape[1]))
        # newMask = np.zeros((real.shape[0]//2,real.shape[1]))
        # for row in range(real.shape[0]//2): #800 -> 200
        #     for column in range(real.shape[1]):
        #         newIdx = np.argmin([real[row*2,column],real[row*2+1,column]])
        #         if(newIdx == 0):
        #             newReal[row,column] = real[row*2,column]
        #             newMask[row,column] = mask[row*2,column]
        #             if(self.return_remission):
        #                 newIntensity[row,column] = intensity[row*2,column]
        #         elif(newIdx == 1):
        #             newReal[row,column] = real[row*2+1,column]
        #             newMask[row,column] = mask[row*2+1,column]
        #             if(self.return_remission):
        #                 newIntensity[row,column] = intensity[row*2+1,column]
        #         else:
        #             print("what the fuck")
        #             print(newIdx)
        # #and now go from 360 to 180
        # real = newReal[:,:newIntensity.shape[1]//2+1]
        # mask = newMask[:,:newIntensity.shape[1]//2+1]
        # if(self.return_remission):
        #     intensity = newIntensity[:,:newIntensity.shape[1]//2+1]
        # print("total mask is")
        # print(np.sum(mask))
        # print("total pixels are")
        # print(mask.shape[0] * mask.shape[1])
        # print("maxes are")
        # print(np.max(real))
        # print(np.max(intensity))
        # #This is where edit resolution section ends -------------------------------------------------------



        #Ok so new plan for training - add the ENTIRE sky to the mask
        #Fuck sky points, I'll deal with that later
        #Because otherwise due to distance difference being so much larger, they get a stupidly high loss and the network predicts sky for 99% of points
        mask = np.where(real>=maxRange, 1, mask)
        #Other works set Sky to 0, so I'll do the same for now
        real = np.where(real>=maxRange, 0, real) + 0.0001
        #Apply log. As my max is now 2057... let's treat (2047+1) as the max. That would result in 11 after the log.
        #No logging right now because it makes the math even more fucked, flat divide instead by an easy 2000
        real = ((np.log2(real+1)) / 11)
        # real = real / 2000
        # real = ((np.log2(real+1)) / 6)
        #Make negatives 0 and the rare few over 2047, be 1
        real = np.clip(real, 0, 1)
        #Random roll lets me have range image always generated pointing north, then randomly rotate the 1800 pixel width by 0...1800 pixels
        random_roll = np.random.randint(self.colMax)


        #No random roll yet - do this once I have all my images from the batch
        # if self.random_roll:
            # real = np.roll(real, random_roll, axis = 1)
            # mask = np.roll(mask, random_roll, axis = 1)
            # label = np.roll(label, random_roll, axis = 1)
        real = np.expand_dims(real, axis = 0)
        mask = np.expand_dims(mask, axis = 0)
        index = np.expand_dims(index, axis = 0)
        #Ok so sky needs to have the an extra row added because the current heuristic falls JUST shy
        #Soooo just shift it down one
        sky[1:] = sky[:-1]
        #let's do this two more times for good measure
        sky[1:] = sky[:-1]
        sky[1:] = sky[:-1]
        # sky = np.expand_dims(sky, axis = 0)

        # mask = np.zeros_like(real).astype(int)
        # mask[:, 0:640:4, :] = 1
        mask = np.logical_not(mask)
        sky = np.logical_not(sky)
        sky = np.expand_dims(sky, axis = 0)

        if self.return_remission:
            intensity = intensity / 5000
            #Any point with an intensity over 5000 is an error, ditch it
            mask = np.where(intensity>=1, 1, mask)
            intensity = np.where(intensity>=1, 0, intensity) + 0.0001
            intensity = np.clip(intensity, 0.0000001, 1.0)
            # if self.random_roll:
            #     intensity = np.roll(intensity, random_roll, axis = 1)
            intensity = np.expand_dims(intensity, axis = 0)
            real = np.concatenate((real, intensity), axis = 0)
            #let's try NOT doing this to save some memory hey
            #Nevermind it breaks shit
            mask = np.concatenate((mask,mask), axis = 0)

        #logical not the mask, as I want mask to retain only points which are visible
        #Now let's just take the first half because hahahaha I'm GPU poor
        # real = real[:,:,:225]
        # mask = mask[:,:,:225]
        # sky = np.zeros(1)
        # return real, np.logical_not(mask), sky
        return real, mask, sky, index

