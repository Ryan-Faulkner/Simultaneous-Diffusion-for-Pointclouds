from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
import h5py
from .lidar_utils import point_cloud_to_range_image
from .lidar_utils import point_cloud_to_range_image_OG
from .convertOxtsToPose import convertOxtsToPose

class KITTI360_im_8batch(Dataset):

    def __init__(self, path, config, split = 'train', resolution=None, transform=None):
        self.transform = transform
        self.return_remission = (config.data.channels == 2)
        self.random_roll = config.data.random_roll
        self.points = []
        self.origins = []
        self.modifications = np.array(config.data.modifications)
        self.batchSize = config.sampling.actualBatchSize
        self.rowMax = config.data.image_size
        self.colMax = config.data.image_width
        full_list = glob(os.path.join("/data/KITTI-360", 'data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/*.bin'))
        print(full_list[0])
        if split == "test":
            self.full_list = list(filter(lambda file: '0000_sync' in file or '0001_sync' in file, full_list))
        else:
            self.full_list = list(filter(lambda file: '0000_sync' not in file, full_list))
        self.length = len(self.full_list)

        # print("DA NEW SHAPE")
        # print(split)
        # print(len(self.full_list))
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
        self.origins = np.zeros(self.length)
        veloToCam = np.loadtxt(os.path.join("/data/KITTI-360", 'calibration/calib_cam_to_velo.txt'))
        veloToCam = np.reshape(veloToCam,[3,4])
        veloToCam = np.concatenate((veloToCam, np.array([0.,0.,0.,1.]).reshape(1,4)))
        veloToCam = np.linalg.inv(veloToCam)
        camToPose = np.loadtxt(os.path.join("/data/KITTI-360", 'calibration/calib_cam_to_pose.txt'))[0]
        camToPose = np.reshape(camToPose,[3,4])
        camToPose = np.concatenate((camToPose, np.array([0.,0.,0.,1.]).reshape(1,4)))
        # camToPose = np.linalg.inv(camToPose)
        veloToPose = np.matmul(camToPose,veloToCam)
        poseFile = os.path.join("/data/KITTI-360", 'data_poses/2013_05_28_drive_0000_sync/poses.txt')
        poses = np.loadtxt(poseFile)
        frames = poses[:,0]
        poses = np.reshape(poses[:,1:],[-1,3,4])
        self.Tr_pose_world = {}
        self.frames = frames - 1
        for frame, pose in zip(self.frames, poses): 
            #Add a nothing transform to the bottom for the intensity channel I think?
            pose = np.concatenate((pose, np.array([0.,0.,0.,1.]).reshape(1,4)))
            pose = np.matmul(pose,veloToPose)
            self.Tr_pose_world[frame] = pose
        counter = 0
        # for frameNum in range(self.length):
        #     if(poses[counter])
        #     self.origins[frameNum] =  poses[counter]
        #for h5py
        # openedFile = h5py.File(file, 'r')
        # pointArray.append(openedFile['Input'][:,:3])
        # colour=(openedFile['Input'][:,3:6])
        # intensityArray.append(0.3*colour[:,0] + 0.6*colour[:,0] + 0.11*colour[:,2])
        #for npy

        # self.length = len(self.full_list) * self.batchSize
        self.length = len(self.frames) * self.batchSize
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
        # scanOrigin = idx // self.batchSize
        # scanOrigin = scanOrigin % (len(self.points))
        #Just make it always scan 2
        # modScale = scanOrigin + 1
        tempFilename = self.full_list[0]
        AllButNum = tempFilename[:-len(tempFilename.split('/')[-1])]
        numberInBatch = idx % self.batchSize
        initialPose = idx // self.batchSize
        initialScan = int(self.frames[initialPose])
        desiredName = AllButNum + str(initialScan).zfill(10) + ".bin"


        filename = desiredName
        scanNumber = initialScan
        poseNum = initialPose
        frameNum = initialScan

        # scanOrigin = 2
        # numberInBatch = idx % self.batchSize
        # numberInRandomFullList = idx // self.batchSize
        # print("DA SHAPE")
        # print(len(self.full_list))
        # print("THe number")
        # print(scanNumber)
        # print(numberInBatch)

        # scanOrigin = self.origins[scanNumber]
        # filename = self.full_list[numberInRandomFullList]
        # scanNumber = int(filename.split('/')[-1][:-4])

        # poseNum = np.searchsorted(self.frames,scanNumber)
        # if(self.frames[poseNum] > scanNumber):
        #     poseNum = poseNum-1
        # frameNum = self.frames[poseNum]

        # AllButNum = filename[:-len(filename.split('/')[-1])]
        # filename = AllButNum + str(int(frameNum)).zfill(10) + ".bin"

        print("san check")
        print(scanNumber)
        print(filename)
        #Load the file from bin and turn into XYZ Points if necessary
        scanPoints = self.loadVelodyneData(filename)
        #Now I need to separate intensity & get x,y,z,1 values
        intensity = scanPoints[:,-1]
        pointVals = np.concatenate((np.transpose(scanPoints[:,:-1]),np.expand_dims(np.ones_like(intensity),0)),0)
        #Convert the points to global scene
        #Now the issue here is that my number may not exist as a key
        # poseNum = np.searchsorted(self.frames,scanNumber)
        # if(self.frames[poseNum] > scanNumber):
        #     poseNum = poseNum-1
        # frameNum = self.frames[poseNum]
        toWorld = self.Tr_pose_world[frameNum]
        toOGView = np.linalg.inv(toWorld)
        pointVals = np.matmul(toWorld,pointVals)
        # print("example Points")
        # print(pointVals[0])
        # print(pointVals[2])

        #Now I need the points from the perspective of the origin I want, which I can apparenlty do
        movementModifier = 5
        originsMoved = (numberInBatch + 1) * movementModifier
        # scanDesired = scanNumber + originsMoved
        poseDesired = poseNum + originsMoved
        if(poseDesired >= len(self.frames)):
            poseDesired = len(self.frames) - 1
        toWorld = self.Tr_pose_world[self.frames[poseDesired]]

        # scanDesired = int(self.frames[poseDesired])
        # desiredName = AllButNum + str(scanDesired).zfill(10) + ".bin"

        #For my GT Generation code instead of the default
        goalPointsStack = []
        # print(desiredName)
        # print(scanDesired)
        # print(poseDesired)
        for frameCount in range(int(self.frames[poseDesired+1] - self.frames[poseDesired])):
            #Don't stack all frames with this "origin" since only first one actually has that origin, rest are slightly off
            if(frameCount > 0):
                continue
            scanDesired = int(self.frames[poseDesired]+frameCount)
            desiredName = AllButNum + str(scanDesired).zfill(10) + ".bin"
            goalPointsStack.append(self.loadVelodyneData(desiredName))
        goalPoints = np.concatenate(goalPointsStack,0)
        fromWorld = np.linalg.inv(toWorld)
        pointVals = np.matmul(fromWorld,pointVals)
        print(np.max(pointVals[-1]))

        scanPoints = np.transpose(np.concatenate((pointVals[:-1],np.expand_dims(intensity,0)),0))
        #As points are shifted my code doesn't have to do anything here
        #But I do have to return the god fucking damn inverse matrix and normal matrix because I need both
        origin = np.zeros(3)
        #These points are now global. I need to now transform to the origin I want so

        maxRange = 2057.701

        #The origin I want to generate for
        # origin = 
        # print(modScale)
        self.saveNum = 0
        if self.return_remission:
            real, intensity, mask, saveNum, sky, index  = point_cloud_to_range_image(scanPoints, origin, self.return_remission, rowMax = self.rowMax, colMax = self.colMax, saveNum = self.saveNum)
        else:
            real, mask, saveNum, sky, index = point_cloud_to_range_image(scanPoints, origin, self.return_remission, rowMax = self.rowMax, colMax = self.colMax, saveNum = self.saveNum)

        #And the goal image
        # print(modScale)
        if self.return_remission:
            goalDepth, goalIntensity, _, _, goalSky, _  = point_cloud_to_range_image(goalPoints, origin, self.return_remission, rowMax = self.rowMax, colMax = self.colMax, saveNum = self.saveNum)
        else:
            goalDepth, _, _, goalSky, _ = point_cloud_to_range_image(goalPoints, origin, self.return_remission, rowMax = self.rowMax, colMax = self.colMax, saveNum = self.saveNum)
        # if self.return_remission:
        #     goalDepth, goalIntensity = point_cloud_to_range_image_OG(desiredName, False, self.return_remission)
        # else:
        #     goalDepth = point_cloud_to_range_image_OG(desiredName, False, self.return_remission)
        self.saveNum = saveNum
        #Ok so new plan for training - add the ENTIRE sky to the mask
        #Fuck sky points, I'll deal with that later
        #Because otherwise due to distance difference being so much larger, they get a stupidly high loss and the network predicts sky for 99% of points
        mask = np.where(real>=maxRange, 1, mask)
        #Other works set Sky to 0, so I'll do the same for now
        real = np.where(real>=maxRange, 0, real) + 0.0001
        goalDepth = np.where(goalDepth>=maxRange, 0, goalDepth) + 0.0001
        #Apply log. As my max is now 2057... let's treat (2047+1) as the max. That would result in 11 after the log.
        real = ((np.log2(real+1)) / 6)
        goalDepth = ((np.log2(goalDepth+1)) / 6)
        # real = real / 2000
        # real = ((np.log2(real+1)) / 6)
        #Make negatives 0 and the rare few over 2047, be 1
        real = np.clip(real, 0, 1)
        goalDepth = np.clip(goalDepth, 0, 1)
        #Random roll lets me have range image always generated pointing north, then randomly rotate the 1800 pixel width by 0...1800 pixels
        random_roll = np.random.randint(self.colMax)



        if self.random_roll:
            real = np.roll(real, random_roll, axis = 1)
            mask = np.roll(mask, random_roll, axis = 1)
            sky = np.roll(sky, random_roll, axis = 1)
            # label = np.roll(label, random_roll, axis = 1)
        real = np.expand_dims(real, axis = 0)
        mask = np.expand_dims(mask, axis = 0)
        goalDepth = np.expand_dims(goalDepth, axis = 0)
        goalIntensity = np.expand_dims(goalIntensity, axis = 0)
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
            # intensity = intensity / 2000
            #Any point with an intensity over 5000 is an error, ditch it
            mask = np.where(intensity>=1, 1, mask)
            intensity = np.where(intensity>=1, 0, intensity) + 0.0001
            intensity = np.clip(intensity, 0, 1.0)
            goalIntensity = np.where(goalIntensity>=1, 0, goalIntensity) + 0.0001
            goalIntensity = np.clip(goalIntensity, 0, 1.0)
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
            goalDepth = np.concatenate((goalDepth, goalIntensity), axis = 0)
            mask = np.concatenate((mask,mask), axis = 0)

        #logical not the mask, as I want mask to retain only points which are visible
        #Now let's just take the first half because hahahaha I'm GPU poor
        # real = real[:,:,:225]
        # mask = mask[:,:,:225]
        # sky = np.zeros(1)
        # returnedScale = np.expand_dims(modScale,axis=0) 
        toWorld = np.expand_dims(toWorld, axis=0)
        fromWorld = np.expand_dims(fromWorld, axis = 0)
        print("mask ratio")
        print(mask.shape)
        print(np.sum(mask[0]) / (np.shape(mask)[1]*np.shape(mask)[2]))
        return real, np.logical_not(mask), np.logical_not(sky), index, toWorld, fromWorld, goalDepth, toOGView, initialScan
        # return real, mask, sky


    #From SemanticKitti360 github
    def loadVelodyneData(self, pcdFile):
        # pcdFile = os.path.join(self.raw3DPcdPath, '%010d.bin' % frame)
        if not os.path.isfile(pcdFile):
            raise RuntimeError('%s does not exist!' % pcdFile)
        pcd = np.fromfile(pcdFile, dtype=np.float32)
        pcd = np.reshape(pcd,[-1,4])
        return pcd 
