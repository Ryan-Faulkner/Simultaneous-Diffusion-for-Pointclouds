import numpy as np
from glob import glob
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.dsm import anneal_dsm_score_estimation_with_mask
from losses.dsm import anneal_dsm_score_estimation_simultaneous

import torch.nn.functional as F
import logging
import torch
import os
import math
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest, NCSN_LiDAR, NCSN_LiDAR_small
from models.ncsn import NCSN, NCSNdeeper
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_densification)
from models import get_sigmas
from models.ema import EMAHelper
#from .nvs import KITTINVS, novel_view_synthesis

__all__ = ['NCSNRunnerSimultaneous']


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == 'KITTI' or config.data.dataset == 'lidar':
        #return NCSN_LiDAR(config).to(config.device)
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'KITTI360':
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'HDVMineGenerate':
        #return NCSN_LiDAR(config).to(config.device)
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'HDVMinePreGenerated':
        #return NCSN_LiDAR(config).to(config.device)
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'HDVMineGenerateFromInvidivualScans':
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'HDVMinePreGenerated8Batch':
        return NCSN_LiDAR_small(config).to(config.device)

class MySampler():
    def __init__(self, num_batches, batch_size):
        self.n_batches = num_batches
        self.batch_size = batch_size

    def __iter__(self):
        # print("the fucking number")
        # print(self.n_batches)
        numbers = np.arange(self.n_batches)
        np.random.shuffle(numbers)
        # print(numbers)
        batches = []
        for chosenNum in range(self.n_batches):
            for i in range(self.batch_size):
                batches.append((numbers[chosenNum] * self.batch_size) + i)
            # batches.append(batch)
        return iter(batches)

class NCSNRunnerSimultaneous():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        # trainingSize = len(glob('PreGenFinal/Depth/*')) * 6 // 10
        # valSize = len(glob('PreGenFinalVal/Depth/*')) * 2 // 10
        trainingSize = len(glob('/data/PreGenFinal/PreGenFinal/Depth/*')) * 6 // 10
        valSize = len(glob('/data/PreGenFinalVal/Depth/*')) * 2 // 10

        trainingSampler = MySampler(num_batches = trainingSize ,batch_size=self.config.training.batch_size)
        valSampler = MySampler(num_batches = valSize ,batch_size=self.config.training.batch_size)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, sampler=trainingSampler,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, sampler=valSampler,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size * self.config.data.image_width * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        # print(score)

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0
        trueStep = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            # states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            # states = torch.load('diffusionNet/checkpoint_148.pth')
            states = torch.load('/data/firstTrainingSession/logs/HDVMine/checkpoint_148.pth')
            # print(len(states[0]))
            # print(len(score.state_dict()))
            statesToLoad = {}
            for key in states[0]:
                # print(states[0][key].shape)
                # print(score.state_dict()[key].shape)
                if(states[0][key].size== score.state_dict()[key].shape):
                    statesToLoad[key] = states[0][key]
            #strict=False lets me ignore that some layers are missing and remain randomised
            score.load_state_dict(statesToLoad, strict=False)
            # ### Make sure we can resume with different eps
            # states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            # optimizer.load_state_dict(states[1])
            # start_epoch = states[2]
            # step = states[3]
            # if self.config.model.ema:
            #     ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        maxTimeStepReachable = 1

        rowMax = self.config.data.image_size
        colMax = self.config.data.image_width
        rowCount = rowMax
        colCount = colMax
        maxRange = 2057.701 #Current record set by Penrice
        minPixels = rowMax*colMax/6
        horizontalScope = 360
        verticalScope = 60 #+5 down to -40
        horizontalAngles = math.radians(horizontalScope) / colMax #6 minutes of an arc for a 800x3600 image, 12 minutes for 800x1800 (assumign 360 coverage)
        verticalAngles = math.radians(verticalScope) / rowMax #6 arc minutes (one tenth of a degree). This means every ten pixels is one degree, 80 pixels = +- 40 degrees
        horizontalMin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
        verticalMin = rowCount*3//(-4) * verticalAngles + (verticalAngles/2)
        #divide origins by 2000 so in same space as my distances
        modificationList = np.array([[0,0,0],
                    [10,0,0],
                    [0,10,0],
                    [10,10,0],
                    [0,0,10],
                    [-10,0,0],
                    [0,-10,0],
                    [-10,-10,0]]) / 2000
        modificationList = np.expand_dims(np.expand_dims(modificationList,-1),-1)
        originListOG = torch.from_numpy(modificationList).to(self.config.device)

        azimuth = torch.from_numpy((np.arange(colMax) * horizontalAngles) + horizontalMin).to(self.config.device)
        elevation = torch.from_numpy((np.arange(rowMax) * verticalAngles) + verticalMin).to(self.config.device)
        #Now I have the horizontal and vertical angle for each pixel in the image
        #Everything else should be done inside the loop as these are the only constants
        for epoch in range(start_epoch, self.config.training.n_epochs):
            #X is point cloud, mask is mask
            for i, (X, mask, sky) in enumerate(dataloader):
                originList = originListOG * sigmas[0]
                # print(X.shape)
                # print(mask.shape)
                #Do the rolling now. All images rolled by the same amount, equivalent to rotating the horizontal plane by that amount.
                #This works because all images have parallel horizontal planes

                random_roll = np.random.randint(self.config.data.image_width)
                #BxCxWxH
                X = torch.roll(X, random_roll, dims = -1).detach()
                mask = torch.roll(mask, random_roll, dims = -1).detach()

                #To do:
                #Rewrite the inbetween step to project to 3D then 2D
                #Rewrite the Unet to pool global scene feature across the entire batch

                step += 1
                X = X.to(self.config.device)
                mask = mask.to(self.config.device)
                sky = sky.to(self.config.device)
                #This does nothing as all are false - I am commenting out as simultaneous diffusion designed assuming no transformations
                # X = data_transform(self.config, X)
                #Ok so for the noise adding:
                #Get every point for each image
                #X[:,0,:,:] is Depth, azimuth and elevation are given from before
                #NEED TO ADD THE ORIGINS HERE AS NOT ALL FROM 0,0,0
                with torch.no_grad():
                    updatedImages = X.clone()
                    #project the images to 3D
                    pointX = torch.add(X[:,0] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,0])
                    pointY = torch.add(X[:,0] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,1])
                    pointZ = torch.add(X[:,0] * torch.sin(elevation).view(1,-1,1),originList[:batchSize,2])
                    #Now generate the noise, once for each viewpoint
                    labels = torch.full((X.shape[0],),0, device=X.device)
                    used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                    twoDimensionalNoise = torch.randn_like(X) * used_sigmas
                    #Now project the noise to 3D
                    NoiseX = torch.add(twoDimensionalNoise[:,0] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,0])
                    NoiseY = torch.add(twoDimensionalNoise[:,0] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,1])
                    NoiseZ = torch.add(twoDimensionalNoise[:,0] * torch.sin(elevation).view(1,-1,1),originList[:batchSize,2])

                    priorCloud = torch.stack((pointX,pointY,pointZ,X[:,1,:,:],mask[:,0]))
                    noiseCloud = torch.stack((noiseX,noiseY,noiseZ,twoDimensionalNoise[:,1,:,:]))

                    #Ok so the new plan is.... give up on BxRxC completely.
                    priorCloud = priorCloud.view(5,-1)
                    #Apply gaussian noise to every point in XYZ as well as Intensity
                    # labels = torch.full((priorCloud.shape[1],),0, device=X.device)
                    # used_sigmas = sigmas[labels].view(priorCloud.shape[0], *([1] * X.shape[0]))

                    
                    labels = torch.full((X.shape[0],),0, device=X.device)
                    used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                    # print("sigmas shape")
                    # print(used_sigmas.shape)
                    #Add max noise to any image sections I don't trust to effectively intialise them
                    # noise = torch.randn_like(priorCloud[:4]) * used_sigmas[0,0,0,0]
                    #The oriignal XYZIM cloud.
                    originalCloud = torch.clone(priorCloud)
                    priorCloud = torch.unsqueeze(torch.add(priorCloud[:4], noiseCloud),0)

                #This is the basic code for this section done.
                #Tomorrow: Before I continue this, set up the inference-only version, AS A SEPARATE RUNNER. Should be a runner for normally-trained networks using simultaneous inference.
                #Goal is to have those results for Thursday, then have this code as close to done as possible / training
                #To do to get this train() code ready:
                #The code each loop which does this for that loop's sigma
                #Work out how to differentiate points which are "known" and ones which are not.
                #Maybe I should use newImages[:,2] to get the ORIGINAL masks so that every loop I can ditch those points?
                #No that means nothing
                #What matters is knowing which points to apply loss to
                #I think I need to:
                #0) make it so the XYZI array (without any noise) is saved forevermore
                #1) calculate all the above for the original point cloud's existing pixels EXCEPT HOLE FILLING STEP
                #2) calculate all the above for the "prior" point cloud 
                #3) Fill images using points from #2, THEN using #1 overriding as necessary, then filling holes as above using prior image which has no holes
                #4) Also the The U-net pooling needs to be coded

                #This is my t-1 (input to network for training)
                #for each spherical image, get point distance, Intensity, spherical gradient (calculated from X noise, Y noise, and Z noise)
                #Mask any holes - we don't care about them for loss.
                #fill holes created based on what was there for previous image (t), adding the change in magnitude applied to the point that used to be there. Need for network calculations.
                #for each network's predicted gradient:
                #calculate magnitude, angle direction and add to the original point's total velocity, for each spherical image it's part of
                #also add this to the "total unit velocity". Something along the lines of if every velocity was a unit vector, what would it's X,Y,Z proportions be?
                #divide by total unit velocity
                #run loss comparing them
                #use the total unit velocity to also weight the three losses (X,Y,Z) for each point

                for timestep in range(maxTimeStepReachable):
                    with torch.no_grad():
                        #when back from water, add the reverse diffusion (priorcloud) to the image generation, no need to add noise as it will have predicted gradients added at end of each timestep loop
                        #remember that forwardDiffusion taxes priority when generating images
                        #also remember I should be hole filling from the priorcloud NOT the original
                        #Ok so for each timestep I have:
                        #Original point cloud originalCloud XYZIM (5xBxRxC)
                        #If I add noise, I get noisedCloud
                        #I also have the point cloud from the prior timestep, priorCloud
                        priorCloud = priorCloud.detach()
                        labels = torch.full((updatedImages.shape[0],),timestep, device=updatedImages.device)
                        used_sigmas = sigmas[labels].view(updatedImages.shape[0], *([1] * len(updatedImages.shape[1:])))
                        #Add max noise to any image sections I don't trust to effectively intialise them
                        noise = torch.randn_like(originalCloud[:4]) * used_sigmas[0,0,0,0]
                        # print("noise size")
                        # print(noise.shape)
                        # print(originalCloud.shape)
                        # print(noise.shape)
                        forwardDiffusion = torch.unsqueeze(torch.add(originalCloud[:4],noise),0)
                        # print(forwardDiffusion.shape)
                        # print("forward")
                        # print(forwardDiffusion.shape)
                        # pointX = pointX + noise*torch.logical_not(mask).int() 
                        # noise = torch.randn_like(pointX) * used_sigmas
                        # pointY = pointY + noise*torch.logical_not(mask).int() 
                        # noise = torch.randn_like(pointX) * used_sigmas
                        # pointZ = pointZ + noise*torch.logical_not(mask).int()  
                        # noise = torch.randn_like(pointX) * used_sigmas
                        # newIntensity = X[:,1,:,:] + noise*torch.logical_not(mask).int()

                        #Generate 2D images from each origin
                        #new cloud should be 1x4xBxRxC
                        bigCloud = torch.tile(forwardDiffusion[:,:3],(self.config.training.batch_size,1,1))
                        # print(bigCloud.shape)
                        relativePoints = torch.subtract(bigCloud,originList[:,:,0])
                        # print("relpoints")
                        # print(relativePoints.shape)
                        xy = torch.square(relativePoints[:,0]) + torch.square(relativePoints[:,1])
                        newDepth = torch.sqrt(xy + torch.square(relativePoints[:,2]))
                        horizontal = torch.atan2(relativePoints[:,1], relativePoints[:,0])
                        # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
                        xy = torch.sqrt(xy)
                        vertical = torch.atan2(relativePoints[:,2], xy)
                        newCol = torch.round(torch.divide((horizontal-horizontalMin),horizontalAngles)).int()
                        newRow = torch.round(torch.divide((vertical-verticalMin),verticalAngles)).int()
                        #This is the forward diffusion - I do not wnat it to include any points which come from unseen pixels so use mask to remove them along with any outside the grid
                        inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.logical_and(torch.less(newRow,rowCount),originalCloud[4])))
                        #ok so now I have:
                        #newCol & newRow, in form OxBxRxC for the original RxC values of each origin O.
                        #I also have Intensity+Noise in 1xBxRxC
                        #also depth is newDepth, in OxBxRxC
                        del bigCloud
                        # print("the prior shapes")
                        # print(priorCloud.shape)
                        bigCloudPrior = torch.tile(priorCloud[:,:3],(self.config.training.batch_size,1,1))
                        # print(bigCloudPrior.shape)
                        relativePoints = torch.subtract(bigCloudPrior,originList[:,:,0])
                        xy = torch.square(relativePoints[:,0]) + torch.square(relativePoints[:,1])
                        newDepthPrior = torch.sqrt(xy + torch.square(relativePoints[:,2]))
                        horizontal = torch.atan2(relativePoints[:,1], relativePoints[:,0])
                        # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
                        xy = torch.sqrt(xy)
                        vertical = torch.atan2(relativePoints[:,2], xy)
                        newColPrior = torch.round(torch.divide((horizontal-horizontalMin),horizontalAngles)).int()
                        newRowPrior = torch.round(torch.divide((vertical-verticalMin),verticalAngles)).int()
                        inGridPrior = torch.logical_and(torch.logical_and(torch.greater(newColPrior,0),torch.less(newColPrior,colCount)),torch.logical_and(torch.greater(newRowPrior,0),torch.less(newRowPrior,rowCount)))
                        #ok now I need to make my new images
                        #So I need to have a IxRxC, which takes the closest depth for each B, for the row and column in new image I.
                        #But to be clear, I have OxBxRxC, which is the distance from each origin O, to the point in each sample B, at pixel R,C.
                        # Keeping B R C is good for future but a huge fucking pain for the present
                        # fuck it goodbye B R C
                        # newMagnitude, newIndices = torch.min(newDepth, dim=1)
                        newStack = []
                        finalIndStack = []
                        imageMaskStack = []
                        del bigCloudPrior

                        for origin in range(self.config.training.batch_size):
                            new_ind = torch.argsort(torch.flatten(newDepth[origin][inGrid[origin]]),dim=0)
                            newRowTemp =torch.flatten(newRow[origin][inGrid[origin]])[new_ind]
                            newColTemp =torch.flatten(newCol[origin][inGrid[origin]])[new_ind]
                            merged = torch.stack((newRowTemp,newColTemp),0)
                            # merged is now a 2xOxBxRxC array
                            # this does not work as it can only return a 1D array, rip. Ahh right
                            # print(merged.shape)
                            merged,reduced_ind = torch.unique(merged, return_inverse=True, dim=1)
                            # print("break it down")
                            final_ind = torch.arange(len(newDepth[1]),device=self.config.device)
                            gridInd = inGrid[origin]
                            # print(newDepth.shape)
                            # print(gridInd.shape)
                            # print(final_ind.shape)
                            final_ind = final_ind[gridInd]
                            final_ind = final_ind[new_ind]
                            reduced_ind = torch.unique(reduced_ind)
                            final_ind = final_ind[reduced_ind]
                            # print(final_ind.shape)
                            # print(merged.shape)
                            # print(reduced_ind.shape)

                            #Ok so if I want to create a new PriorCloud:
                            #Flatten the first one so it's always unstructured, no BxCxR
                            #collect all the final_ind for each origin as well as mask, intensity, etc
                            #3Dgradients = X,Y,Z projection of the gradient instead of the new magnitude 
                            #finalGradsForward = torch.sparse_coo_tensor(3Dgradients[mask].view(3,-1), final_ind[mask].view(-1),size=3,len(torch.max(final_indPrior+final_ind))).to_dense() #sets gradient to 0 for points not used
                            #finalGradsReverse = torch.sparse_coo_tensor(3Dgradients[not mask].view(3,-1), final_ind[mask].view(-1),size=3,len(torch.max(final_indPrior+final_ind))).to_dense() #sets gradient to 0 for points not used
                            #newCloud = torch.where(mask,forwardCloud + forwardgrads, priorCloud + ReverseGrads)
                            #I also need to then add the individual gradietns to each image and pass it to the next loop purely for hole-filling backup

                            # final_ind = new_ind[reduced_ind]
                            # newCol = torch.view(newCol[reduced_ind],(config.data.batch_size,config.data.batch_size,rowCount,colCount))
                            # newRow = torch.view(newRow[reduced_ind],(config.data.batch_size,config.data.batch_size,rowCount,colCount))

                            imageDepth = torch.sparse_coo_tensor(merged,torch.flatten(newDepth[origin])[final_ind], size=(rowMax,colMax)).to_dense()
                            #Don't need this as not stupid enough to run a heuristic at timestep of 0
                            # imageXY = coo_matrix((xy[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), size=(rowMax,colMax)).toarray()[tempDepth != 0]
                            # print(forwardDiffusion.shape)
                            imageIntensity = torch.sparse_coo_tensor(merged,torch.flatten(forwardDiffusion[0,3,:])[final_ind], size=(rowMax,colMax)).to_dense()
                            #Only use points taken from forward diffusion for loss
                            imageMask = torch.sparse_coo_tensor(merged,torch.ones_like(merged[0], device=self.config.device), size=(rowMax,colMax)).to_dense()
                            # print(torch.flatten(X[:,0]).shape)
                            # print(torch.flatten(X[:,0])[final_ind].shape)
                            oldDepth = torch.sparse_coo_tensor(merged,torch.flatten(X[:,0])[final_ind], size=(rowMax,colMax)).to_dense()
                            oldIntensity = torch.sparse_coo_tensor(merged,torch.flatten(X[:,1])[final_ind], size=(rowMax,colMax)).to_dense()
                            final_ind_image = torch.sparse_coo_tensor(merged,final_ind, size=(rowMax,colMax)).to_dense()
                            # print(oldDepth.shape)
                            magnitudeChange = imageDepth - oldDepth
                            intensityChange = imageIntensity - oldIntensity

                            
                            new_ind = torch.argsort(torch.flatten(newDepthPrior[origin][inGridPrior[origin]]),dim=0)
                            newRowTemp =torch.flatten(newRowPrior[origin][inGridPrior[origin]])[new_ind]
                            newColTemp =torch.flatten(newColPrior[origin][inGridPrior[origin]])[new_ind]
                            merged = torch.stack((newRowTemp,newColTemp),0)
                            merged,reduced_indPrior = torch.unique(merged, return_inverse=True, dim=1)
                            reduced_indPrior = torch.unique(reduced_indPrior)
                            final_indPrior = torch.arange(len(newDepthPrior[1]),device=self.config.device)
                            gridIndPrior = inGridPrior[origin]
                            final_indPrior = final_indPrior[gridIndPrior]
                            final_indPrior = final_indPrior[new_ind]
                            final_indPrior = final_indPrior[reduced_indPrior]
                            # print(final_ind.shape)
                            # print(newDepthPrior.shape)

                            imageDepthPrior = torch.sparse_coo_tensor(merged,torch.flatten(newDepthPrior[origin])[final_indPrior], size=(rowMax,colMax)).to_dense()
                            #Don't need this as not stupid enough to run a heuristic at timestep of 0
                            # imageXY = coo_matrix((xy[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), size=(rowMax,colMax)).toarray()[tempDepth != 0]
                            # print(priorCloud.shape)
                            imageIntensityPrior = torch.sparse_coo_tensor(merged,torch.flatten(priorCloud[0,3])[final_indPrior], size=(rowMax,colMax)).to_dense()
                            final_ind_imagePrior = torch.sparse_coo_tensor(merged,final_indPrior, size=(rowMax,colMax)).to_dense()
                            
                            #fill with prior point cloud
                            imageIntensity = torch.where(imageDepth != 0, imageIntensity, imageIntensityPrior)
                            final_ind_image = torch.where(imageDepth != 0, final_ind_image, final_ind_imagePrior)
                            imageDepth = torch.where(imageDepth != 0, imageDepth, imageDepthPrior)
                            #Magnitude doesn't matter as mask is 0 for all these points, so only need to fill for intensity & depth
                            #Ok in theory I now have a new image
                            #Fill any holes with the Depth and Intensity that their point was changed to last time so there is something for network to actually use
                            #To be clear, this step is taking the magnitude & intensity of the point which USED to be in that Row & Column, 
                            #but was shifted to a different Row & Column after the prior timestep's gradients were applied
                            #This is my solution to ensure that the reverse diffusion does not have gaps appear as points move around
                            imageIntensity = torch.where(imageDepth != 0, imageIntensity, updatedImages[origin,1])
                            final_ind_image = torch.where(imageDepth != 0, final_ind_image, torch.tensor(-1,device=self.config.device))
                            imageDepth = torch.where(imageDepth != 0, imageDepth, updatedImages[origin,0])
                            # print("all the shapes")
                            # print(imageDepth.shape)
                            # print(imageIntensity.shape)
                            # print(imageMask.shape)
                            # print(magnitudeChange.shape)
                            # print(intensityChange.shape)
                            # print(final_ind_image.shape)

                            newStack.append(torch.stack((imageDepth,imageIntensity, magnitudeChange,intensityChange)))
                            finalIndStack.append(final_ind_image.long())
                            imageMaskStack.append(imageMask.bool())
                        newImages = torch.stack(newStack)
                        finalImages = torch.stack(finalIndStack)
                        maskImages = torch.stack(imageMaskStack)




                    trueStep += 1
                    # print("The final shape")
                    # print(newImages.shape)
                    X = newImages
                    X.requires_grad_()
                    # labels.requires_grad_()
                    used_sigmas.requires_grad_()
                    score.train()
                    # labels = torch.full((X.shape[0],),timestep, device=X.device)
                    #No need to data_tranform the mask, as that shouldn't have noise added, etc

                    loss, grad = anneal_dsm_score_estimation_simultaneous(score, X[:,:2], used_sigmas, X[:,2:4], maskImages, labels,
                                                       self.config.training.anneal_power,
                                                       hook)

                    
                    tb_logger.add_scalar('loss', loss, global_step=trueStep)
                    tb_hook()

                    logging.info("step: {}, timestep: {}, loss: {}".format(step, timestep, loss.item()))

                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(score.parameters(), 10.0, 'inf')
                    optimizer.step()

                    #now that loss is calculated, detach everything and set up images and point cloud to pass to next timestep
                    updatedImages = X[:,:2].detach().clone()
                    grad = grad.detach()
                    # print(maskImages)
                    with torch.no_grad():

                        # print(maskImages)
                        noiseX = (grad[:,0,:,:] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1))
                        noiseY = (grad[:,0,:,:] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1))
                        noiseZ = (grad[:,0,:,:] * torch.sin(elevation).view(1,-1,1))
                        noiseI = (grad[:,1,:,:])
                        finalImages = finalImages.flatten()
                        generateNewMask = finalImages != -1
                        noiseX = noiseX.flatten()[generateNewMask]
                        noiseY = noiseY.flatten()[generateNewMask]
                        noiseZ = noiseZ.flatten()[generateNewMask]
                        noiseI = noiseI.flatten()[generateNewMask]
                        # print(maskImages)
                        maskImages = maskImages.flatten()
                        # maskImages = maskImages.view(1,1,-1)
                        # maskImages = torch.tile(maskImages,(1,4,1))
                        # print(maskImages)
                        maskImages = maskImages[generateNewMask]
                        finalImages = finalImages[generateNewMask]
                        indicesForward = finalImages[maskImages]
                        indicesReverse = finalImages[torch.logical_not(maskImages)]
                        #If I pass them both through torch unique and convert to the indices, it will naturally make them both values 0->len(array)
                        indicesForward, newForward = torch.unique(indicesForward, return_inverse=True)
                        indicesReverse, newReverse = torch.unique(indicesReverse, return_inverse=True)  
                        #Ok so for finalImages
                        #All -1 values mean I should generate a new point using that updatedImage
                        # maxSize = self.config.training.batch_size * self.config.training.image_width * self.config.training.image_size
                        # print("fuck you")
                        # print(indicesReverse.shape)
                        # print(newReverse.shape)
                        # print(torch.max(newReverse))
                        # print(noiseX[torch.logical_not(maskImages)].shape)
                        # print("extra fuck you")
                        # print(torch.count_nonzero(noiseX[torch.logical_not(maskImages)]))
                        noiseXReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseX[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()
                        noiseYReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseY[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()
                        noiseZReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseZ[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()
                        noiseIReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseI[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()

                        noiseXForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseX[maskImages], size=([indicesForward.shape[0]])).to_dense()
                        noiseYForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseY[maskImages], size=([indicesForward.shape[0]])).to_dense()
                        noiseZForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseZ[maskImages], size=([indicesForward.shape[0]])).to_dense()
                        noiseIForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseI[maskImages], size=([indicesForward.shape[0]])).to_dense()
                        # print("sanity check")
                        # print(noiseXReverse.shape)


                        #I now have noise for all the original points from both point clouds, and ONLY the points actually relevant.
                        #Now I need to create the combo clouds to apply this noise to
                        #so I need only the values which exist within indicesForward
                        #ALl other to be ignored
                        #soooooo just filter it?
                        #yeah no duplicates anymore because I ran it under unique

                        #make it unique because we only want the first from each one
                        finalImages = torch.unique(finalImages)
                        #when mask is true, take the XYZ values from forwardDiffusion at the index indicated by finalImages
                        #I need to do a coo array using finalImages[mask] as the indices and forwardDiffusion as values
                        #then add more values to that array using finalimages[not mask] and priorCloud
                        #finally, for all pixels which correspond to a -1, I need to generate a new point, using updatedImages as the base
                        #
                        #In theory I could do this at the end and add them after all the noise is finished
                        #I need to redo this to ACTUALLY be able to add multiple noiseX values to the same point

                        #In contrast, 
                        # print(maskImages)
                        # print(maskImages.shape)
                        comboX = torch.cat((priorCloud[0,0][indicesReverse],forwardDiffusion[0,0][indicesForward]),-1)
                        comboY = torch.cat((priorCloud[0,1][indicesReverse],forwardDiffusion[0,1][indicesForward]),-1)
                        comboZ = torch.cat((priorCloud[0,2][indicesReverse],forwardDiffusion[0,2][indicesForward]),-1)
                        comboI = torch.cat((priorCloud[0,3][indicesReverse],forwardDiffusion[0,3][indicesForward]),-1)
                        comboCloud = torch.stack((comboX,comboY,comboZ,comboI))
                        # print(comboCloud.shape)
                        # print("why crash")
                        noiseX = torch.cat((noiseXReverse,noiseXForward),-1)
                        noiseY = torch.cat((noiseYReverse,noiseYForward),-1)
                        noiseZ = torch.cat((noiseZReverse,noiseZForward),-1)
                        noiseI = torch.cat((noiseIReverse,noiseIForward),-1)
                        noiseCloud = torch.stack((noiseX,noiseY,noiseZ,noiseI))
                        for s in range(self.config.sampling.n_steps_each):
                        
                            step_size = self.config.sampling.step_lr * (sigmas[timestep] / sigmas[-1]) ** 2    
                            # grad_likelihood = -mask * (X - originalX) # - 0.05*(1-mask)*(x_mod - raw_interp)

                            noise2 = torch.randn_like(updatedImages)
                            # grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                            # grad_likelihood_norm = torch.norm(grad_likelihood.view(grad.shape[0], -1), dim=-1).mean()
                            # noise_norm = torch.norm(noise2.view(noise2.shape[0], -1), dim=-1).mean()
                            updatedImages = updatedImages + step_size * grad + noise2 * torch.sqrt(step_size * 2)
                            comboCloud = comboCloud + (noiseCloud*step_size)
                        #now I need to add new points from updatedImages for anywhere that generateNewMask == 0
                        pointX = torch.add(updatedImages[:,0,:,:] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:,0]).flatten()
                        pointY = torch.add(updatedImages[:,0,:,:] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:,1]).flatten()
                        pointZ = torch.add(updatedImages[:,0,:,:] * torch.sin(elevation).view(1,-1,1),originList[:,2]).flatten()
                        toAdd = torch.stack((pointX[torch.logical_not(generateNewMask)],pointY[torch.logical_not(generateNewMask)],pointZ[torch.logical_not(generateNewMask)],updatedImages[:,1,:,:].flatten()[torch.logical_not(generateNewMask)]))
                        comboCloud = torch.cat((comboCloud,toAdd),-1)
                        # print("true check")
                        # print(comboCloud.shape)
                        priorCloud = torch.unsqueeze(comboCloud,0).detach() 
                        # print(priorCloud.shape)
                        # print(originalCloud.shape)

                    if self.config.model.ema:
                        ema_helper.update(score)

                    if step >= self.config.training.n_iters:
                        return 0

                    if step % 10 == 0 and timestep == maxTimeStepReachable-1: #100
                    # if (step % 100 == 0 or maxTimeStepReachable > 100) and timestep == maxTimeStepReachable-1: #100
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()
                        try:
                            test_X, test_y, test_sky = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            test_X, test_y, test_sky = next(test_iter)

                        X = test_X.to(self.config.device)
                        mask = test_y.to(self.config.device)
                        sky = test_sky.to(self.config.device)
                        lossTotal = 0
                        with torch.no_grad():

                            updatedImages = X.clone()
                            pointX = torch.add(X[:,0,:,:] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:,0])
                            pointY = torch.add(X[:,0,:,:] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:,1])
                            pointZ = torch.add(X[:,0,:,:] * torch.sin(elevation).view(1,-1,1),originList[:,2])
                            priorCloud = torch.stack((pointX,pointY,pointZ,X[:,1,:,:],mask[:,0]))
                            #Ok so the new plan is.... give up on BxRxC completely.
                            priorCloud = priorCloud.view(5,-1)
                            #Apply gaussian noise to every point in XYZ as well as Intensity
                            # labels = torch.full((priorCloud.shape[1],),0, device=X.device)
                            # used_sigmas = sigmas[labels].view(priorCloud.shape[0], *([1] * X.shape[0]))

                            
                            labels = torch.full((X.shape[0],),0, device=X.device)
                            used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                            # print("sigmas shape")
                            # print(used_sigmas.shape)
                            #Add max noise to any image sections I don't trust to effectively intialise them
                            noise = torch.randn_like(priorCloud[:4]) * used_sigmas[0,0,0,0]
                            #The oriignal XYZIM cloud.
                            originalCloud = torch.clone(priorCloud)
                            priorCloud = torch.unsqueeze(priorCloud[:4] + noise,0)

                            #This is the basic code for this section done.
                            #Tomorrow: Before I continue this, set up the inference-only version, AS A SEPARATE RUNNER. Should be a runner for normally-trained networks using simultaneous inference.
                            #Goal is to have those results for Thursday, then have this code as close to done as possible / training
                            #To do to get this train() code ready:
                            #The code each loop which does this for that loop's sigma
                            #Work out how to differentiate points which are "known" and ones which are not.
                            #Maybe I should use newImages[:,2] to get the ORIGINAL masks so that every loop I can ditch those points?
                            #No that means nothing
                            #What matters is knowing which points to apply loss to
                            #I think I need to:
                            #0) make it so the XYZI array (without any noise) is saved forevermore
                            #1) calculate all the above for the original point cloud's existing pixels EXCEPT HOLE FILLING STEP
                            #2) calculate all the above for the "prior" point cloud 
                            #3) Fill images using points from #2, THEN using #1 overriding as necessary, then filling holes as above using prior image which has no holes
                            #4) Also the The U-net pooling needs to be coded

                            #This is my t-1 (input to network for training)
                            #for each spherical image, get point distance, Intensity, spherical gradient (calculated from X noise, Y noise, and Z noise)
                            #Mask any holes - we don't care about them for loss.
                            #fill holes created based on what was there for previous image (t), adding the change in magnitude applied to the point that used to be there. Need for network calculations.
                            #for each network's predicted gradient:
                            #calculate magnitude, angle direction and add to the original point's total velocity, for each spherical image it's part of
                            #also add this to the "total unit velocity". Something along the lines of if every velocity was a unit vector, what would it's X,Y,Z proportions be?
                            #divide by total unit velocity
                            #run loss comparing them
                            #use the total unit velocity to also weight the three losses (X,Y,Z) for each point

                            for testTimestep in range(maxTimeStepReachable):
                                #when back from water, add the reverse diffusion (priorcloud) to the image generation, no need to add noise as it will have predicted gradients added at end of each timestep loop
                                #remember that forwardDiffusion taxes priority when generating images
                                #also remember I should be hole filling from the priorcloud NOT the original
                                #Ok so for each timestep I have:
                                #Original point cloud originalCloud XYZIM (5xBxRxC)
                                #If I add noise, I get noisedCloud
                                #I also have the point cloud from the prior timestep, priorCloud
                                priorCloud = priorCloud.detach()
                                labels = torch.full((updatedImages.shape[0],),testTimestep, device=updatedImages.device)
                                used_sigmas = sigmas[labels].view(updatedImages.shape[0], *([1] * len(updatedImages.shape[1:])))
                                #Add max noise to any image sections I don't trust to effectively intialise them
                                noise = torch.randn_like(originalCloud[:4]) * used_sigmas[0,0,0,0]
                                # print("noise size")
                                # print(noise.shape)
                                # print(originalCloud.shape)
                                # print(noise.shape)
                                forwardDiffusion = torch.unsqueeze(torch.add(originalCloud[:4],noise),0)
                                # print("forward")
                                # print(forwardDiffusion.shape)
                                # pointX = pointX + noise*torch.logical_not(mask).int() 
                                # noise = torch.randn_like(pointX) * used_sigmas
                                # pointY = pointY + noise*torch.logical_not(mask).int() 
                                # noise = torch.randn_like(pointX) * used_sigmas
                                # pointZ = pointZ + noise*torch.logical_not(mask).int()  
                                # noise = torch.randn_like(pointX) * used_sigmas
                                # newIntensity = X[:,1,:,:] + noise*torch.logical_not(mask).int()

                                #Generate 2D images from each origin
                                #new cloud should be 1x4xBxRxC
                                bigCloud = torch.tile(forwardDiffusion[:,:3],(self.config.training.batch_size,1,1))
                                relativePoints = torch.subtract(bigCloud,originList[:,:,0])
                                # print("relpoints")
                                # print(relativePoints.shape)
                                xy = torch.square(relativePoints[:,0]) + torch.square(relativePoints[:,1])
                                newDepth = torch.sqrt(xy + torch.square(relativePoints[:,2]))
                                horizontal = torch.atan2(relativePoints[:,1], relativePoints[:,0])
                                # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
                                xy = torch.sqrt(xy)
                                vertical = torch.atan2(relativePoints[:,2], xy)
                                newCol = torch.round(torch.divide((horizontal-horizontalMin),horizontalAngles)).int()
                                newRow = torch.round(torch.divide((vertical-verticalMin),verticalAngles)).int()
                                #This is the forward diffusion - I do not wnat it to include any points which come from unseen pixels so use mask to remove them along with any outside the grid
                                inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.logical_and(torch.less(newRow,rowCount),originalCloud[4])))
                                #ok so now I have:
                                #newCol & newRow, in form OxBxRxC for the original RxC values of each origin O.
                                #I also have Intensity+Noise in 1xBxRxC
                                #also depth is newDepth, in OxBxRxC
                                del bigCloud
                                # print("the prior shapes")
                                # print(priorCloud.shape)
                                bigCloudPrior = torch.tile(priorCloud[:,:3],(self.config.training.batch_size,1,1))
                                # print(bigCloudPrior.shape)
                                relativePoints = torch.subtract(bigCloudPrior,originList[:,:,0])
                                xy = torch.square(relativePoints[:,0]) + torch.square(relativePoints[:,1])
                                newDepthPrior = torch.sqrt(xy + torch.square(relativePoints[:,2]))
                                horizontal = torch.atan2(relativePoints[:,1], relativePoints[:,0])
                                # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
                                xy = torch.sqrt(xy)
                                vertical = torch.atan2(relativePoints[:,2], xy)
                                newColPrior = torch.round(torch.divide((horizontal-horizontalMin),horizontalAngles)).int()
                                newRowPrior = torch.round(torch.divide((vertical-verticalMin),verticalAngles)).int()
                                inGridPrior = torch.logical_and(torch.logical_and(torch.greater(newColPrior,0),torch.less(newColPrior,colCount)),torch.logical_and(torch.greater(newRowPrior,0),torch.less(newRowPrior,rowCount)))
                                #ok now I need to make my new images
                                #So I need to have a IxRxC, which takes the closest depth for each B, for the row and column in new image I.
                                #But to be clear, I have OxBxRxC, which is the distance from each origin O, to the point in each sample B, at pixel R,C.
                                # Keeping B R C is good for future but a huge fucking pain for the present
                                # fuck it goodbye B R C
                                # newMagnitude, newIndices = torch.min(newDepth, dim=1)
                                newStack = []
                                finalIndStack = []
                                imageMaskStack = []
                                del bigCloudPrior

                                for origin in range(self.config.training.batch_size):
                                    new_ind = torch.argsort(torch.flatten(newDepth[origin][inGrid[origin]]),dim=0)
                                    newRowTemp =torch.flatten(newRow[origin][inGrid[origin]])[new_ind]
                                    newColTemp =torch.flatten(newCol[origin][inGrid[origin]])[new_ind]
                                    merged = torch.stack((newRowTemp,newColTemp),0)
                                    # merged is now a 2xOxBxRxC array
                                    # this does not work as it can only return a 1D array, rip. Ahh right
                                    # print(merged.shape)
                                    merged,reduced_ind = torch.unique(merged, return_inverse=True, dim=1)
                                    # print("break it down")
                                    final_ind = torch.arange(len(newDepth[1]),device=self.config.device)
                                    gridInd = inGrid[origin]
                                    # print(newDepth.shape)
                                    # print(gridInd.shape)
                                    # print(final_ind.shape)
                                    final_ind = final_ind[gridInd]
                                    final_ind = final_ind[new_ind]
                                    reduced_ind = torch.unique(reduced_ind)
                                    final_ind = final_ind[reduced_ind]
                                    # print(final_ind.shape)
                                    # print(merged.shape)
                                    # print(reduced_ind.shape)

                                    #Ok so if I want to create a new PriorCloud:
                                    #Flatten the first one so it's always unstructured, no BxCxR
                                    #collect all the final_ind for each origin as well as mask, intensity, etc
                                    #3Dgradients = X,Y,Z projection of the gradient instead of the new magnitude 
                                    #finalGradsForward = torch.sparse_coo_tensor(3Dgradients[mask].view(3,-1), final_ind[mask].view(-1),size=3,len(torch.max(final_indPrior+final_ind))).to_dense() #sets gradient to 0 for points not used
                                    #finalGradsReverse = torch.sparse_coo_tensor(3Dgradients[not mask].view(3,-1), final_ind[mask].view(-1),size=3,len(torch.max(final_indPrior+final_ind))).to_dense() #sets gradient to 0 for points not used
                                    #newCloud = torch.where(mask,forwardCloud + forwardgrads, priorCloud + ReverseGrads)
                                    #I also need to then add the individual gradietns to each image and pass it to the next loop purely for hole-filling backup

                                    # final_ind = new_ind[reduced_ind]
                                    # newCol = torch.view(newCol[reduced_ind],(config.data.batch_size,config.data.batch_size,rowCount,colCount))
                                    # newRow = torch.view(newRow[reduced_ind],(config.data.batch_size,config.data.batch_size,rowCount,colCount))

                                    imageDepth = torch.sparse_coo_tensor(merged,torch.flatten(newDepth[origin])[final_ind], size=(rowMax,colMax)).to_dense()
                                    #Don't need this as not stupid enough to run a heuristic at timestep of 0
                                    # imageXY = coo_matrix((xy[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), size=(rowMax,colMax)).toarray()[tempDepth != 0]
                                    # print(forwardDiffusion.shape)
                                    imageIntensity = torch.sparse_coo_tensor(merged,torch.flatten(forwardDiffusion[0,3,:])[final_ind], size=(rowMax,colMax)).to_dense()
                                    #Only use points taken from forward diffusion for loss
                                    imageMask = torch.sparse_coo_tensor(merged,torch.ones_like(merged[0], device=self.config.device), size=(rowMax,colMax)).to_dense()
                                    # print(torch.flatten(X[:,0]).shape)
                                    # print(torch.flatten(X[:,0])[final_ind].shape)
                                    oldDepth = torch.sparse_coo_tensor(merged,torch.flatten(X[:,0])[final_ind], size=(rowMax,colMax)).to_dense()
                                    oldIntensity = torch.sparse_coo_tensor(merged,torch.flatten(X[:,1])[final_ind], size=(rowMax,colMax)).to_dense()
                                    final_ind_image = torch.sparse_coo_tensor(merged,final_ind, size=(rowMax,colMax)).to_dense()
                                    # print(oldDepth.shape)
                                    magnitudeChange = imageDepth - oldDepth
                                    intensityChange = imageIntensity - oldIntensity

                                    
                                    new_ind = torch.argsort(torch.flatten(newDepthPrior[origin][inGridPrior[origin]]),dim=0)
                                    newRowTemp =torch.flatten(newRowPrior[origin][inGridPrior[origin]])[new_ind]
                                    newColTemp =torch.flatten(newColPrior[origin][inGridPrior[origin]])[new_ind]
                                    merged = torch.stack((newRowTemp,newColTemp),0)
                                    merged,reduced_indPrior = torch.unique(merged, return_inverse=True, dim=1)
                                    reduced_indPrior = torch.unique(reduced_indPrior)
                                    final_indPrior = torch.arange(len(newDepthPrior[1]),device=self.config.device)
                                    gridIndPrior = inGridPrior[origin]
                                    final_indPrior = final_indPrior[gridIndPrior]
                                    final_indPrior = final_indPrior[new_ind]
                                    final_indPrior = final_indPrior[reduced_indPrior]
                                    # print(final_ind.shape)
                                    # print(newDepthPrior.shape)

                                    imageDepthPrior = torch.sparse_coo_tensor(merged,torch.flatten(newDepthPrior[origin])[final_indPrior], size=(rowMax,colMax)).to_dense()
                                    #Don't need this as not stupid enough to run a heuristic at timestep of 0
                                    # imageXY = coo_matrix((xy[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), size=(rowMax,colMax)).toarray()[tempDepth != 0]
                                    # print(priorCloud.shape)
                                    imageIntensityPrior = torch.sparse_coo_tensor(merged,torch.flatten(priorCloud[0,3])[final_indPrior], size=(rowMax,colMax)).to_dense()
                                    final_ind_imagePrior = torch.sparse_coo_tensor(merged,final_indPrior, size=(rowMax,colMax)).to_dense()
                                    
                                    #fill with prior point cloud
                                    imageIntensity = torch.where(imageDepth != 0, imageIntensity, imageIntensityPrior)
                                    final_ind_imagePrior = torch.where(imageDepth != 0, final_ind_image, final_ind_imagePrior)
                                    imageDepth = torch.where(imageDepth != 0, imageDepth, imageDepthPrior)
                                    #Magnitude doesn't matter as mask is 0 for all these points, so only need to fill for intensity & depth
                                    #Ok in theory I now have a new image
                                    #Fill any holes with the Depth and Intensity that their point was changed to last time so there is something for network to actually use
                                    #To be clear, this step is taking the magnitude & intensity of the point which USED to be in that Row & Column, 
                                    #but was shifted to a different Row & Column after the prior timestep's gradients were applied
                                    #This is my solution to ensure that the reverse diffusion does not have gaps appear as points move around
                                    imageIntensity = torch.where(imageDepth != 0, imageIntensity, updatedImages[origin,1])
                                    imageDepth = torch.where(imageDepth != 0, imageDepth, updatedImages[origin,0])
                                    # print("all the shapes")
                                    # print(imageDepth.shape)
                                    # print(imageIntensity.shape)
                                    # print(imageMask.shape)
                                    # print(magnitudeChange.shape)
                                    # print(intensityChange.shape)
                                    # print(final_ind_image.shape)

                                    newStack.append(torch.stack((imageDepth,imageIntensity, magnitudeChange,intensityChange)))
                                    finalIndStack.append(final_ind_image.long())
                                    imageMaskStack.append(imageMask.bool())
                                newImages = torch.stack(newStack)
                                finalImages = torch.stack(finalIndStack)
                                maskImages = torch.stack(imageMaskStack)




                                
                                # print("The final shape")
                                # print(newImages.shape)
                                X = newImages
                                test_dsm_loss, grad = anneal_dsm_score_estimation_simultaneous(score, X[:,:2], used_sigmas, X[:,2:4], maskImages, labels,
                                                       self.config.training.anneal_power,
                                                       hook=test_hook)
                                # test_dsm_loss, grad = anneal_dsm_score_estimation_with_mask(test_score, original_test_X, None, None, test_y, test_sky, sigmas, None,
                                #                                             self.config.training.anneal_power,
                                #                                             hook=test_hook)
                                lossTotal += test_dsm_loss
                                updatedImages = X[:,:2].detach().clone()
                                grad = grad.detach()
                                # print(maskImages)

                                # print(maskImages)
                                noiseX = (grad[:,0,:,:] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1))
                                noiseY = (grad[:,0,:,:] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1))
                                noiseZ = (grad[:,0,:,:] * torch.sin(elevation).view(1,-1,1))
                                noiseI = (grad[:,1,:,:])
                                finalImages = finalImages.flatten()
                                generateNewMask = finalImages != -1
                                noiseX = noiseX.flatten()[generateNewMask]
                                noiseY = noiseY.flatten()[generateNewMask]
                                noiseZ = noiseZ.flatten()[generateNewMask]
                                noiseI = noiseI.flatten()[generateNewMask]
                                # print(maskImages)
                                maskImages = maskImages.flatten()
                                # maskImages = maskImages.view(1,1,-1)
                                # maskImages = torch.tile(maskImages,(1,4,1))
                                # print(maskImages)
                                maskImages = maskImages[generateNewMask]
                                finalImages = finalImages[generateNewMask]
                                indicesForward = finalImages[maskImages]
                                indicesReverse = finalImages[torch.logical_not(maskImages)]
                                #If I pass them both through torch unique and convert to the indices, it will naturally make them both values 0->len(array)
                                indicesForward, newForward = torch.unique(indicesForward, return_inverse=True)
                                indicesReverse, newReverse = torch.unique(indicesReverse, return_inverse=True)  
                                #Ok so for finalImages
                                #All -1 values mean I should generate a new point using that updatedImage
                                # maxSize = self.config.training.batch_size * self.config.training.image_width * self.config.training.image_size
                                # print("fuck you")
                                # print(indicesReverse.shape)
                                # print(newReverse.shape)
                                # print(torch.max(newReverse))
                                # print(noiseX[torch.logical_not(maskImages)].shape)
                                # print("extra fuck you")
                                # print(torch.count_nonzero(noiseX[torch.logical_not(maskImages)]))
                                noiseXReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseX[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()
                                noiseYReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseY[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()
                                noiseZReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseZ[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()
                                noiseIReverse = torch.sparse_coo_tensor(torch.unsqueeze(newReverse,0),noiseI[torch.logical_not(maskImages)], size=(indicesReverse.shape)).to_dense()

                                noiseXForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseX[maskImages], size=([indicesForward.shape[0]])).to_dense()
                                noiseYForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseY[maskImages], size=([indicesForward.shape[0]])).to_dense()
                                noiseZForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseZ[maskImages], size=([indicesForward.shape[0]])).to_dense()
                                noiseIForward = torch.sparse_coo_tensor(torch.unsqueeze(newForward,0),noiseI[maskImages], size=([indicesForward.shape[0]])).to_dense()
                                # print("sanity check")
                                # print(noiseXReverse.shape)


                                #I now have noise for all the original points from both point clouds, and ONLY the points actually relevant.
                                #Now I need to create the combo clouds to apply this noise to
                                #so I need only the values which exist within indicesForward
                                #ALl other to be ignored
                                #soooooo just filter it?
                                #yeah no duplicates anymore because I ran it under unique

                                #make it unique because we only want the first from each one
                                finalImages = torch.unique(finalImages)
                                #when mask is true, take the XYZ values from forwardDiffusion at the index indicated by finalImages
                                #I need to do a coo array using finalImages[mask] as the indices and forwardDiffusion as values
                                #then add more values to that array using finalimages[not mask] and priorCloud
                                #finally, for all pixels which correspond to a -1, I need to generate a new point, using updatedImages as the base
                                #
                                #In theory I could do this at the end and add them after all the noise is finished
                                #I need to redo this to ACTUALLY be able to add multiple noiseX values to the same point

                                #In contrast, 
                                # print(maskImages)
                                # print(maskImages.shape)
                                comboX = torch.cat((priorCloud[0,0][indicesReverse],forwardDiffusion[0,0][indicesForward]),-1)
                                comboY = torch.cat((priorCloud[0,1][indicesReverse],forwardDiffusion[0,1][indicesForward]),-1)
                                comboZ = torch.cat((priorCloud[0,2][indicesReverse],forwardDiffusion[0,2][indicesForward]),-1)
                                comboI = torch.cat((priorCloud[0,3][indicesReverse],forwardDiffusion[0,3][indicesForward]),-1)
                                comboCloud = torch.stack((comboX,comboY,comboZ,comboI))
                                # print(comboCloud.shape)
                                # print("why crash")
                                noiseX = torch.cat((noiseXReverse,noiseXForward),-1)
                                noiseY = torch.cat((noiseYReverse,noiseYForward),-1)
                                noiseZ = torch.cat((noiseZReverse,noiseZForward),-1)
                                noiseI = torch.cat((noiseIReverse,noiseIForward),-1)
                                noiseCloud = torch.stack((noiseX,noiseY,noiseZ,noiseI))
                                for s in range(self.config.sampling.n_steps_each):
                                
                                    step_size = self.config.sampling.step_lr * (sigmas[testTimestep] / sigmas[-1]) ** 2    
                                    # grad_likelihood = -mask * (X - originalX) # - 0.05*(1-mask)*(x_mod - raw_interp)

                                    noise2 = torch.randn_like(updatedImages)
                                    # grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                                    # grad_likelihood_norm = torch.norm(grad_likelihood.view(grad.shape[0], -1), dim=-1).mean()
                                    # noise_norm = torch.norm(noise2.view(noise2.shape[0], -1), dim=-1).mean()
                                    updatedImages = updatedImages + step_size * grad + noise2 * torch.sqrt(step_size * 2)
                                    comboCloud = comboCloud + (noiseCloud*step_size)
                                #now I need to add new points from updatedImages for anywhere that generateNewMask == 0
                                pointX = torch.add(updatedImages[:,0,:,:] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:,0]).flatten()
                                pointY = torch.add(updatedImages[:,0,:,:] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:,1]).flatten()
                                pointZ = torch.add(updatedImages[:,0,:,:] * torch.sin(elevation).view(1,-1,1),originList[:,2]).flatten()
                                toAdd = torch.stack((pointX[torch.logical_not(generateNewMask)],pointY[torch.logical_not(generateNewMask)],pointZ[torch.logical_not(generateNewMask)],updatedImages[:,1,:,:].flatten()[torch.logical_not(generateNewMask)]))
                                comboCloud = torch.cat((comboCloud,toAdd),-1)
                                # print("true check")
                                # print(comboCloud.shape)
                                priorCloud = torch.unsqueeze(comboCloud,0).detach() 
                            test_dsm_loss = lossTotal / maxTimeStepReachable
                            tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=trueStep)
                            test_tb_hook()
                            logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                            del test_score

                    if trueStep % 20 == 0:
                        if(maxTimeStepReachable < len(sigmas)):
                            maxTimeStepReachable += 1
                    if trueStep % self.config.training.snapshot_freq == 0:
                        states = [
                            score.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                        if self.config.training.snapshot_sampling:
                            if self.config.model.ema:
                                test_score = ema_helper.ema_copy(score)
                            else:
                                test_score = score

                            test_score.eval()

                            ## Different part from NeurIPS 2019.
                            ## Random state will be affected because of sampling during training time.
                            init_samples = torch.rand(36, self.config.data.channels,
                                                      self.config.data.image_size, self.config.data.image_width,
                                                      device=self.config.device)
                            init_samples = data_transform(self.config, init_samples)

                            all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                                   self.config.sampling.n_steps_each,
                                                                   self.config.sampling.step_lr,
                                                                   final_only=True, verbose=True,
                                                                   denoise=self.config.sampling.denoise)

                            sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                          self.config.data.image_size,
                                                          self.config.data.image_width)

                            sample = inverse_data_transform(self.config, sample)

                            torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))
                            torch.save(all_samples, os.path.join(self.args.log_sample_path, 'samples_all_{}.pth'.format(step)))

                            if sample.dim() == 4 and sample.size(1) == 2:  # two-channel images
                                sample = sample.transpose(1, 0)
                                sample = sample.reshape((sample.size(1)*sample.size(0), 1, sample.size(2), sample.size(3)))
                                sample = torch.cat((sample, sample, sample), 1)

                            image_grid = make_grid(sample, 6)
                            save_image(image_grid,
                                       os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))

                            del test_score
                            del all_samples

    def nvs(self):
        '''
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        dataset, _ = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=4)
        score.eval()

        dataset = KITTINVS('/mnt/data/KITTI-360', seq_number = 0)

        # compute common mask
        range_sum = np.zeros((64, 1024))
        for idx in range(100):
            lidar_range_src, v2w_src = dataset[idx]
            range_sum = range_sum + lidar_range_src[0]
        common_mask = range_sum < 1e-2

        for src_idx in [100, 200, 300, 400, 500, 1000, 1500, 2000]:
            batch_size = 24
            batch = []
            batch_gt = []
            for tgt_idx in range(src_idx + 1, src_idx + batch_size + 1):
                lidar_range_src, v2w_src = dataset[src_idx]
                lidar_range_tgt, v2w_tgt = dataset[tgt_idx]
                real, real_log, xyz_src_tgt, xyz_src_tgt_world = novel_view_synthesis(lidar_range_src, v2w_src, lidar_range_tgt, v2w_tgt, common_mask)
                batch.append(real_log)
                batch_gt.append(lidar_range_tgt)
                # pcd_src_tgt = o3d.geometry.PointCloud()
                # pcd_src_tgt.points = o3d.utility.Vector3dVector(xyz_src_tgt_world)
                # pcd_src_tgt.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([pcd_src_tgt])

            batch = np.stack(batch, axis = 0)
            samples = torch.tensor(batch)
            batch_gt = np.stack(batch_gt, axis = 0)
            samples_gt = torch.tensor(batch_gt)

            samples = samples.to(self.config.device)
            samples = data_transform(self.config, samples)

            samples_gt = samples_gt.to(self.config.device)
            samples_gt = data_transform(self.config, samples_gt)

            init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                        self.config.data.image_size, self.config.data.image_width,
                                        device=self.config.device)
            init_samples = data_transform(self.config, init_samples)


            for grad_ref in [1, 2, 0.5, 0.2]:

                all_samples, targets = anneal_Langevin_dynamics_nvs(init_samples, samples, samples_gt, score, sigmas,
                                                    self.config.sampling.n_steps_each,
                                                    self.config.sampling.step_lr,
                                                    denoise=self.config.sampling.denoise,
                                                    grad_ref=grad_ref,
                                                    sampling_step=4)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples[-3:]), total=len(all_samples[-3:]),
                                            desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'nvs_image_grid_{}_{}_{}.png'.format(grad_ref, src_idx, i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'nvs_samples_{}_{}_{}.pth'.format(grad_ref, src_idx, i)))

                    sample = targets[0]
                    sample = sample.view(sample.shape[0], self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_width)
                    sample = inverse_data_transform(self.config, sample)
                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder, 'nvs_ref_grid_{}.png'.format(src_idx)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'nvs_ref_{}.pth'.format(src_idx)))

            sample = samples_gt
            sample = sample.view(sample.shape[0], self.config.data.channels,
                                    self.config.data.image_size,
                                    self.config.data.image_width)
            sample = inverse_data_transform(self.config, sample)
            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
            save_image(image_grid, os.path.join(self.args.image_folder, 'nvs_gt_grid_{}.png'.format(src_idx)))
            torch.save(sample, os.path.join(self.args.image_folder, 'nvs_gt_{}.pth'.format(src_idx)))
        '''
        return


    def sample(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            # states = torch.load('secondSession.pth',
            states = torch.load('diffusionNet/checkpoint_148.pth',
                                map_location=self.config.device)
            # states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
            #                     map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        train_dataset, test_dataset = get_dataset(self.args, self.config)
        dataset = test_dataset
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)

        score.eval()

        if not self.config.sampling.fid:
            if self.config.sampling.inpainting:
                data_iter = iter(dataloader)
                for doThis in range(10):
                    refer_images_full,refer_mask_full, sky = next(data_iter)
                    sample_full = []
                    numSubsections = 1
                    subsectionSize = self.config.data.image_size // numSubsections
                    for subPart in range(1):
                        refer_images = refer_images_full[:,:,subPart*subsectionSize:(subPart+1)*subsectionSize,:].float().to(self.config.device)
                        refer_mask = refer_mask_full[:,:,subPart*subsectionSize:(subPart+1)*subsectionSize,:].int().to(self.config.device)
                        width = int(self.config.sampling.batch_size)
                        init_samples = torch.rand(width, self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_width,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)
                        # print("why do you hate me")
                        # print(init_samples.shape)
                        # print(refer_images.shape)
                        # print(refer_mask.shape)
                        #init_samples starts as pure random noise
                        #the function incrementally calculates gradients, and checks against the masked reference image to ensure output matches the known pixels
                        all_outputs, all_targets = anneal_Langevin_dynamics_inpainting(init_samples, refer_images, refer_mask ,score, sigmas,
                                                                self.config.sampling.n_steps_each,
                                                                self.config.sampling.step_lr,
                                                                denoise=self.config.sampling.denoise,
                                                                grad_ref=1,
                                                                sampling_step=4)
                        # print("whyyyy")
                        # print(refer_images.shape)
                        # print(sum(refer_images))
                        # torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
                        # refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                        #                                                                                      *refer_images.shape[
                        #                                                                                       1:])
                        sample_full.append(all_outputs[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_width))


                    refer_images = refer_images_full * refer_mask_full
                    refer_images = inverse_data_transform(self.config, refer_images)
                    sample = torch.cat(sample_full,2)
                    # sample = sample_full[-1]

                    if refer_images.dim() == 4 and refer_images.size(1) == 2:  # two-channel images
                        refer_images = refer_images.transpose(1, 0)
                        refer_images = refer_images.reshape((refer_images.size(1)*refer_images.size(0), 1, refer_images.size(2), refer_images.size(3)))
                        refer_images = torch.cat((refer_images, refer_images, refer_images), 1)

                    image_grid = make_grid(refer_images, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        str(doThis) + '_GT_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    np.save(os.path.join(self.args.image_folder,
                                                    str(doThis) + '_GT_completion_{}.pth'.format(self.config.sampling.ckpt_id)),refer_images.cpu().detach().numpy())

                    # save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'))

                    if not self.config.sampling.final_only:
                        for i, sample in enumerate(tqdm.tqdm(all_samples)):
                            sample = sample.view(self.config.sampling.batch_size, self.config.data.channels,
                                                 self.config.data.image_size,
                                                 self.config.data.image_width)

                            sample = inverse_data_transform(self.config, sample)

                            if sample.dim() == 4 and sample.size(1) == 2:  # two-channel images
                                sample = sample.transpose(1, 0)
                                sample = sample.reshape((sample.size(1)*sample.size(0), 1, sample.size(2), sample.size(3)))
                                sample = torch.cat((sample, sample, sample), 1)

                            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                            save_image(image_grid, os.path.join(self.args.image_folder, str(doThis) + '_image_grid_{}.png'.format(i)))
                            np.save(sample.cpu().detach().numpy(), os.path.join(self.args.image_folder, str(doThis) + '_completion_{}.pth'.format(i)))
                    else:
                        # sample = all_outputs[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                        #                               self.config.data.image_size,
                        #                               self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)
                        maskedSample = sample * sky

                        if sample.dim() == 4 and sample.size(1) == 2:  # two-channel images
                            sample = sample.transpose(1, 0)
                            sample = sample.reshape((sample.size(1)*sample.size(0), 1, sample.size(2), sample.size(3)))
                            sample = torch.cat((sample, sample, sample), 1)
                            maskedSample = maskedSample.transpose(1, 0)
                            maskedSample = maskedSample.reshape((maskedSample.size(1)*maskedSample.size(0), 1, maskedSample.size(2), maskedSample.size(3)))
                            maskedSample = torch.cat((maskedSample, maskedSample, maskedSample), 1)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder,
                                                            str(doThis) + '_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                        np.save(os.path.join(self.args.image_folder,
                                                        str(doThis) + '_completion_{}.pth'.format(self.config.sampling.ckpt_id)),sample.cpu().detach().numpy())
                        image_grid = make_grid(maskedSample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder,
                                                            str(doThis) + '_Masked_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                        np.save(os.path.join(self.args.image_folder,
                                                        str(doThis) + '_Masked_completion_{}.pth'.format(self.config.sampling.ckpt_id)),maskedSample.cpu().detach().numpy())

            
            elif self.config.sampling.densification:
                data_iter = iter(dataloader)
                samples, masks, _ = next(data_iter)
                samples = samples.to(self.config.device)
                samples = data_transform(self.config, samples)

                init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                            self.config.data.image_size, self.config.data.image_width,
                                            device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                #if self.config.sampling.diverse:
                #    samples[1:] = samples[2]
                #    print("same init guidance")


                for grad_ref in [1]:

                    all_samples, targets = anneal_Langevin_dynamics_densification(init_samples, samples, score, sigmas,
                                                        self.config.sampling.n_steps_each,
                                                        self.config.sampling.step_lr,
                                                        denoise=self.config.sampling.denoise,
                                                        grad_ref=grad_ref,
                                                        sampling_step=4)

                    if not self.config.sampling.final_only:
                        for i, sample in tqdm.tqdm(enumerate(all_samples[-3:]), total=len(all_samples[-3:]),
                                                desc="saving image samples"):
                            sample = sample.view(sample.shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_width)

                            sample = inverse_data_transform(self.config, sample)

                            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                            save_image(image_grid, os.path.join(self.args.image_folder, 'densify_image_grid_{}_{}.png'.format(grad_ref, i)))
                            torch.save(sample, os.path.join(self.args.image_folder, 'densify_samples_{}_{}.pth'.format(grad_ref, i)))

                        sample = targets[0]
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'densify_ref_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'densify_ref_{}.pth'.format(i)))

                    else:
                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                    self.config.data.image_size,
                                                    self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder,
                                                            'densify_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                        #torch.save(sample, os.path.join(self.args.image_folder,
                        #                                'densify_samples_{}.pth'.format(self.config.sampling.ckpt_id)))
                        torch.save(sample, os.path.join(self.args.image_folder,
                                                        'densify_samples_result.pth'))
                        torch.save(samples, os.path.join(self.args.image_folder,
                                                        'densify_samples_target.pth'))

            else:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_width,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=True,
                                                       final_only=self.config.sampling.final_only,
                                                       denoise=self.config.sampling.denoise)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_width)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    #save_image(image_grid, os.path.join(self.args.image_folder,
                    #                                    'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    #torch.save(sample, os.path.join(self.args.image_folder,
                    #                                'samples_{}.pth'.format(self.config.sampling.ckpt_id)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid.png'))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'samples.pth'))

        else:
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size
            if self.config.sampling.data_init:
                dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                        num_workers=4)
                data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.sampling.data_init:
                    try:
                        samples, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples = samples + sigmas_th[0] * torch.randn_like(samples)
                else:
                    samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_width, device=self.config.device)
                    samples = data_transform(self.config, samples)

                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       denoise=self.config.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = inverse_data_transform(self.config, img)
                    torch.save(img, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(img_id)))
                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1

    def test(self):
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas = get_sigmas(self.config)

        dataset, test_dataset = get_dataset(self.args, self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=True)

        verbose = False
        for ckpt in tqdm.tqdm(range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, 5000),
                              desc="processing ckpt:"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            step = 0
            mean_loss = 0.
            mean_grad_norm = 0.
            average_grad_scale = 0.
            for x, y in test_dataloader:
                step += 1

                x = x.to(self.config.device)
                x = data_transform(self.config, x)

                with torch.no_grad():
                    test_loss = anneal_dsm_score_estimation(score, x, sigmas, None,
                                                            self.config.training.anneal_power)
                    if verbose:
                        logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))

                    mean_loss += test_loss.item()

            mean_loss /= step
            mean_grad_norm /= step
            average_grad_scale /= step

            logging.info("ckpt: {}, average test loss: {}".format(
                ckpt, mean_loss
            ))

