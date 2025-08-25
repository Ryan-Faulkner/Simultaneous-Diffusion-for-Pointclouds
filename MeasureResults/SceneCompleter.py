import os, glob, pickle
import sys
from glob import glob
sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import metrics.iou as lidargen_iou
import numpy as np
import torch
import math
from scipy.sparse import coo_matrix
import scipy
if __name__ == '__main__':
    expOG = "DGXDataLiDARGenSettings/SceneCompletionWide"
    # print("hwy")
    # listOfExperiments = glob(expOG)
    # print(listOfExperiments)
    # for experiment in listOfExperiments:
    exp = expOG
    expRaw = os.path.join(exp, "Raw")
    expSky = os.path.join(exp, "Sky")
    expSimultaneous = os.path.join(exp, "Organised")
    expFinal = os.path.join(exp, "Final")
    expFinalSmall = os.path.join(exp, "FinalSmall")
    os.system("rm -r " + str(expFinal))
    os.system("mkdir " + str(expFinal))

    # os.system("rm -r " + str(expLiDARGenOrganised))
    # os.system("mkdir " + str(expLiDARGenOrganised))

    # os.system("rm -r " + str(expSimultaneousOrganised))
    # os.system("mkdir " + str(expSimultaneousOrganised))
    existVals = np.load("existTotalLiDARGenSettings.npy")
    existVals = existVals > np.max(existVals) / 3
    #Get pixels which show up in less than one third of scans.
    #Do some erosion to make sure poor points don't get used. Don't run top two rows to ensure erosion doesn't ruin them by accident due to border issues
    existVals[2:] = scipy.ndimage.binary_erosion(existVals[2:], border_value = 1, iterations=4)
    rowCount = 64#scan.row_count
    colCount = 1024#scan.column_count

    rowMax = rowCount
    colMax = colCount
    #LiDAR Specs:
    # horizontalScope = 180 * 2
    # # verticalScope = 60 #+40 down to -40
    # #KITTI Specs:
    # horizontalScope = 360
    # verticalScope = 26.8 #Kitti specifically assumes +2 from origin down to -24.8 below the scanner location.
    #KITTI Specs LIDARGen:
    horizontalScope = 360
    verticalScope = 28 #Kitti specifically assumes +2 from origin down to -24.8 below the scanner location.
    verticalPositive = 3
    maxRange = 2057.701 #Current record set by Penrice
    minPixels = rowMax*colMax/6


    horizontalAngles = math.radians(horizontalScope) / colMax #6 minutes of an arc for a 800x3600 image, 12 minutes for 800x1800 (assumign 360 coverage)
    verticalAngles = math.radians(verticalScope) / rowMax #6 arc minutes (one tenth of a degree). This means every ten pixels is one degree, 80 pixels = +- 40 degrees
    # verticalAngles = 0.00145444 + 0.00028235549 # 5 minutes and 58 seconds
    #If centre of image is a perfect 0,0 origin, then Xmin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
    horizontalMin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
    #LiDAR, with 0,0 in centre:
    # verticalMin = rowCount//(-2) * verticalAngles + (verticalAngles/2)
    #My LiDAR training version, reducing useless sky pixels by cutting top thirty degrees off
    # verticalMin = rowCount*3//(-4) * verticalAngles + (verticalAngles/2)
    #KITTI, with 0,0 5 pixels from the top:
    #A smarter min caluclation
    verticalMin = math.radians(verticalPositive - (verticalScope))
    azimuth = np.reshape(((np.arange(colMax-1,-1,-1) * horizontalAngles) + horizontalMin),(1,colCount))
    elevation = np.reshape(((np.arange(rowMax-1,-1,-1) * verticalAngles) + verticalMin),(rowCount,1))
    toGlob = expSimultaneous
    for inputScan in  np.sort(glob(toGlob + '/*')):
        finalCloud = []
        for filePath in np.sort(glob(inputScan + '/*.npy')):
            file = np.load(filePath)
            filename = filePath.split('/')[-1]
            inputName = inputScan.split('/')[-1]
            originMods = np.load(inputScan + '/Origins/' + filename)
            print(originMods.shape)
            segmentations = torch.load(inputScan + '/result_rangenet_segmentations/' + filename[:-3] + 'pth').cpu().numpy()
            #Convert the segmentations
            learningMap = {0:0, #unlabeled
            1:26, #car
            2:33,#bicycle
            3:32,#motorbike
            4:27,#truck
            5:43,#other Vehicle
            6:24,#person
            7:25,#bicyclist
            8:25,#motorcyclist
            9:7,#road
            10:9,#parking
            11:8,#sidewalk
            12:6,#other ground
            13:11,#building
            14:13,#fence
            15:21,#vegetation
            16:21,#trunk
            17:22,#terrain
            18:17,#pole
            19:20,#traffic-sign
            }
            actualSegementations = np.vectorize(learningMap.get)(segmentations)

            rawScan = np.load(expRaw + '/' + inputName + '.npy')
            #get origin mod
            roughMedian = np.array([ 0.73530043,  0.12196524, -1.23688836])
            actualOriginMod = roughMedian - np.squeeze(originMods) 
            meanXYZ = np.median(rawScan,axis=0)
            #Now I need to separate intensity & get x,y,z,1 values
            # intensity = scanPoints[:,-1]
            finalMod = meanXYZ  - actualOriginMod

            distance = np.squeeze(file[:file.shape[0]//2,0])
            intensity = np.squeeze(file[file.shape[0]//2:,0])
            realDistance = (np.power(2,(np.abs(distance)*6))-1)
            #Now I have distance, segmentations, and required origin mod, so just need to project to point cloud

            pointX = np.add(realDistance * np.cos(azimuth) * np.cos(elevation),finalMod[0])
            pointY = np.add(realDistance * np.sin(azimuth) * np.cos(elevation),finalMod[1])
            pointZ = np.add(realDistance * np.sin(elevation),finalMod[2])



            imageXY = np.zeros([rowMax,colMax]) + maxRange
            imageDepth = np.zeros([rowMax,colMax]) + maxRange
            imageIntensity = np.zeros([rowMax,colMax])
            imageColour = np.zeros([rowMax,colMax,3])
            obfuscationMask = np.zeros([rowMax,colMax]).astype(bool)
            fakeOrigin = finalMod# + modification
            #shift 1 metre to "true centre"
            # distanceToTrue = (randomPoint-trueOrigin)
            # fakeOrigin = fakeOrigin + (distanceToTrue/np.sqrt(np.sum(np.square(distanceToTrue))))

            relativePoints = rawScan - fakeOrigin
            xy = np.square(relativePoints[:,0]) + np.square(relativePoints[:,1])
            newDepth = np.sqrt(xy + np.square(relativePoints[:,2]))
            horizontal = np.arctan2(relativePoints[:,1], relativePoints[:,0])
            # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
            xy = np.sqrt(xy)
            vertical = np.arctan2(relativePoints[:,2], xy)
            newCol = np.round(np.divide((horizontal-horizontalMin),horizontalAngles)).astype(int)
            newRow = np.round(np.divide((vertical-verticalMin),verticalAngles)).astype(int)

            #lidargen clamps the edges here, which I guess I can also do
            # newCol = np.floor(proj_x)
            newCol = np.minimum(colMax - 1, newCol)
            newCol = np.maximum(0, newCol).astype(np.int32)   # in [0,W-1]
            # self.proj_x = np.copy(proj_x)  # store a copy in orig order

            # newRow = np.floor(proj_y)
            newRow = np.minimum(rowMax - 1, newRow)
            newRow = np.maximum(0, newRow).astype(np.int32)   # in [0,H-1]
            # print(np.min(newRow))
            # print(np.median(newRow))
            # print(np.max(newRow))
            #I need to make an image here
            #Something along lines of 
            # for each pixel
            # mask = newCol == pixelCol and newRow == pixelRow
            # bestIndex = np.argmin(newDepth[mask])
            # if(newDepth[mask][bestIndex] < imageDepth[modification,-1-pixelRow,-pixelCol-1]):
            #    replace it
            inGrid = np.logical_and(np.logical_and(np.greater(newCol,0),np.less(newCol,colCount)),np.logical_and(np.greater(newRow,0),np.less(newRow,rowCount)))
            # indices = np.argwhere(inGrid)
            # closerPoint = imageDepth[modification,-1-newRow[indices],-newCol[indices]-1] > newDepth[indices]
            # print("now crash")
            # depthArray = sparse.COO((newDepth, (newRow[np.nonzero(inGrid)[0]], newCol[np.nonzero(inGrid)[0]])), [shape=(800, 1800)])
            # if(len(np.nonzero(inGrid)[0]) == 0):
            #     continue
            # if(not groundTruth):
            #     # isBetter = imageDepthGT[modification,-1-newRow[inGrid],-newCol[inGrid]-1] > newDepth[inGrid]
            # # else:
            #     isBetter = imageDepth[newRow[inGrid],newCol[inGrid]] > newDepth[inGrid]
            #     inGrid[inGrid] = isBetter
            # newGrid = np.logical_and(inGrid, oldIndices == -1)
            #Second method likes to just ignore points that don't get used
            # newIndices = np.unique(PointCloudIndices[newGrid], return_inverse = True)[-1]
            # oldIndices[newGrid] = newIndices
            #However Densification requires the actual index relative to entire input point cloud 
            # oldIndices[newGrid] = PointCloudIndices[newGrid]
            # if(len(np.nonzero(inGrid)[0]) < minPixels):
            #     # print("not enough pixels")False
            #     print("too few pixels")
            #     print(len(np.nonzero(inGrid)[0]))
            #     fakeOrigin = origin
            #     #shift 1 metre to "true centre"
            #     # distanceToTrue = (randomPoint-trueOrigin)
            #     # fakeOrigin = fakeOrigin + (distanceToTrue/np.sqrt(np.sum(np.square(distanceToTrue))))

            #     relativePoints = point_cloud - fakeOrigin
            #     xy = np.square(relativePoints[:,0]) + np.square(relativePoints[:,1])
            #     newDepth = np.sqrt(xy + np.square(relativePoints[:,2]))
            #     horizontal = np.arctan2(relativePoints[:,1], relativePoints[:,0])
            #     # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
            #     xy = np.sqrt(xy)
            #     vertical = np.arctan2(relativePoints[:,2], xy)
            #     newCol = np.round(np.divide((horizontal-horizontalMin),horizontalAngles)).astype(int)
            #     newRow = np.round(np.divide((vertical-verticalMin),verticalAngles)).astype(int)
            #     # print(np.min(newRow))
            #     # print(np.median(newRow))
            #     # print(np.max(newRow))
            #     #I need to make an image here
            #     #Something along lines of 
            #     # for each pixel
            #     # mask = newCol == pixelCol and newRow == pixelRow
            #     # bestIndex = np.argmin(newDepth[mask])
            #     # if(newDepth[mask][bestIndex] < imageDepth[modification,-1-pixelRow,-pixelCol-1]):
            #     #    replace it
            #     inGrid = np.logical_and(np.logical_and(np.greater(newCol,0),np.less(newCol,colCount)),np.logical_and(np.greater(newRow,0),np.less(newRow,rowCount)))

            new_ind = np.argsort(newDepth[inGrid])
            newRow = newRow[inGrid][new_ind]
            # print(np.max(newCol))
            newCol = newCol[inGrid][new_ind]
            # print(np.max(newCol))
            merged = np.stack((newRow,newCol))
            # print("number of pixels")
            # print(len(merged[0]))
            reduced_ind = np.unique(merged, return_index = True, axis=1)[-1]
            # print(len(reduced_ind))
            # print(np.max(newCol[reduced_ind]))
            final_ind = np.arange(len(newDepth))[inGrid][new_ind][reduced_ind]

            tempDepth = coo_matrix((newDepth[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()
            imageDepth[tempDepth != 0] = tempDepth[tempDepth != 0]
            imageXY[tempDepth != 0] = coo_matrix((xy[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
         
            #Now flip so image is as I expect it to be
            imageDepth = np.flip(imageDepth)
            # imageColour = np.flip(imageColour)
            # imageRGBA = np.flip(imageRGBA,axis=(0,1))
            # imageLabel = np.flip(imageLabel)
            imageXY = np.flip(imageXY)
            xy = np.square(pointX) + np.square(pointY)
            xy = np.sqrt(xy)
            minDepth = np.zeros([colMax]) + maxRange
            skyMask = np.zeros([rowMax,colMax]).astype(bool)
            skyMask[0,:] = True
            skyMask[1,:] = True
            for row in range(2,rowCount-1):
                obfuscationMask[row,:] = imageXY[row,:] > minDepth+5

                equalMask = np.concatenate((np.zeros((1)),np.add((imageXY[row,:] != minDepth).astype(int), np.add((imageXY[row-1,:] != minDepth).astype(int), (imageXY[row+1,:] != minDepth).astype(int))),np.zeros((1))), axis=-1)
                # print(equalMask.shape)
                equalMask = np.add(equalMask[1:-1], np.add(equalMask[:-2],equalMask[2:]))
                equalMask = (equalMask <= 1).astype(bool) 
                # print(obfuscationMask.shape)
                # print(skyMask.shape)
                # print(imageXY.shape)
                currentSky = np.logical_and(equalMask,skyMask[row-1,:] == 1)
                skyMask[row,:] = currentSky
                currentSky = np.logical_not(currentSky)
                newMin = np.minimum(imageXY[row,:], minDepth)
                minDepth[currentSky] = newMin[currentSky] 
            # penultimateCloud = np.stack((pointX,pointY,pointZ,actualSegementations),1)
            print(existVals.shape)
            print(realDistance.shape)
            mask = np.logical_and(existVals,realDistance > 1.5)
            mask = np.logical_and(mask,np.logical_not(skyMask))
            finalCloud.append(np.stack((pointX[mask],pointY[mask],pointZ[mask],actualSegementations[mask]),1))
        smallScope = np.load(expFinalSmall + '/' + inputName + '.npy')
        finalCloud.append(smallScope)
        actualFinal = np.concatenate(finalCloud,0)
        np.save(expFinal + '/' + inputName, actualFinal)



#To DO
#Make program that just fucking shifts all the results into the format the existing LiDARGen evaluation code expects - that means batch size 1, etc etc. Check the original iou code to see how it wants it's shit saved