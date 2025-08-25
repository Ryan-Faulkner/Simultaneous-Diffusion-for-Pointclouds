import torch
#import math
import datetime
import gc
import os
import math
import colorsys
 #comment this out for cpu, uncomment for gpu
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

import time
import math
import numpy as np
import h5py
from torch.utils.data import Dataset
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from mapteksdk.project import Project
from mapteksdk.data import PointSet
from mapteksdk.data import Scan
from mapteksdk.data import SubblockedBlockModel
from torch.linalg import svd
# import sparse
import copy
from PIL import Image
from PIL import ImageChops

from scipy.sparse import coo_matrix

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

project = Project()
files = project.get_children(project.find_object(path="scans//ClassColour")).names()
# files = files[28:]
# files = files[33:]
#only do the 3 I care about
# files = ["20100603 penrice_stn16_nth_C1L_01","20100603 penrice_stn05_nth_C2L_01","20100603 penrice_stn06_nth_C2L_01", "20100603 penrice_stn04_nth_C1L_01", "20100603 penrice_stn17_nth_C2L_01"]
# Get the data  
# a = datetime.datetime.now()
print("Getting Data")
maxI = 5000
maxRange = 2057.701 #Current record set by Penrice
# for filename in files:
#     print(filename)
#     with project.edit("scans//OG//" + filename) as scan:
#         tempMax = np.max(scan.point_ranges[scan.point_ranges != 10000]) #sometimes scanner has error and records point as 10,000 away
#         if(tempMax > maxRange):
#           print("MAX CHANGEDMAX CHANGEDMAX CHANGEDMAX CHANGEDMAX CHANGEDMAX CHANGEDMAX CHANGEDMAX CHANGED")
#           maxRange = tempMax
        #tempMax = np.max(scan.point_intensity[scan.point_intensity < 4000]) #sometimes scanner has error and records point as 10,000 away
        #if(tempMax > maxI):
        #    maxI = tempMax
print("Max is:")
print(maxI)
print(maxRange)

colorArray = _get_colors(20)
colorArray[0] = [0,0,0]

hypotheticalViewPoint = [0,0,0]
horizontalAngles = np.zeros(1)
verticalAngles = np.zeros(1)
rowCount = 0
colCount = 0
originCounter = 0
originList = np.zeros([len(files),3])
quickCounter = 0
print("getting origins")
for filename in files:
    # print(fileNum)
    # filename = files[fileNum]
    print(filename)
    with project.read("scans//ClassColour//" + filename) as scan:
        originList[quickCounter] = scan.origin
        quickCounter = quickCounter+1
print("origins before "  + str(len(originList)))
fileIndexes = np.unique((originList/10).astype(int), return_index=True, axis=0)[-1]
originList = originList[fileIndexes]
print("origins after "  + str(len(originList)))
print(originList)
# for scanOriginIndex in range(len(files)):
                                
#     saveLocation = "D:/ImagesFakeFull/" + str(scanOriginIndex) 

    # if not os.path.exists(saveLocation):
    #     os.mkdir(saveLocation)
    # if not os.path.exists(saveLocation + '/Distance'):
    #     os.mkdir(saveLocation + '/Distance')
    # # if not os.path.exists(saveLocation + '/DistanceReversed'):
    # #     os.mkdir(saveLocation + '/DistanceReversed')
    # if not os.path.exists(saveLocation + '/Intensity'):
    #    os.mkdir(saveLocation + '/Intensity')
    # if not os.path.exists(saveLocation + '/Label'):  
    #     os.mkdir(saveLocation + '/Label')
    # if not os.path.exists(saveLocation + '/Bright'):
    #     os.mkdir(saveLocation + '/Bright')
    # if not os.path.exists(saveLocation + '/RGBPoint'):
    #     os.mkdir(saveLocation + '/RGBPoint')

modificationNum = 5
for modification in range(modificationNum):
    for scanOriginIndex in range(len(originList)):
        a = datetime.datetime.now()
        print("new scan origin")
        print(scanOriginIndex)
        #For each scan origin
        quickCounter = 0

        rowMax = 800#scan.row_count
        colMax = 1800#scan.column_count

        # if(scan.max_range > maxRange):
            # maxRange = scan.max_range
        # if(len(scan.point_attributes['ClassFuckYou2'] != 0) < 10000): #if less than 10K points labelled, skip
        #     print("Skipped due to not labelled yet")
        #     continue
        print(rowMax)
        print(colMax)


        imageRGBA = np.zeros([rowMax,colMax,3]).astype('uint8') #Have to add 1 as rowMax is the highest row number starting at 0.
        # image = np.zeros([rowMax+1,colMax+1]) #Have to add 1 as rowMax is the highest row number starting at 0.
        imageIntensity = np.zeros([rowMax,colMax]) #Have to add 1 as rowMax is the highest row number starting at 0.
        imageLabel = np.zeros([rowMax,colMax])
        imageXY = np.zeros([rowMax,colMax]) + maxRange
        imageDepth = np.zeros([rowMax,colMax]) + maxRange

        imageRGBAGT = np.zeros([rowMax,colMax,3]).astype('uint8') #Have to add 1 as rowMax is the highest row number starting at 0.
        # image = np.zeros([rowMax+1,colMax+1]) #Have to add 1 as rowMax is the highest row number starting at 0.
        imageIntensityGT = np.zeros([rowMax,colMax]) #Have to add 1 as rowMax is the highest row number starting at 0.
        imageLabelGT = np.zeros([rowMax,colMax])
        imageXYGT = np.zeros([rowMax,colMax]) + maxRange
        imageDepthGT = np.zeros([rowMax,colMax]) + maxRange

        for fileNum in range(len(files)):

            h = datetime.datetime.now()
            # a = datetime.datetime.now()
            # print(a-d)
            # d = a
            # if(fileNum > 5):
            #     continue

            #do GT first
            if(fileNum == 0):
                filename = files[fileIndexes[scanOriginIndex]]
            elif(fileNum == fileIndexes[scanOriginIndex]):
                filename = files[0]
            else:
                filename = files[fileNum]
            print(quickCounter)
            quickCounter = quickCounter + 1
            #collect every point in the scene
            # print(filename)
            groundTruth = False
            if(filename == files[fileIndexes[scanOriginIndex]]):
                print("This is the GT scan")
                groundTruth = True
            with project.read("scans//ClassColour//" + filename) as scan:
                print(scan.points.shape)

                # imageRGBA = np.zeros([colMax+1,rowMax+1,3]) #Have to add 1 as rowMax is the highest row number starting at 0.
                # image = np.zeros([colMax+1,rowMax+1]) #Have to add 1 as rowMax is the highest row number starting at 0.
                # imageIntensity = np.zeros([colMax+1,rowMax+1]) #Have to add 1 as rowMax is the highest row number starting at 0.
                # imageLabel = np.zeros([colMax+1,rowMax+1])
                counter = 0
                # print(len(scan.point_to_grid_index[0]))
                # print(len(scan.point_intensity[:]))
                # print("For loop starting")

                horizontalAngles = 0.00174533 * 2 #6 minutes of an arc for a 800x3600 image, 12 minutes for 800x1800 (assumign 360 coverage)
                verticalAngles = 0.00174533 #6 minutes of an arc
                # verticalAngles = 0.00145444 + 0.00028235549 # 5 minutes and 58 seconds
                rowCount = 800#scan.row_count
                colCount = 1800#scan.column_count
                #If centre of image is a perfect 0,0 origin, then Xmin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
                horizontalMin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
                verticalMin = rowCount//(-2) * verticalAngles + verticalAngles/2
                # print("entering loop")
                with project.read("scans//OG//" + filename) as scanOG:
                        # vertMin = -0.66886123 #This is ~-40 degress. Vertical has 400 pixels from -40 to 0, and another 400 from 0 to +40.
                        # horizontalMin =
                    #These are the azimuth and elevation respectively for each individual cell of the grid. For reasons that are unclear, cell's index must be obtained as [row*(colCount)+colCount]
                    # horizontalAngles = scan.horizontal_angles
                    # verticalAngles = scan.vertical_angles
                    #The grid having
                    #Ok now we want to test my fake-zoom

                    #For each point (currently in 1 scan, potentially in multiple)
                    #Slot it into the appropriate cell
                    #Unless that cell already has a point with a smaller depth
                    modificationDict = {0 : (0,0,0),
                               1 : (10,0,0),
                               2 : (0,10,0),
                               3 : (0,0,10),
                               4 : (10,10,0),
                               5 : (-10,0,0),
                               6 : (0,-10,0),
                               7 : (-10,-10,0),
                               8 : (20,0,0),
                               9 : (0,20,0),
                               10 : (20,20,0),
                               11 : (-20,0,0),
                               12 : (0,-20,0),
                               13 : (-20,-20,0),
                    }
                    # print(scan.point_to_grid_index.shape)
                    fakeOrigin = originList[scanOriginIndex]
                    #smart numpy calculations
                    c = a
                    # for modification in range(modificationNum):
                        # b = datetime.datetime.now()
                        # print("modification " + str(modification))
                        # print(b-c)
                        # c = b

                    fakeOrigin = fakeOrigin + modificationDict[modification]
                    relativePoints = scan.points - fakeOrigin
                    xy = np.square(relativePoints[:,0]) + np.square(relativePoints[:,1])
                    newDepth = np.sqrt(xy + np.square(relativePoints[:,2]))
                    horizontal = np.arctan2(relativePoints[:,1], relativePoints[:,0])
                    # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
                    xy = np.sqrt(xy)
                    vertical = np.arctan2(relativePoints[:,2], xy)
                    newCol = np.round(np.divide((horizontal-horizontalMin),horizontalAngles)).astype(int)
                    newRow = np.round(np.divide((vertical-verticalMin),verticalAngles)).astype(int)
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
                    if(len(np.nonzero(inGrid)[0]) == 0):
                        continue
                    if(not groundTruth):
                        # isBetter = imageDepthGT[modification,-1-newRow[inGrid],-newCol[inGrid]-1] > newDepth[inGrid]
                    # else:
                        isBetter = imageDepth[newRow[inGrid],newCol[inGrid]] > newDepth[inGrid]
                        inGrid[inGrid] = isBetter
                        if(len(np.nonzero(inGrid)[0]) == 0):
                            continue
                    print("for each index out of " + str(len(np.nonzero(inGrid)[0])))
                    c = datetime.datetime.now()
                    print("time taken for this stage c:")
                    print(c-h)
                    #hoping this is fast
                    new_ind = np.argsort(newDepth[inGrid])
                    newRow = newRow[inGrid][new_ind]
                    newCol = newCol[inGrid][new_ind]
                    merged = np.stack((newRow,newCol))
                    reduced_ind = np.unique(merged, return_index = True, axis=1)[-1]
                    final_ind = np.arange(len(newDepth))[inGrid][new_ind][reduced_ind]
                    # newRow = newRow[final_ind]
                    # newCol = newCol[final_ind]
                    d = datetime.datetime.now()
                    print("time taken for this stage d:")
                    print(d-c)

                    tempDepth = coo_matrix((newDepth[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()
                    imageDepth[tempDepth != 0] = tempDepth[tempDepth != 0]
                    imageXY[tempDepth != 0] = coo_matrix((xy[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
                    imageIntensity[tempDepth != 0] = coo_matrix((scan.point_intensity[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
                    

                    imageRGBA[tempDepth != 0,0] = coo_matrix((scanOG.point_colours[final_ind,0].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
                    imageRGBA[tempDepth != 0,1] = coo_matrix((scanOG.point_colours[final_ind,1].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
                    imageRGBA[tempDepth != 0,2] = coo_matrix((scanOG.point_colours[final_ind,2].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
                    # imageRGBA[tempDepth != 0,0] = np.stack((imageR,imageG,imageB), axis=-1)[tempDepth != 0]
                    imageLabel[tempDepth != 0] = coo_matrix((scan.point_attributes['ClassFuckYou2'][final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0] + 2

                    e = datetime.datetime.now()
                    print("time taken for this stage e:")
                    print(e-d)

                    # for index in np.nonzero(inGrid)[0]:
                    #     # print(index)
                    #     if(imageDepth[-1-newRow[index],-newCol[index]-1] > newDepth[index]):
                    #         imageDepth[-1-newRow[index],-newCol[index]-1] = newDepth[index] #For a nice image gotta flip it on row axis
                    #         imageXY[-1-newRow[index],-newCol[index]-1] = xy[index]
                    #         imageIntensity[-1-newRow[index],-newCol[index]-1] = (scan.point_intensity[index]) #For a nice image gotta flip it on row axis

                    #         imageRGBA[-1-newRow[index],-newCol[index]-1] = (scanOG.point_colours[index,:3]).astype('uint8') #For a nice image gotta flip it on row axis

                    #         imageLabel[-1-newRow[index],-newCol[index]-1] = (scan.point_attributes['ClassFuckYou2'][index]) + 2
                    if(groundTruth):
                        imageDepthGT = np.flip(copy.deepcopy(imageDepth)) #For a nice image gotta flip it on row axis
                        imageIntensityGT= np.flip(copy.deepcopy(imageIntensity))
                        imageRGBAGT= np.flip(copy.deepcopy(imageRGBA),axis=(0,1))
                        imageLabelGT= np.flip(copy.deepcopy(imageLabel))
                        imageXYGT= np.flip(copy.deepcopy(imageXY))
                            # image[col,row] = (scan.point_ranges[counter]/maxRange)*255 #For a nice image gotta flip it on row axis
                        # imageIntensity[col,row] = (scan.point_intensity[counter]/maxI)*255 #For a nice image gotta flip it on row axis
                        # imageRGBA[col,row] = (scanOG.point_colours[counter][:3]) #For a nice image gotta flip it on row axis
                        # imageLabel[col,row] = (scan.point_attributes['ClassNum'][counter])

        print("For loop ended")
        f = datetime.datetime.now()
        # a = datetime.datetime.now()
        # print(a-d)
        imageDepth = np.flip(imageDepth)
        imageIntensity = np.flip(imageIntensity)
        imageRGBA = np.flip(imageRGBA,axis=(0,1))
        imageLabel = np.flip(imageLabel)
        imageXY = np.flip(imageXY)
        #Now filter the images
        minDepth = np.zeros([colMax]) + maxRange
        obfuscationMask = np.zeros([rowMax,colMax]).astype(bool)
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
        obfuscationMask[-1,:] = imageXY[-1,:] > minDepth+5

        #now for GT
        minDepth = np.zeros([colMax]) + maxRange
        obfuscationMaskGT = np.zeros([rowMax,colMax]).astype(bool)
        skyMaskGT = np.zeros([rowMax,colMax]).astype(bool)
        skyMaskGT[0,:] = True
        skyMaskGT[1,:] = True
        for row in range(2,rowCount-1):
            obfuscationMaskGT[row,:] = imageXYGT[row,:] > minDepth+5

            equalMask = np.concatenate((np.zeros((1)),np.add((imageXYGT[row,:] != minDepth).astype(int), np.add((imageXYGT[row-1,:] != minDepth).astype(int), (imageXYGT[row+1,:] != minDepth).astype(int))),np.zeros((1))), axis=-1)
            # print(equalMask.shape)
            equalMask = np.add(equalMask[1:-1], np.add(equalMask[:-2],equalMask[2:]))
            equalMask = (equalMask <= 1).astype(bool) 
            # print(obfuscationMask.shape)
            # print(skyMask.shape)
            # print(imageXY.shape)
            currentSky = np.logical_and(equalMask,skyMaskGT[row-1,:] == 1)
            skyMaskGT[row,:] = currentSky
            currentSky = np.logical_not(currentSky)
            newMin = np.minimum(imageXYGT[row,:], minDepth)
            minDepth[currentSky] = newMin[currentSky] 
        obfuscationMaskGT[-1,:] = imageXYGT[-1,:] > minDepth+5

        #Now get the median skyline to prevent the sky going stupidly low
        #Don't include any columns whhich are all Sky in this calculation
        maskSum = np.sum(skyMask,0)
        # for modification in range(modificationNum):
            # numColumns = np.sum(maskSum == rowCount,-1)
        skyline = int(np.median(maskSum[skyMask[-2,:] != 1]))
        obfuscationMask[skyline:, maskSum > skyline] = np.logical_or(obfuscationMask[skyline:, maskSum > skyline], skyMask[skyline:, maskSum > skyline]) 
        obfuscationMaskGT[skyline:, maskSum > skyline] = np.logical_or(obfuscationMask[skyline:, maskSum > skyline], skyMaskGT[skyline:, maskSum > skyline])
        # obfuscationMask[:skyline] = 0
        # obfuscationMaskGT[:skyline] = 0
        obfuscationMask = np.logical_or(obfuscationMask, np.expand_dims(skyMask[-2,:],0))
        obfuscationMaskGT = np.logical_or(obfuscationMaskGT,np.expand_dims(skyMaskGT[-2,:],0))




        imageLabel[skyMask] = 1
        imageLabelGT[skyMaskGT] = 1

        g = datetime.datetime.now()
        print("time taken for this stage f:")
        print(g-f)

        i = datetime.datetime.now()
        print("time taken for this image aside from saving:")
        print(i-a)

        saveLocation = "D:/ImagesFakeMasked/" + str(scanOriginIndex) 
        if not os.path.exists("D:/ImagesFakeMasked"):
            os.mkdir("D:/ImagesFakeMasked")
        # saveLocation = "D:/ImagesFakeMasked/" + str(scanOriginIndex) 
        if not os.path.exists(saveLocation):
            os.mkdir(saveLocation)
        if not os.path.exists(saveLocation + '/Distance'):
            os.mkdir(saveLocation + '/Distance')
        # if not os.path.exists(saveLocation + '/DistanceReversed'):
        #     os.mkdir(saveLocation + '/DistanceReversed')
        if not os.path.exists(saveLocation + '/Intensity'):
           os.mkdir(saveLocation + '/Intensity')
        if not os.path.exists(saveLocation + '/Label'):  
            os.mkdir(saveLocation + '/Label')
        if not os.path.exists(saveLocation + '/Bright'):
            os.mkdir(saveLocation + '/Bright')
        if not os.path.exists(saveLocation + '/BrightFiltered'):
            os.mkdir(saveLocation + '/BrightFiltered')
        if not os.path.exists(saveLocation + '/RGBPoint'):
            os.mkdir(saveLocation + '/RGBPoint')
        if not os.path.exists(saveLocation + '/Numpy'):
            os.mkdir(saveLocation + '/Numpy')

        # imageRGBA = imageRGBA.astype('uint8')
        # imageRGBAGT = imageRGBAGT.astype('uint8')
            #Save the np files
        fakeNum = modification
        np.save(saveLocation + "/Numpy/depth_" + str(fakeNum) + ".npy",imageDepth.astype(np.float16))
        np.save(saveLocation + "/Numpy/intensity_" + str(fakeNum) + ".npy",imageIntensity.astype(np.float16))
        np.save(saveLocation + "/Numpy/label_" + str(fakeNum) + ".npy",imageLabel.astype(int))
        np.save(saveLocation + "/Numpy/mask_" + str(fakeNum) + ".npy",obfuscationMask)
        np.save(saveLocation + "/Numpy/GTdepth_" + str(fakeNum) + ".npy",imageDepthGT.astype(np.float16))
        np.save(saveLocation + "/Numpy/GTintensity_" + str(fakeNum) + ".npy",imageIntensityGT.astype(np.float16))
        np.save(saveLocation + "/Numpy/GTlabel_" + str(fakeNum) + ".npy",imageLabelGT.astype(int))
        np.save(saveLocation + "/Numpy/GTmask_" + str(fakeNum) + ".npy",obfuscationMaskGT)


        print(imageRGBA.shape)
        imRGB = Image.fromarray(imageRGBA, 'RGB')
        imRGB.save(saveLocation +"/RGBPoint/" + str(fakeNum) + ".png","PNG")

        # imL = Image.fromarray(imageLabel)
        # imL = imL.convert("L")
        # imL.save(saveLocation +"/Label/" + str(fakeNum) + ".png", "PNG")

        imBright = np.zeros([rowMax,colMax,3])
        for label in range(20):
            imBright[imageLabel==label] = colorArray[label]
        imBright = imBright * 255
        imBright = imBright.astype('uint8')
        imColoured = Image.fromarray(imBright, 'RGB')
        imColoured.save(saveLocation +"/Bright/" + str(fakeNum) + ".png","PNG")


        imRGB = Image.fromarray(imageRGBAGT, 'RGB')
        imRGB.save(saveLocation +"/RGBPoint/GT" + str(fakeNum) + ".png","PNG")

        imBright = np.zeros([rowMax,colMax,3])
        for label in range(20):
            imBright[imageLabelGT==label] = colorArray[label]
        imBright = imBright * 255
        imBright = imBright.astype('uint8')
        imColoured = Image.fromarray(imBright, 'RGB')
        imColoured.save(saveLocation +"/Bright/GT" + str(fakeNum) + ".png","PNG")

        imageLabel[obfuscationMask] = 0
        imageLabelGT[obfuscationMaskGT] = 0

        imBright = np.zeros([rowMax,colMax,3])
        for label in range(20):
            imBright[imageLabel==label] = colorArray[label]
        imBright = imBright * 255
        imBright = imBright.astype('uint8')
        imColoured = Image.fromarray(imBright, 'RGB')
        imColoured.save(saveLocation +"/BrightFiltered/" + str(fakeNum) + ".png","PNG")

        imBright = np.zeros([rowMax,colMax,3])
        for label in range(20):
            imBright[imageLabelGT==label] = colorArray[label]
        imBright = imBright * 255
        imBright = imBright.astype('uint8')
        imColoured = Image.fromarray(imBright, 'RGB')
        imColoured.save(saveLocation +"/BrightFiltered/GT" + str(fakeNum) + ".png","PNG")

        imageIntensity[obfuscationMask] = 0
        imageIntensityGT[obfuscationMaskGT] = 0
        # imageDepth = imageDepth/maxRange*255
        imageIntensity = imageIntensity/maxI*255
        # imageDepthGT = imageDepth/maxRange*255
        imageIntensityGT = imageIntensityGT/maxI*255
        # im = Image.fromarray(imageDepth)
        # im = im.convert("L")
        # im.save(saveLocation + "/Distance/" + str(fakeNum) + ".png")

        imI = Image.fromarray(imageIntensity)
        imI = imI.convert("L")
        imI.save(saveLocation +"/Intensity/" + str(fakeNum) + ".png")

        # im = Image.fromarray(imageDepthGT)
        # im = im.convert("L")
        # im.save(saveLocation + "/Distance/GT" + str(fakeNum) + ".png")

        imI = Image.fromarray(imageIntensityGT)
        imI = imI.convert("L")
        imI.save(saveLocation +"/Intensity/GT" + str(fakeNum) + ".png")
        b = datetime.datetime.now()
        print("time taken for this image:")
        print(b-a)

time.sleep(86400)