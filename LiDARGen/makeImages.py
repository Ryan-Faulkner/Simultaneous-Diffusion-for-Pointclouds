from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import os
from glob import glob
import h5py
from scipy.sparse import coo_matrix


full_list = glob(os.path.join('HDVMineData', 'Penrice*.h5'))
pointArray = [] 
for file in full_list:
    openedFile = h5py.File(file, 'r')
    pointArray.append(openedFile['Input'][:,:3])
    # LabelArray.append(openedFile['Input'])
# print(pointArray[0].shape)
# x = a/0
point_cloud = np.concatenate(pointArray,axis=0)
idx = np.arange(len(point_cloud)).astype(int)
print("total points")
print(len(point_cloud))
np.random.shuffle(idx)
rowModifier = 8
columnModifier = 4
horizontalAngleLimiter = 2
rowMax = int(800/rowModifier)#scan.row_count
colMax = int(1800/columnModifier/horizontalAngleLimiter/2)*2#scan.column_count
maxRange = 2057.701 #Current record set by Penrice
minPixels = rowMax*colMax/6

# if(scan.max_range > maxRange):
    # maxRange = scan.max_range
# if(len(scan.point_attributes['ClassFuckYou2'] != 0) < 10000): #if less than 10K points labelled, skip
#     print("Skipped due to not labelled yet")
#     continue
# print(rowMax)
# print(colMax)


# imageRGBA = np.zeros([rowMax,colMax,3]).astype('uint8') #Have to add 1 as rowMax is the highest row number starting at 0.
# image = np.zeros([rowMax+1,colMax+1]) #Have to add 1 as rowMax is the highest row number starting at 0.
# imageIntensity = np.zeros([rowMax,colMax]) #Have to add 1 as rowMax is the highest row number starting at 0.
# imageLabel = np.zeros([rowMax,colMax])

horizontalAngles = 0.00174533 * 2 * columnModifier #6 minutes of an arc for a 800x3600 image, 12 minutes for 800x1800 (assumign 360 coverage)
verticalAngles = 0.00174533 * rowModifier #6 minutes of an arc
# verticalAngles = 0.00145444 + 0.00028235549 # 5 minutes and 58 seconds
rowCount = rowMax#scan.row_count
colCount = colMax#scan.column_count

#If not even a fifth of the pixels have a legitimate value... try again. 


#If centre of image is a perfect 0,0 origin, then Xmin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
horizontalMin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
verticalMin = rowCount//(-2) * verticalAngles + verticalAngles/2

#get point cloud origin
trueOrigin = np.max(point_cloud) - np.min(point_cloud)

tooFewPoints = True
memoryLimit = 300000
memoryCount = 0
for randomPoint in idx:
    if(memoryCount > memoryLimit):
        break
    imageXY = np.zeros([rowMax,colMax]) + maxRange
    imageDepth = np.zeros([rowMax,colMax]) + maxRange
    obfuscationMask = np.zeros([rowMax,colMax]).astype(bool)
    #make fake origin, for now... just pick a random point to use as the origin. After all a point by definition, has to be on the surface....
    #Then shift it 1 metre towards XY centre to prevent it being inside of a mine wall or something.
    #This totally can't end poorly /s
    fakeOrigin = point_cloud[randomPoint]
    #shift 1 metre to "true centre"
    distanceToTrue = (randomPoint-trueOrigin)
    fakeOrigin = fakeOrigin + (distanceToTrue/np.sqrt(np.sum(np.square(distanceToTrue))))

    relativePoints = point_cloud - fakeOrigin
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
    # if(len(np.nonzero(inGrid)[0]) == 0):
    #     continue
    # if(not groundTruth):
    #     # isBetter = imageDepthGT[modification,-1-newRow[inGrid],-newCol[inGrid]-1] > newDepth[inGrid]
    # # else:
    #     isBetter = imageDepth[newRow[inGrid],newCol[inGrid]] > newDepth[inGrid]
    #     inGrid[inGrid] = isBetter
    if(len(np.nonzero(inGrid)[0]) < minPixels):
        # print("not enough pixels")False
        print("too few pixels")
        print(len(np.nonzero(inGrid)[0]))
        continue

    new_ind = np.argsort(newDepth[inGrid])
    newRow = newRow[inGrid][new_ind]
    newCol = newCol[inGrid][new_ind]
    merged = np.stack((newRow,newCol))
    reduced_ind = np.unique(merged, return_index = True, axis=1)[-1]
    final_ind = np.arange(len(newDepth))[inGrid][new_ind][reduced_ind]

    tempDepth = coo_matrix((newDepth[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()
    imageDepth[tempDepth != 0] = tempDepth[tempDepth != 0]
    imageXY[tempDepth != 0] = coo_matrix((xy[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageIntensity[tempDepth != 0] = coo_matrix((scan.point_intensity[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageRGBA[tempDepth != 0,0] = coo_matrix((scanOG.point_colours[final_ind,0].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageRGBA[tempDepth != 0,1] = coo_matrix((scanOG.point_colours[final_ind,1].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageRGBA[tempDepth != 0,2] = coo_matrix((scanOG.point_colours[final_ind,2].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageLabel[tempDepth != 0] = coo_matrix((labels[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0] + 2   

    #Now flip so image is as I expect it to be
    imageDepth = np.flip(imageDepth)
    # imageIntensity = np.flip(imageIntensity)
    # imageRGBA = np.flip(imageRGBA,axis=(0,1))
    # imageLabel = np.flip(imageLabel)
    imageXY = np.flip(imageXY)
    #Now filter the images
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
    if(np.sum(skyMask != 1) < minPixels):
        print("too much sky")
        continue
    obfuscationMask[-1,:] = imageXY[-1,:] > minDepth+5
    imageDepth[skyMask] = maxRange
    
    maskSum = np.sum(skyMask,0)
    skyline = int(np.median(maskSum[skyMask[-2,:] != 1]))
    obfuscationMask[skyline:, maskSum > skyline] = np.logical_or(obfuscationMask[skyline:, maskSum > skyline], skyMask[skyline:, maskSum > skyline]) 
    obfuscationMask = np.logical_or(obfuscationMask, np.expand_dims(skyMask[-2,:],0))
    tooFewPoints = False
    np.save("PreGenImages/Depth/" + str(randomPoint),imageDepth)
    np.save("PreGenImages/Mask/" + str(randomPoint),obfuscationMask)
    memoryCount = memoryCount+1
    # if(randomPoint == idx[0] or randomPoint == idx[1] or randomPoint == idx[2] or randomPoint == idx[3] or randomPoint == idx[4] or randomPoint == idx[5]):
    #     im = Image.fromarray(imageDepth/maxRange*255)
    #     im = im.convert("L")
    #     im.save("sanitycheck/" + str(int(randomPoint)) + ".png")
# imageLabel[skyMask] = 1
# imageLabel[obfuscationMask] = 0
# return imageDepth,imageLabel,obfuscationMask
c = datetime.datetime.now()
print("time taken")
print(c-b)