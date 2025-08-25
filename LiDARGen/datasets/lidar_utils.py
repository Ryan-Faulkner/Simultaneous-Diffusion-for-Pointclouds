
import numpy as np
import random
from scipy.sparse import coo_matrix
from PIL import Image
import math

import datetime



def load_matrices(kitti_path, data_name):
    cam_to_velo_path = kitti_path + '/calibration/calib_cam_to_velo.txt'
    cam_to_velo = np.identity(4)
    cam_to_velo[0:3, :] =  np.loadtxt(cam_to_velo_path, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)).reshape(3, 4)
    velo_to_cam = np.linalg.inv(cam_to_velo)

    calib_cam_to_pose_path = kitti_path + '/calibration/calib_cam_to_pose.txt'
    calib_cam_to_pose = np.identity(4)
    calib_cam_to_pose[0:3, :] = np.loadtxt(calib_cam_to_pose_path, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))[0].reshape(3, 4)

    poses_path = kitti_path + '/data_poses/' + data_name + '/poses.txt'
    poses_loaded = np.loadtxt(poses_path, usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)).reshape(-1, 3, 4)

    poses = np.identity(4)
    poses = poses[np.newaxis, :, :]
    poses = np.repeat(poses, poses_loaded.shape[0], axis=0)
    poses[:,0:3,:] = poses_loaded

    velo_to_pose = calib_cam_to_pose @ velo_to_cam

    return velo_to_pose, poses




def point_cloud_to_range_image_OG(point_cloud, isMatrix, return_remission = False, return_points=False):
    if (isMatrix):
        laser_scan = LaserScan()
        laser_scan.set_points(point_cloud)
        laser_scan.do_range_projection()
    else:
        laser_scan = LaserScan()
        laser_scan.open_scan(point_cloud)
        laser_scan.do_range_projection()

    if return_points:
        return laser_scan.proj_xyz, laser_scan.proj_range, laser_scan.proj_remission
    elif return_remission:
        return laser_scan.proj_range, laser_scan.proj_remission
    else:
        return laser_scan.proj_range

def point_cloud_to_range_image(point_cloud, origin, return_remission = False, return_points=False, provided_origin = False, rowMax = 64, colMax = 1024, saveNum = 0):
    # origin = origin * 10
    print(len(point_cloud))
    # print(origin)
    # print(np.median(point_cloud, 0))
    b = datetime.datetime.now()
    intensity = None
    colour = None
    if(return_remission):
        intensity = point_cloud[:,3]
        colour = point_cloud[:,4:7]
        point_cloud = point_cloud[:,:3]
    else:
        # intensity = point_cloud[:,3]
        colour = point_cloud[:,3:7]
        point_cloud = point_cloud[:,:3]
    #LiDAR Specs
    # horizontalScope = 360
    # verticalScope = 80 #+40 down to -40
    #my training LiDAR modified specs
    #I want scope of 360 for saving
    # colMax = colMax * 2

    # colMax = 225
    # rowMax = 100

    # Maptek Specs
    # horizontalScope = 180 * 2
    # verticalScope = 60 #+5 down to -40
    
    PointCloudIndices = np.arange(len(point_cloud))
    oldIndices = np.zeros(len(point_cloud)) - 1
    imagePointCloudIndices = np.zeros([rowMax,colMax]) - 1
    # verticalScope = 80
    # colMax = 1800
    # rowMax = 800 #+5 down to -40
    #KITTI Specs:
    horizontalScope = 360
    # verticalScope = 26.8 #Kitti specifically assumes +2 from origin down to -24.8 below the scanner location.
    # verticalPositive = 2
    #LIDARGEN's Incorrect/Unprecise KITTI specs
    verticalScope = 28 #Kitti specifically assumes +2 from origin down to -24.8 below the scanner location.
    verticalPositive = 3
    # rowMax = 100#scan.row_count
    # colMax = 224#scan.column_count

    horizontalAngles = math.radians(horizontalScope) / colMax #6 minutes of an arc for a 800x3600 image, 12 minutes for 800x1800 (assumign 360 coverage)
    verticalAngles = math.radians(verticalScope) / rowMax #6 arc minutes (one tenth of a degree). This means every ten pixels is one degree, 80 pixels = +- 40 degrees
    # verticalAngles = 0.00145444 + 0.00028235549 # 5 minutes and 58 seconds
    rowCount = rowMax#scan.row_count
    colCount = colMax#scan.column_count
    #If centre of image is a perfect 0,0 origin, then Xmin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
    horizontalMin = colCount//(-2) * (horizontalAngles) + (horizontalAngles)/2
    #LiDAR, with 0,0 in centre:
    # verticalMin = rowCount//(-2) * verticalAngles + (verticalAngles/2)
    #My LiDAR training version, reducing useless sky pixels by cutting top thirty degrees off
    # verticalMin = rowCount*3//(-4) * verticalAngles + (verticalAngles/2)
    #KITTI, with 0,0 5 pixels from the top:
    verticalMin = (rowCount-5)* -1 * verticalAngles + (verticalAngles/2)
    #A smarter min caluclation
    verticalMin = math.radians(verticalPositive - (verticalScope))

    #get point cloud origin
    trueOrigin = np.median(point_cloud,0)

    # print("compare origins")
    # print(trueOrigin)
    # print(origin)
    # origin = trueOrigin


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
    imageXY = np.zeros([rowMax,colMax]) + maxRange
    imageDepth = np.zeros([rowMax,colMax]) + maxRange
    imageIntensity = np.zeros([rowMax,colMax])
    imageColour = np.zeros([rowMax,colMax,3])
    obfuscationMask = np.zeros([rowMax,colMax]).astype(bool)

    tooFewPoints = True
    #make fake origin, for now... just pick a random point to use as the origin. After all a point by definition, has to be on the surface....
    #Then shift it 1 metre towards XY centre to prevent it being inside of a mine wall or something.
    #This totally can't end poorly /s
    # modification = np.array([random.uniform(-10,10),random.uniform(-10,10),random.uniform(0,5)])
    fakeOrigin = origin# + modification
    #shift 1 metre to "true centre"
    # distanceToTrue = (randomPoint-trueOrigin)
    # fakeOrigin = fakeOrigin + (distanceToTrue/np.sqrt(np.sum(np.square(distanceToTrue))))

    relativePoints = point_cloud - fakeOrigin
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
    newGrid = np.logical_and(inGrid, oldIndices == -1)
    #Second method likes to just ignore points that don't get used
    # newIndices = np.unique(PointCloudIndices[newGrid], return_inverse = True)[-1]
    # oldIndices[newGrid] = newIndices
    #However Densification requires the actual index relative to entire input point cloud 
    oldIndices[newGrid] = PointCloudIndices[newGrid]
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

    imagePointCloudIndices[tempDepth != 0] = coo_matrix((oldIndices[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
 
    if(return_remission):
        imageIntensity[tempDepth != 0] = coo_matrix((intensity[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
        # imageColour[tempDepth != 0] = coo_matrix((colour[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax,3)).toarray()[tempDepth != 0]
    # imageRGBA[tempDepth != 0,0] = coo_matrix((scanOG.point_colours[final_ind,0].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageRGBA[tempDepth != 0,1] = coo_matrix((scanOG.point_colours[final_ind,1].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageRGBA[tempDepth != 0,2] = coo_matrix((scanOG.point_colours[final_ind,2].astype('uint8'), (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0]
    # imageLabel[tempDepth != 0] = coo_matrix((labels[final_ind], (newRow[reduced_ind], newCol[reduced_ind])), shape=(rowMax,colMax)).toarray()[tempDepth != 0] + 2   

    #Now flip so image is as I expect it to be
    imageDepth = np.flip(imageDepth)
    imageIntensity = np.flip(imageIntensity)
    # imageColour = np.flip(imageColour)
    # imageRGBA = np.flip(imageRGBA,axis=(0,1))
    # imageLabel = np.flip(imageLabel)
    imageXY = np.flip(imageXY)
    #FLIP EVERYTHING ELSE TOO YOU PROFESSIONAL IDIOT
    #well it is just this one which I had previously forgotten to flip, but still
    #have to make it a copy or it yells at me because torch doesn't like negative strides (from flipping view) :(
    imagePointCloudIndices = np.flip(imagePointCloudIndices).copy()

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

    # Without Sky
    skyMask[:] = False
    if(np.sum(skyMask != 1) < minPixels):
        print("too much sky")
        print(np.sum(skyMask != 1))
    # else:
    #     print("has " + str(np.sum(skyMask != 1) * 100 // (minPixels*6)) + " existence")
    obfuscationMask[-1,:] = imageXY[-1,:] > minDepth+5
    # With Sky:
    # imageDepth[skyMask] = maxRange
    
    # maskSum = np.sum(skyMask,0)
    # skyline = int(np.median(maskSum[skyMask[-2,:] != 1]))
    # #temp remove this too
    # obfuscationMask[skyline:, maskSum > skyline] = np.logical_or(obfuscationMask[skyline:, maskSum > skyline], skyMask[skyline:, maskSum > skyline]) 
    # obfuscationMask = np.logical_or(obfuscationMask, np.expand_dims(skyMask[-2,:],0))
    
    tooFewPoints = False
    # im = Image.fromarray(imageDepth/maxRange*255)
    # im = im.convert("L")
    # im.save("sanitycheck/" + str(int(randomPoint)) + ".png")
    # imageLabel[skyMask] = 1
    # imageLabel[obfuscationMask] = 0
    # return imageDepth,imageLabel,obfuscationMask
    c = datetime.datetime.now()
    # print("time taken")
    # intensitytoSave = ((np.log2(imageIntensity+1)) / 12)# * obfuscationMask
    # intensitytoSave = imageIntensity * obfuscationMask
    # intensitytoSave = np.where(intensitytoSave>=1500, 0, intensitytoSave) + 0.0001
    # print("max is")
    # print(np.max(intensitytoSave))
    
    # obfuscationMask = np.where(imageDepth>=2056, 0, obfuscationMask)
    # # # intensitytoSave = imageDepth / 1500 * obfuscationMask
    # intensitytoSave = ((np.log2(imageDepth+1)) / 11) * np.logical_not(obfuscationMask)
    # # intensitytoSave = ((np.log2(imageDepth+1)))# * obfuscationMask
    # # intensitytoSave = ((np.log2(intensitytoSave+1)))# * obfuscationMask
    # intensitytoSave = np.clip(intensitytoSave, 0, 1.0)
    # img = Image.fromarray(np.uint8(intensitytoSave*255), 'L')
    # img.save("PreGenImages4/" + str(saveNum) + '.png')
    # saveNum = saveNum + 1
    
    # if(saveNum < 20002):
    #     if(saveNum > 0):
    #         if(np.sum(skyMask != 1) > minPixels):
    #             np.save("/data/PreGenImages/Depth/" + str(saveNum),imageDepth)
    #             np.save("/data/PreGenImages/Intensity/" + str(saveNum),obfuscationMask)
    #             np.save("/data/PreGenImages/Mask/" + str(saveNum),obfuscationMask)
    #             saveNum = saveNum + 1
    # print(c-b)
    # print(np.sum(imageDepth < 2056))
    if(return_remission):
        return imageDepth,imageIntensity, obfuscationMask, saveNum, skyMask, imagePointCloudIndices
    return imageDepth,obfuscationMask, saveNum, skyMask, imagePointCloudIndices



'''
    Class taken fom semantic-kitti-api project.  https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/laserscan.py
'''
class LaserScan:
    """Class that contains LaserScan with x,y,z,r"""
    EXTENSIONS_SCAN = ['.bin']

    def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0):
        self.project = project
        self.proj_H = H
        self.proj_W = W
        self.proj_fov_up = fov_up
        self.proj_fov_down = fov_down
        self.reset()

    def reset(self):
        """ Reset scan members. """
        self.points = np.zeros(
            (0, 3), dtype=np.float32)        # [m, 3]: x, y, z
        self.remissions = np.zeros(
            (0, 1), dtype=np.float32)    # [m ,1]: remission

        # projected range image - [H,W] range (-1 is no data)
        self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

        # unprojected range (list of depths for each point)
        self.unproj_range = np.zeros((0, 1), dtype=np.float32)

        # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                                dtype=np.float32)

        # projected remission - [H,W] intensity (-1 is no data)
        self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                      dtype=np.float32)

        # projected index (for each pixel, what I am in the pointcloud)
        # [H,W] index (-1 is no data)
        self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                                dtype=np.int32)

        # for each point, where it is in the range image
        self.proj_x = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: x
        self.proj_y = np.zeros((0, 1), dtype=np.float32)        # [m, 1]: y

        # mask containing for each pixel, if it contains a point or not
        self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                                  dtype=np.int32)       # [H,W] mask

    def size(self):
        """ Return the size of the point cloud. """
        return self.points.shape[0]

    def __len__(self):
        return self.size()

    def open_scan(self, filename):
        """ Open raw scan and fill in attributes
        """
        # reset just in case there was an open structure
        self.reset()

        # check filename is string
        if not isinstance(filename, str):
            raise TypeError("Filename should be string type, "
                            "but was {type}".format(type=str(type(filename))))

        # check extension is a laserscan
        if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
            raise RuntimeError("Filename extension is not valid scan file.")

        # if all goes well, open pointcloud
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]    # get xyz
        remissions = scan[:, 3]  # get remission
        self.set_points(points, remissions)

    def set_points(self, points, remissions=None):
        """ Set scan attributes (instead of opening from file)
        """
        # reset just in case there was an open structure
        self.reset()

        # check scan makes sense
        if not isinstance(points, np.ndarray):
            raise TypeError("Scan should be numpy array")

        # check remission makes sense
        if remissions is not None and not isinstance(remissions, np.ndarray):
            raise TypeError("Remissions should be numpy array")

        # put in attribute
        self.points = points    # get xyz
        if remissions is not None:
            self.remissions = remissions  # get remission
        else:
            self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

        # if projection is wanted, then do it and fill in the structure
        if self.project:
            self.do_range_projection()

    def do_range_projection(self):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(self.points, 2, axis=1)

        # get scan components
        scan_x = self.points[:, 0]
        scan_y = self.points[:, 1]
        scan_z = self.points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W                              # in [0.0, W]
        proj_y *= self.proj_H                              # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
        self.proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
        self.proj_y = np.copy(proj_y)  # stope a copy in original order

        # copy of depth in original order
        self.unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        remission = self.remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # assing to images
        self.proj_range[proj_y, proj_x] = depth
        self.proj_xyz[proj_y, proj_x] = points
        self.proj_remission[proj_y, proj_x] = remission
        self.proj_idx[proj_y, proj_x] = indices
        self.proj_mask = (self.proj_idx > 0).astype(np.float32)
