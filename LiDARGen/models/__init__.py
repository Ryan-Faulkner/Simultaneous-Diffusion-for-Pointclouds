import torch
import numpy as np
import math

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

@torch.no_grad()
def anneal_Langevin_dynamics_densification(x_mod, refer_image, scorenet, sigmas,
                                        n_steps_each=100, step_lr=0.000008, denoise = True, verbose = True, grad_ref = 0.1, sampling_step = 16):
    images = []
    targets = []
    mask = torch.zeros_like(x_mod, device = x_mod.device)
    raw = refer_image[:, :, 0:64:sampling_step, :]
    raw_interp = torch.nn.functional.interpolate(raw, size=(64, 1024), mode='bilinear')
    mask[:, :, 0:64:sampling_step, :] = 1
    step_refer = grad_ref # 0.001

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)
                grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                grad_likelihood_norm = torch.norm(grad_likelihood.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + step_refer * grad_likelihood + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                images.append(x_mod.to('cpu'))
            if verbose and c % 20 == 0:
                print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))
                print("level: {}, step_size: {}, grad_norm: {}, grad_likelihood_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), grad_likelihood_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise) + step_refer * grad_likelihood
            images.append(x_mod.to('cpu'))
            print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))

        grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)
        x_mod = x_mod + step_refer * grad_likelihood
        images.append(x_mod.to('cpu'))
        print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))

        #images.append(refer_image.to('cpu'))
        targets.append(refer_image.to('cpu'))
        return images, targets

@torch.no_grad()
def anneal_Langevin_dynamics_inpainting_simultaneous_basic(x_mod, refer_image, refer_mask, sky, x_indices, minStepToShare,setting, scorenet, sigmas, modificationList, actualBatchSize,
                                        n_steps_each=100, step_lr=0.000008, existMask = None, denoise = True, verbose = True, grad_ref = 0.1, correlation_coefficient = 0.1, sampling_step = 16):
    #Settings variants
    #setting = 1 : min
    #setting = 2 : average
    #setting = 3 : min (with origin modification)
    #Setting = 4: second method only setting
    #Setting = 5&6: increase correlation_coefficient as nears completion.
    #Setting = 7: Controlled Average
    images = []
    targets = []
    sharedImages = []
    mask = refer_mask
    raw = refer_image
    sky = sky.to(x_mod.device)
    # raw_interp = torch.nn.functional.interpolate(raw, size=(64, 1024), mode='bilinear')
    # mask[:, :, 0:64:sampling_step, :] = 1
    step_refer = grad_ref# 0.001
    #for now make it the same as co-efficient to base input
    # correlation_coefficient = grad_ref
    rowMax = x_mod.shape[-2]
    colMax = x_mod.shape[-1]
    batchSize = x_mod.shape[0]
    rowCount = rowMax
    colCount = colMax
    maxRange = 2057.701 #Current record set by Penrice
    minPixels = rowMax*colMax/6
    horizontalScopeMin = -180
    horizontalScopeMax = 180
    horizontalScope = horizontalScopeMax-horizontalScopeMin
    #LiDAR
    # verticalScopeMax = 15
    # verticalScopeMin = -45
    # #KITTI
    # verticalScopeMax = 2
    # verticalScopeMin = -24.8
    #LIDARGEN's Incorrect/Unprecise KITTI specs
    verticalScopeMax = 3 #Kitti specifically assumes +2 from origin down to -24.8 below the scanner location.
    verticalScopeMin = -25

    verticalScope = verticalScopeMax-verticalScopeMin #+15 down to -45 for LiDAR, +2 to -24.8 for KITTI
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #vertical SCOOOOPE
    #it goes +5 to -45
    #so when an angle is negative?
    #it ends up as +40
    #which obviously I DO NOT HAVE
    #fuck
    #solution?
    #I need to create for the grid -40 to +40
    #then crop just the part I actually want AFTER I've done the isNeg part.
    #This means I need two rowMax/ColMax values. One for the grid I'm using (Input), one for the I'm making (Output)
    #Angles are the same, everything other than imageDepth is the same.
    #fuuuuuuck me
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    horizontalAngles = math.radians(horizontalScope) / colMax #6 minutes of an arc for a 800x3600 image, 12 minutes for 800x1800 (assumign 360 coverage)
    verticalAngles = math.radians(verticalScope) / rowMax #6 arc minutes (one tenth of a degree). This means every ten pixels is one degree, 80 pixels = +- 40 degrees
    horizontalMin = ((colCount*horizontalScopeMin)//horizontalScope) * (horizontalAngles) + (horizontalAngles)/2

    bigRowCount = int((np.max((np.absolute(verticalScopeMin),np.absolute(verticalScopeMax)))*2)*rowCount//verticalScope)
    # print("the row")
    # print(bigRowCount)
    bigRowMin = ((bigRowCount)//-2) * verticalAngles + (verticalAngles/2)
    #ok so this *3//4 is because it ends with 45 isntead of 30. So instead of min being -30, for scope -30 -> 30, it is -45 to 15
    #specifically I have 120 pixels below, 40 above.
    #so I need 120 below, 120 above
    #for 240 total
    #I need it to be -45 to 45 for the big version

    verticalMin = ((rowCount*verticalScopeMin)//verticalScope) * verticalAngles + (verticalAngles/2)
    #divide origins by 2000 so in same space as my distances
    # modificationList = np.array([[0,0,0],
    #             [10,0,0],
    #             [0,10,0],
    #             [10,10,0],
    #             [0,0,10],
    #             [-10,0,0],
    #             [0,-10,0],
    #             [-10,-10,0]]) #/ -1
    # modificationList = np.array([[0,0,0],
    #             [0,10,0],
    #             [10,0,0],
    #             [10,10,0],
    #             [0,0,10],
    #             [0,-10,0],
    #             [-10,0,0],
    #             [-10,-10,0]]) / -1
    # modificationList = np.expand_dims(np.expand_dims(modificationList,-1),-1)
    # originListOG = torch.from_numpy(modificationList).to(x_mod.device)
    originListOG = torch.unsqueeze(torch.unsqueeze(modificationList,-1),-1)

    #These need to go in reverse because for (reasons??) the raw data was originally upside down so the XYZ projection requires this, I dunno it's 2am right now spherical/cartesian is pain
    azimuth = torch.from_numpy(np.reshape(((np.arange(colMax-1,-1,-1) * horizontalAngles) + horizontalMin),(1,colCount))).to(x_mod.device)
    elevation = torch.from_numpy(np.reshape(((np.arange(rowMax-1,-1,-1) * verticalAngles) + verticalMin),(rowCount,1))).to(x_mod.device)
    # OGCorrelationCoefficient = correlation_coefficient

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            if(setting == 5):
                correlation_coefficient = 1 / (len(sigmas) / (c+1))
            if(setting == 6):
                correlation_coefficient = 0.5 / (len(sigmas) / (c+1))
        # for c, sigma in enumerate(sigmas[:3]):
            heWhoModsTheModMan = 1
            #No longer doing this, now doing sigma Mod
            
            # if(setting >= 3 and sigma > 1):
            #     heWhoModsTheModMan = sigma
            sigmaMod = 1
            if(sigma > 1):
                sigmaMod = sigma
            #No logging, it fucks shit up
            originList = ((torch.log2(torch.abs(originListOG)+1)) / 6) * heWhoModsTheModMan
            originList = (torch.pow(2,(originList*6))-1)

            #Ok so the issue here is that my origins are all positive right
            # print("THE NEW ORIGINS")
            # print(originList)
            #No need for this unstil I'm logging
            originList = originList/(originListOG+0.00000001) * 10

            # originList = ((((originListOG))) / 2000) * heWhoModsTheModMan
            # originList = (originList*2000)

            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)
                grad = torch.nan_to_num(grad)
                # used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                # expectedNoise = torch.randn_like(X) * used_sigmas
                # #X = X + noise * mask

                # grad_likelihood = -mask * (x_mod - (refer_image+expectedNoise)) # - 0.05*(1-mask)*(x_mod - raw_interp)
                grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                grad_likelihood_norm = torch.norm(grad_likelihood.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                # print(torch.min(grad))
                # print(torch.max(grad))
                # print(torch.min(grad_likelihood))
                # print(torch.min(correlation_correction))

                # x_mod = x_mod + step_size * grad + step_refer * grad_likelihood + correlation_coefficient * correlation_correction + noise * np.sqrt(step_size * 2)
                x_mod = x_mod + step_size * grad + step_refer * grad_likelihood + noise * np.sqrt(step_size * 2)

                #get the shared image version
                #But only if it is after min step
                if(c >= minStepToShare):
                    isNeg = (x_mod[:,0,:,:] < 0).int()
                    tooHigh = torch.max(torch.abs(x_mod[:,0,:,:]))*6/sigmaMod > 50
                    # print(torch.max(torch.abs(x_mod[:,0,:,:]))*6)
                    #ok so this needs to handle negatives as well
                    modifierArray = torch.ones_like(x_mod[:,0,:,:]) - (isNeg*2)
                    realDistance = (torch.pow(2,(torch.abs(x_mod[:,0,:,:])*6/sigmaMod))-1) * modifierArray

                    # realDistance = x_mod[:,0,:,:]*2000
                    # isNeg = (realDistance < 0).int()
                    #for now just outright skip this when distances too large to overlap
                    #smart
                    # print("sanity check origins")
                    # print(originList.shape)
                    # print(originList[:batchSize,0])
                    # print(originList[:batchSize,:,0])
                    relativePointsStack = []
                    # print(bigCloud.shape)
                    # print(bigCloud[:,:,0].shape)
                    # print(fromWorld.shape)
                    for currentMegaBatch in range(int(batchSize/actualBatchSize)):  
                        pointX = torch.add(realDistance[(actualBatchSize*(currentMegaBatch)):(actualBatchSize*(currentMegaBatch+1))] * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:actualBatchSize,0])
                        pointY = torch.add(realDistance[(actualBatchSize*(currentMegaBatch)):(actualBatchSize*(currentMegaBatch+1))] * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:actualBatchSize,1])
                        pointZ = torch.add(realDistance[(actualBatchSize*(currentMegaBatch)):(actualBatchSize*(currentMegaBatch+1))] * torch.sin(elevation).view(1,-1,1),originList[:actualBatchSize,2])
                        sharedCloud = torch.unsqueeze(torch.stack((pointX,pointY,pointZ,x_mod[(actualBatchSize*(currentMegaBatch)):(actualBatchSize*(currentMegaBatch+1)),1,:,:])).view(4,-1),0)
                        #now I need to get the resultant distance for EVERY origin
                        bigCloud = torch.tile(sharedCloud[:,:3],(actualBatchSize,1,1))
                        #the 0 in originlist is necessary to reduce it from Ox3x1x1 to Ox3x1
                        relativePointsStack.append(torch.subtract(bigCloud,originList[:actualBatchSize,:,0]))
                    relativePoints = torch.cat(relativePointsStack,0)
                    # print("POINTS SHAPE")
                    # print(relativePoints.shape)

                    xy = torch.square(relativePoints[:,0]) + torch.square(relativePoints[:,1])
                    newDepth = torch.sqrt(xy + torch.square(relativePoints[:,2]))
                    # print("depths")
                    # print(torch.max(x_mod))
                    # print(torch.min(x_mod))
                    # print(torch.max(realDistance))
                    # print(torch.min(realDistance))
                    # print(torch.max(newDepth))
                    newDepth = ((torch.log2(newDepth+1)) / 6*sigmaMod)
                    # newDepth = newDepth / 2000
                    # print("sanity check")
                    #this passes
                    # print(torch.sum(torch.abs(newDepth)) - torch.sum(torch.abs(x_mod[:,0,:,:])))
                    # #this also passes
                    # print(torch.sum(newDepth * modifierArray.flatten()) - torch.sum(x_mod[:,0,:,:]))
                    # #now the real test
                    # print(torch.sum(newDepth * modifierArray.flatten() - x_mod[:,0,:,:].flatten()))
                    #This also passes because it is the new depth for each pixel
                    #I have the row and column already
                    #If I multiply depth by modifierArray I get the original depth value before I absoluted it
                    #So now I have for that row & column, all the depths as they should be
                    #oh no this fails outside batch=1
                    #because what about for other viewpoints idiot
                    #yeah stick with current method
                    #So we're back to drawing board
                    #on why the fuck I get these white patches of super far points
                    # newDepth = newDepth / 2000
                    # print(torch.max(newDepth))
                    # newDepth = torch.clip(newDepth,min=0)
                    # print(torch.max(newDepth))
                    horizontal = torch.atan2(relativePoints[:,1], relativePoints[:,0])
                    # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
                    xy = torch.sqrt(xy)
                    vertical = torch.atan2(relativePoints[:,2], xy)

                    newCol = torch.round(torch.divide((horizontal-horizontalMin),horizontalAngles)).int()
                    newRow = torch.round(torch.divide((vertical-bigRowMin),verticalAngles)).int()
                    #Ok so here I have the row and column values
                    #I need to flip them to make it work though
                    #
                    #Now I need to unflip it *again* because... reasons. The projection hates me that's why. I don't know why it had to be reversed in the raw data but I pay for that consequence now.
                    newCol = newCol * -1 + colCount -1
                    newRow = newRow * -1 + bigRowCount- 1
                    #Post flip needs to be 0 -> bigRow
                    #I think it should work
                    #but clearly it doesn't
                    #ugh

                    # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.less(newRow,rowCount)))
                    # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.logical_and(torch.less(newRow,rowCount),torch.greater(torch.unsqueeze(realDistance.flatten(),0),0))))
                    # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.less(newRow,rowCount)))

                    isFour = torch.stack((torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.ones(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount)),0)
                    isFour = torch.unsqueeze(isFour,0)
                    isFour = torch.cat((torch.tile(torch.logical_not(isFour),(4,1,1)),torch.ones_like(isFour),torch.tile(torch.logical_not(isFour),(3,1,1))),0).view(8,-1).to(x_mod.device)
                    # print(isFour.shape)
                    inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,-1),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,-1),torch.less(newRow,bigRowCount)))
                    #Do not include sky points when sharing because it fucks shit up
                    #Except right now it fucks me up because
                    # inGrid = torch.logical_and(inGrid,torch.unsqueeze(sky.flatten(),0))
                    gridStack = []
                    for currentMegaBatch in range(int(batchSize/actualBatchSize)):  
                        gridStack.append(torch.logical_and(inGrid[(actualBatchSize*(currentMegaBatch)):(actualBatchSize*(currentMegaBatch+1))],torch.unsqueeze(sky[(actualBatchSize*(currentMegaBatch)):(actualBatchSize*(currentMegaBatch+1))].flatten(),0)))
                    inGrid = torch.cat(gridStack,0)
                    #Now get rid of points which exist inside "sensor failure positions" because the network WILL fuck them up. Without fail.
                    # inGrid = torch.logical_and(inGrid, existMask.flatten())
                    inGrid = torch.logical_and(inGrid, existMask[:actualBatchSize].flatten())
                    #Do not include points where depth is 0
                    #If something is less than 10cm away away from the scanner, assume it is an error and ignore it
                    minDepthToTrust = 0.2
                    minDepthToTrust = ((torch.log2(torch.tensor(minDepthToTrust)+1)) / 6*sigmaMod)
                    inGrid = torch.logical_and(inGrid,newDepth > minDepthToTrust)
                    # print(inGrid.shape)
                    # inGrid = torch.logical_and(isFour,inGrid)
                            
                    del bigCloud
                    newStack = []
                    imageMaskStack = []
                    for origin in range(batchSize):
                        new_ind = torch.argsort(torch.flatten(newDepth[origin][inGrid[origin]]),dim=0, descending = True)
                        # print("depth sort sanity check")
                        # print(torch.flatten(newDepth[origin][inGrid[origin]])[new_ind])
                        # print(torch.max(new_ind))
                        # print("the shapes")
                        # print(torch.min(x_mod))
                        # print(torch.min(pointX))
                        # print(torch.min(pointY))
                        # print(torch.min(pointZ))
                        # print(torch.min(horizontal))
                        # print(torch.min(vertical))
                        #Because pytorch is stupid and sorts before any torch.unique() I need to do the depth sorting AFTER getting unique values
                        #It's very dumb I know
                        #but if I take reduced_ind and sort it by new_ind
                        #then I run unique() on it again to remove duplicates
                        newRowTemp =torch.flatten(newRow[origin][inGrid[origin]])[new_ind]
                        # print(newRowTemp.shape)
                        newColTemp =torch.flatten(newCol[origin][inGrid[origin]])[new_ind]
                        #At this point merged is a row,column list sorted by depth
                        #I need to remove duplicates, but with removal based on depth
                        newRowTemp, theSortProjectionOne = torch.sort(newRowTemp, stable = True)
                        newColTemp = newColTemp[theSortProjectionOne]
                        newColTemp, theSortProjectionTwo = torch.sort(newColTemp, stable = True)
                        newRowTemp = newRowTemp[theSortProjectionTwo]
                        merged = torch.stack((newRowTemp,newColTemp),0)

                        # sortedMerged, theSortProjectionOne  = torch.sort(merged, stable = True, dim = 1)
                        # print(theSortProjectionOne)
                        #the inverse is useless, but because it's sorted, the count can be cumsum'd into what I need
                        _,duplicate_count= torch.unique_consecutive(merged, return_counts=True, dim=1)
                        # print(reduced_ind.shape)
                        reduced_ind = torch.cumsum(duplicate_count,0)-1
                        # print(reduced_ind.shape)

                        #this reduced_ind is for the depth_sorted row and col values
                        #It will take the values in any depth-sorted array, and convert them to a unique row/column sorted array



                        #Because Pytorch is garbage, I have no choice but to sort arrays before using unique on them. It's a huge fucking pain.
                        #This code can be massively cleaned up when pytorch finally gets around to making sorted=False work on CUDA.
                        #But until then, here we are, suffering together :)
                        # reduced_ind, theSortProjectionThree = torch.sort(reduced_ind,stable=True)
                        # print(torch.flatten(newDepth[origin][inGrid[origin]])[new_ind][theSortProjection])


                        final_ind = torch.arange(len(newDepth[0]),device=x_mod.device)
                        
                        gridInd = inGrid[origin]
                        
                        final_ind = final_ind[gridInd]
                        final_ind = final_ind[new_ind]
                        mergedReduced = torch.stack((newRowTemp[reduced_ind],newColTemp[reduced_ind]),0)
                        #ok so after doing this merged is no longer sorte by depth but by row/col
                        #so by definition any unique() function will not remove depth first
                        #except I use reduced_ind as a mask
                        #reduced_ind which is..... also sorted
                        #ughhhhh
                        #I need a fucking mask
                        # print(merged.shape)
                        #And with this we have merged and final_ind, both ready to be used to make new images
                        final_ind = final_ind[theSortProjectionOne][theSortProjectionTwo]

                        #for average across shared point cloud
                        imageDepth = torch.sparse_coo_tensor(merged,torch.flatten(newDepth[origin][final_ind]), size=(bigRowCount,colMax)).to_dense()
                        # print("AAAAAAHHHHHHH")
                        # print(origin // (actualBatchSize))
                        batchProportion = batchSize//actualBatchSize
                        toAddToIntensity = (len(torch.flatten(x_mod[:,1,:,:]))//batchProportion) * (origin // (actualBatchSize))
                        imageIntensity = torch.sparse_coo_tensor(merged,torch.flatten(x_mod[:,1,:,:])[final_ind+toAddToIntensity], size=(bigRowCount,colMax)).to_dense()
                        scaling = torch.sparse_coo_tensor(mergedReduced,duplicate_count, size=(bigRowCount,colMax)).to_dense() + 0.000000001
                        imageDepth = torch.div(imageDepth,scaling)
                        imageIntensity = torch.div(imageIntensity,scaling)
                        
                        if(setting >= 7):

                            # #for minimum
                            final_ind = final_ind[reduced_ind]
                            imageDepthMin = torch.sparse_coo_tensor(mergedReduced,torch.flatten(newDepth[origin])[final_ind], size=(bigRowCount,colMax)).to_dense()
                            imageIntensityMin = torch.sparse_coo_tensor(mergedReduced,torch.flatten(x_mod[:,1,:,:])[final_ind+toAddToIntensity], size=(bigRowCount,colMax)).to_dense()

                            # #For controlled Average
                            #Need to unlog to compare in metres
                            imageDepth = (torch.pow(2,(torch.abs(imageDepth)*6/sigmaMod))-1) #* modifierArray
                            imageDepthMin = (torch.pow(2,(torch.abs(imageDepthMin)*6/sigmaMod))-1)# * modifierArray

                            # # #Remove HUGE differences from fucking shit up
                            # maxDifferenceAllowed = allowance*3
                            # #Just take the closest Depth's intensity if averaging has issues
                            # imageIntensity = torch.where(imageDepth > imageDepthMin + maxDifferenceAllowed, imageIntensityMin, imageIntensity)
                            # #Meanwhile for average, don't let it be more than 10 metres shift
                            # imageDepth = torch.where(imageDepth > imageDepthMin + maxDifferenceAllowed, imageDepthMin, imageDepth)
                            # #Now if there is a 1-2 difference, just shift it a wittle bit. As huge differences have been set to min, they will have no difference now.
                            allowance = 10
                            if(setting >= 8):
                                allowance = 5
                            maxDifferenceAllowed = allowance
                            modOtherwise = allowance / 5
                            #Only let it go 1 metre beyond minimum
                            # maxDifferenceAllowed = ((torch.log2(torch.tensor(maxDifferenceAllowed*4)+1)) / 6*sigmaMod)
                            # modOtherwise = ((torch.log2(torch.tensor(maxDifferenceAllowed)+1)) / 6*sigmaMod)
                            #Just take the closest Depth's intensity if averaging has issues
                            imageIntensity = torch.where(imageDepth > imageDepthMin + maxDifferenceAllowed, imageIntensityMin, imageIntensity)
                            #Meanwhile for average, don't let it be more than 10 metres shift
                            imageDepth = torch.where(imageDepth > imageDepthMin + maxDifferenceAllowed, imageDepthMin + modOtherwise, imageDepth)
                            #And now relog it
                            imageDepth = (torch.log2(imageDepth+1)) / 6*sigmaMod

                        #in case some pixels have rounding errors and end up empty just ignore them
                        imageMask = torch.sparse_coo_tensor(mergedReduced,torch.ones_like(mergedReduced[0], device=x_mod.device), size=(bigRowCount,colMax)).to_dense()
                        
                        #ok now lets get the negs, since during diffusion some pixels will have a negative distance. For these we simply take the value of the opposite pixel and multiply by -1
                        #to keep it simple - if a pixel was negative before the shared pointcloud, it shall be negative after the shared point cloud.
                        # print("is it the neg")
                        # print(isNeg.shape)
                        # print(imageDepth.shape)
                        imageIntensity = torch.add(imageIntensity[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageIntensity,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])
                        imageMask = torch.add(imageMask[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageMask,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])
                        imageDepth = torch.add(imageDepth[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageDepth,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin] * -1)

                        # print(imageDepth.shape)
                        # make sure image Mask does not include dead pixels
                        #Just use same exist mask for all as all data uses same sensor
                        imageMask = torch.logical_and(existMask[0],imageMask)
                        
                        

                        newStack.append(torch.stack((imageDepth,imageIntensity)))
                        imageMaskStack.append(torch.unsqueeze(imageMask.int(),0))
                    newImages = torch.stack(newStack).float()
                    if(c == 0 or c == 20 or c == 110):
                        sharedImages.append(newImages.to('cpu'))
                    if(c == len(sigmas) - 1):
                        images.append(newImages.to('cpu'))
                    maskImages = torch.stack(imageMaskStack).float()
                    # print(torch.min(newImages))
                    #ok now I have my new image and mask
                    # print(newImages.shape)
                    # print(maskImages.shape)
                    #This needs to not overwrite any pixel which has an actual known groundtruth, so multiply by not(mask) as well to only update unknown points
                    #Don't bother updating sky points either
                    maskImages = torch.logical_and(maskImages, sky).int() 
                    correlation_correction = -maskImages * torch.logical_not(mask).int() * torch.sub(x_mod, newImages) # - 0.05*(1-mask)*(x_mod - raw_interp)
                    #if the distances are too large right now, don't even bother with this shit
                    correlation_correction = torch.where(tooHigh, torch.tensor(0, device=x_mod.device).float(), correlation_correction)
                    
                    # print("why do you hurt meeeee")
                    # print(x_mod.shape)
                    # print(newImages.shape)
                    # testVal = x_mod[:,0,:,:] - newImages[:,0,:,:]
                    # print(testVal.shape)
                    # testOne = torch.argmax(testVal)
                    # testTwo = torch.argmin(testVal)
                    # print("ugh")
                    # print(testOne)
                    # print(x_mod[:,0,:,:].flatten()[testOne])
                    # print(newImages[:,0,:,:].flatten()[testOne])
                    # print(newDepth.flatten()[testOne])
                    # print(isNeg.flatten()[testOne])
                    # print(newRow.flatten()[testOne])
                    # predRow = testOne//colCount
                    # print(predRow)
                    # print(newCol.flatten()[testOne])
                    # print(testOne % colCount)


                    # print("yet this one fails??")
                    # print(testTwo)
                    # print(x_mod[:,0,:,:].flatten()[testTwo])
                    # print(newImages[:,0,:,:].flatten()[testTwo])
                    # print(newDepth.flatten()[testTwo])
                    # print(isNeg.flatten()[testTwo])
                    # print(newRow.flatten()[testTwo])
                    # predRow = testTwo//colCount
                    # print(predRow)
                    # print(newCol.flatten()[testTwo])
                    # print(testTwo % colCount)



                    # print("it's probably my roll or flip being slightly off")
                    # #a one pixel difference could account for the issues
                    # #because in correlation correction a negative x_mod minus a positive newImages results in double the absolute value
                    # #It's not that because the newImage itself has the white pixels with super large values
                    # print(torch.max(torch.abs(x_mod[:,0,:,:]))*6)
                    # print(torch.sum(correlation_correction) == 0)
                    # print(torch.sum(torch.abs(x_mod)) - torch.sum(torch.abs(newImages)))
                    # print(torch.sum(torch.abs(x_mod) - torch.abs(newImages)))
                    # print(torch.sum((x_mod - (newImages))))
                    # print(torch.max(torch.abs(newImages)))
                    #somehow this is... extremely high
                    #despite batch size being 1
                    #so in theory it should ALWAYS be 0
                    #shared should always be the only one
                    #Lazy option is just put some code in to set it to 0 when it's stupid high
                    #Good option is work out why it is usually 0 for batch 1 but sometimes 100286.1406 or 3514.5759 etc
                    #It's like it decided every single pixel should just be way off
                    #or maybe one pixel is insanely off
                    #Small error could be explained as when a negative pixel overlaps with a positive causing one of them to differ even with batch size 1
                    #but unless my coefficient 9999999999...
                    #
                    #who fucking knows
                    #It being 0 MOST of the time means the rest of my code is correct
                    #with the generated shared cloud being identical to the OG 
                    x_mod = x_mod + correlation_coefficient * correlation_correction

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                # images.append(x_mod.to('cpu'))
            if verbose and c % 20 == 0:
                print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))
                print("level: {}, step_size: {}, grad_norm: {}, grad_likelihood_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), grad_likelihood_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise) + step_refer * grad_likelihood
            # images.append(x_mod.to('cpu'))
            print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))

        grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)
        x_mod = x_mod + step_refer * grad_likelihood
        images.append(x_mod.to('cpu'))

        return images, targets, sharedImages

@torch.no_grad()
def anneal_Langevin_dynamics_inpainting_simultaneous_second_method(x_mod, refer_image, refer_mask, sky, refer_indices, minStepToShare,setting, scorenet, sigmas, modificationList,
                                        n_steps_each=100, step_lr=0.000008, denoise = True, verbose = True, grad_ref = 0.1, correlation_coefficient = 0.1, sampling_step = 16):

    #This version applies the noise to the 3D cloud, but is otherwise the same.
    #Essentially, instead of 1) +Noise 2) project to 3D, 3) back to 2D it is now 1)Project to 3D, 2)Apply Noise, 3)Back to 2D
    #The hard part is I need to track the point cloud
    #So we are now ALSO bringing in the original indices of the images
    #Settings variants
    #setting = 1 : not included for this method
    #setting = 2 : not included for this method
    #setting = 3 : min (with origin modification)
    #setting = 4 : min (with origin modification, GT hard reset at every timestep)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    images = []
    targets = []
    sharedImages = []
    mask = refer_mask
    raw = refer_image
    sky = sky.to(x_mod.device)
    refer_indices = refer_indices.to(x_mod.device)
    x_indices = refer_indices.clone().detach()
    # raw_interp = torch.nn.functional.interpolate(raw, size=(64, 1024), mode='bilinear')
    # mask[:, :, 0:64:sampling_step, :] = 1
    step_refer = grad_ref# 0.001

    grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)
    x_mod = x_mod + step_refer * grad_likelihood
    #for now make it the same as co-efficient to base input
    # correlation_coefficient = grad_ref
    rowMax = x_mod.shape[-2]
    colMax = x_mod.shape[-1]
    batchSize = x_mod.shape[0]
    rowCount = rowMax
    colCount = colMax
    maxRange = 2057.701 #Current record set by Penrice
    minPixels = rowMax*colMax/6
    horizontalScopeMin = -180
    horizontalScopeMax = 180
    horizontalScope = horizontalScopeMax-horizontalScopeMin
    verticalScopeMax = 15
    verticalScopeMin = -45
    verticalScope = verticalScopeMax-verticalScopeMin #+5 down to -40
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #vertical SCOOOOPE
    #it goes +5 to -45
    #so when an angle is negative?
    #it ends up as +40
    #which obviously I DO NOT HAVE
    #fuck
    #solution?
    #I need to create for the grid -40 to +40
    #then crop just the part I actually want AFTER I've done the isNeg part.
    #This means I need two rowMax/ColMax values. One for the grid I'm using (Input), one for the I'm making (Output)
    #Angles are the same, everything other than imageDepth is the same.
    #fuuuuuuck me
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    horizontalAngles = math.radians(horizontalScope) / colMax #6 minutes of an arc for a 800x3600 image, 12 minutes for 800x1800 (assumign 360 coverage)
    verticalAngles = math.radians(verticalScope) / rowMax #6 arc minutes (one tenth of a degree). This means every ten pixels is one degree, 80 pixels = +- 40 degrees
    horizontalMin = ((colCount*horizontalScopeMin)//horizontalScope) * (horizontalAngles) + (horizontalAngles)/2

    bigRowCount = (np.max((np.absolute(verticalScopeMin),np.absolute(verticalScopeMax)))*2)*rowCount//verticalScope
    # print("the row")
    # print(bigRowCount)
    bigRowMin = ((bigRowCount)//-2) * verticalAngles + (verticalAngles/2)
    #ok so this *3//4 is because it ends with 45 isntead of 30. So instead of min being -30, for scope -30 -> 30, it is -45 to 15
    #specifically I have 120 pixels below, 40 above.
    #so I need 120 below, 120 above
    #for 240 total
    #I need it to be -45 to 45 for the big version

    verticalMin = ((rowCount*verticalScopeMin)//verticalScope) * verticalAngles + (verticalAngles/2)
    #divide origins by 2000 so in same space as my distances
    # modificationList = np.array([[0,0,0],
    #             [10,0,0],
    #             [0,10,0],
    #             [10,10,0],
    #             [0,0,10],
    #             [-10,0,0],
    #             [0,-10,0],
    #             [-10,-10,0]]) #/ -1
    # modificationList = np.array([[0,0,0],
    #             [0,10,0],
    #             [10,0,0],
    #             [10,10,0],
    #             [0,0,10],
    #             [0,-10,0],
    #             [-10,0,0],
    #             [-10,-10,0]]) / -1

    # modificationList = np.expand_dims(np.expand_dims(modificationList,-1),-1)
    # originListOG = torch.from_numpy(modificationList).to(x_mod.device)
    originListOG = torch.unsqueeze(torch.unsqueeze(modificationList,-1),-1)

    #These need to go in reverse because for (reasons??) the raw data was originally upside down so the XYZ projection requires this, I dunno it's 2am right now spherical/cartesian is pain
    azimuth = torch.from_numpy(np.reshape(((np.arange(colMax-1,-1,-1) * horizontalAngles) + horizontalMin),(1,colCount))).to(x_mod.device)
    elevation = torch.from_numpy(np.reshape(((np.arange(rowMax-1,-1,-1) * verticalAngles) + verticalMin),(rowCount,1))).to(x_mod.device)

    #Now calculate Ground Truth Point Cloud which is to be used at every timestep as the priority source
    #Actually wait no fucking need
    #we have refer_image
    #we have x_indices
    #just fill the gaps bro
    #Ok it's settings time
    #settings == 4 
    GTMed = torch.ones(3)
    if(setting == 7):
        #Move all points toward median of groundtruth
        realDistance = (torch.pow(2,(torch.abs(x_mod[:,0,:,:])*6))-1)
        pointX = torch.add(realDistance * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originListOG[:batchSize,0])[mask[:,0] != 0]
        pointY = torch.add(realDistance * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originListOG[:batchSize,1])[mask[:,0] != 0]
        pointZ = torch.add(realDistance * torch.sin(elevation).view(1,-1,1),originListOG[:batchSize,2])[mask[:,0] != 0]
        GTMed = torch.tensor([torch.median(pointX),torch.median(pointY),torch.median(pointZ)])


    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
        # for c, sigma in enumerate(sigmas[:3]):
            heWhoModsTheModMan = 1
            #Not doing this anymore - now we divide by sigma so it's always norm 1
            # if(setting >= 3 and sigma > 1):
            #     heWhoModsTheModMan = sigma
            #No logging, it fucks shit up
            originList = ((torch.log2(torch.abs(originListOG)+1)) / 6) * heWhoModsTheModMan
            originList = (torch.pow(2,(originList*6))-1)

            #Ok so the issue here is that my origins are all positive right
            # print("THE NEW ORIGINS")
            # print(originList)
            #No need for this unstil I'm logging
            originList = originList/(originListOG+0.00000001) * 10

            # originList = ((((originListOG))) / 2000) * heWhoModsTheModMan
            # originList = (originList*2000)
            # print("fixed")
            # print(originList)
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)
                grad = torch.nan_to_num(grad)
                # used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                # expectedNoise = torch.randn_like(X) * used_sigmas
                # #X = X + noise * mask

                # grad_likelihood = -mask * (x_mod - (refer_image+expectedNoise)) # - 0.05*(1-mask)*(x_mod - raw_interp)
                grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                grad_likelihood_norm = torch.norm(grad_likelihood.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                # print(torch.min(grad))
                # print(torch.max(grad))
                # print(torch.min(grad_likelihood))
                # print(torch.min(correlation_correction))

                # x_mod = x_mod + step_size * grad + step_refer * grad_likelihood + correlation_coefficient * correlation_correction + noise * np.sqrt(step_size * 2)
                #The final modification per pixel:
                if(setting < 4):
                    totalGrad = step_size * grad + step_refer * grad_likelihood + noise * np.sqrt(step_size * 2)
                else:
                    totalGrad = step_size * grad + noise * np.sqrt(step_size * 2)
                # realNoise = totalGrad[:,0] * 2000
                # x_mod = x_mod + step_size * grad + step_refer * grad_likelihood + noise * np.sqrt(step_size * 2)

                #get the shared image version
                #But only if it is after min step
                #Even when not logging, need isNeg to keep negative values after projecting to 3D then back to 2D
                isNeg = (x_mod[:,0,:,:] < 0).int()
                # tooHigh = torch.max(torch.abs(x_mod[:,0,:,:]))*6 > 50
                # print(torch.max(torch.abs(x_mod[:,0,:,:]))*6)
                #ok so this needs to handle negatives as well
                modifierArray = torch.ones_like(x_mod[:,0,:,:]) - (isNeg*2)
                sigmaMod = 1
                if(sigma > 1):
                    sigmaMod = sigma
                realDistance = (torch.pow(2,(torch.abs(x_mod[:,0,:,:])*6/sigmaMod))-1) * modifierArray
                realNoise = (torch.pow(2,(torch.abs(totalGrad[:,0])*6))-1) * modifierArray
                # realDistance = x_mod[:,0,:,:] * 2000

                # realDistance = x_mod[:,0,:,:]*2000
                # isNeg = (realDistance < 0).int()
                #for now just outright skip this when distances too large to overlap
                #smart
                # print("sanity check origins")
                # print(originList.shape)
                # print(originList[:batchSize,0])
                # print(originList[:batchSize,:,0])
                pointX = torch.add(realDistance * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,0])
                pointY = torch.add(realDistance * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,1])
                pointZ = torch.add(realDistance * torch.sin(elevation).view(1,-1,1),originList[:batchSize,2])
                #I need to make a new point cloud, using x_indices to track original points, and then making a new point for any with index of -1
                #Noise should not be adding the origin you idiot
                noiseX = realNoise * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1)
                noiseY = realNoise * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1)
                noiseZ = realNoise * torch.sin(elevation).view(1,-1,1)

                #At this point there should be no -1's as new points also have an index
                #So let's make our point cloud
                #A bunch of pixels have index of -1, let's fix that

                # print("XINDICES SHAPE")
                # print(x_indices.shape)
                x_indices = torch.flatten(x_indices)
                pointX = torch.flatten(pointX)
                pointY = torch.flatten(pointY)
                pointZ = torch.flatten(pointZ)
                noiseX = torch.flatten(noiseX)
                noiseY = torch.flatten(noiseY)
                noiseZ = torch.flatten(noiseZ)

                pointX = torch.cat((pointX[x_indices != -1],pointX[x_indices == -1]))
                pointY = torch.cat((pointY[x_indices != -1],pointY[x_indices == -1]))
                pointZ = torch.cat((pointZ[x_indices != -1],pointZ[x_indices == -1]))
                pointIntensity = torch.cat((torch.flatten(x_mod[:,1])[x_indices != -1],torch.flatten(x_mod[:,1])[x_indices == -1]))

                noiseX = torch.cat((noiseX[x_indices != -1],noiseX[x_indices == -1]))
                noiseY = torch.cat((noiseY[x_indices != -1],noiseY[x_indices == -1]))
                noiseZ = torch.cat((noiseZ[x_indices != -1],noiseZ[x_indices == -1]))
                noiseIntensity = torch.cat((torch.flatten(totalGrad[:,1])[x_indices != -1],torch.flatten(totalGrad[:,1])[x_indices == -1]))

                #Ok now sorted to be old, new 
                #remember I still need to add 2D noise though - I'll be using that to fill empty spaces created by the point cloud shift at the very end
                x_mod = x_mod + step_size * grad + step_refer * grad_likelihood + noise * np.sqrt(step_size * 2)

                #with this indices also sorted to be old, new
                updatedIndices = torch.cat((x_indices[x_indices != -1],torch.arange(torch.sum(x_indices == -1),device=x_mod.device)+torch.max(x_indices).to(x_mod.device)),0)
                baseOnes = torch.ones_like(updatedIndices)
                updatedIndices = torch.unsqueeze(updatedIndices,0)
                # baseOnes = torch.unsqueeze(baseOnes,1)

                #Just going to use a torch.ones_like to handle this one for now - a more efficient method almost certaintly exists
                dupeCounts = torch.sparse_coo_tensor(updatedIndices,baseOnes, size=(torch.max(updatedIndices).int()+1,)).to_dense()
                pointCloudX = torch.sparse_coo_tensor(updatedIndices,pointX, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts
                pointCloudY = torch.sparse_coo_tensor(updatedIndices,pointY, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts
                pointCloudZ = torch.sparse_coo_tensor(updatedIndices,pointZ, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts
                pointCloudI = torch.sparse_coo_tensor(updatedIndices,pointIntensity, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts

                #now I need to use coo to get the total noise. Averaging this as well so, among other things, I don't overshoot when going back to GT values.

                totalNoiseX = torch.sparse_coo_tensor(updatedIndices,noiseX, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts
                totalNoiseY = torch.sparse_coo_tensor(updatedIndices,noiseY, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts
                totalNoiseZ = torch.sparse_coo_tensor(updatedIndices,noiseZ, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts
                totalNoiseI = torch.sparse_coo_tensor(updatedIndices,noiseIntensity, size=(torch.max(updatedIndices).int()+1,)).to_dense()/dupeCounts

                #ok now I have my cloud I can add noise to it
                pointCloudX = torch.add(pointCloudX,totalNoiseX)
                pointCloudY = torch.add(pointCloudY,totalNoiseY)
                pointCloudZ = torch.add(pointCloudZ,totalNoiseZ)
                pointCloudI = torch.add(pointCloudI,totalNoiseI)

                #If setting is 7, also shift each point toward the median by 10%
                if(setting == 7 and c < 200):
                    pointCloudX = (0.9*pointCloudX) + 0.1*(GTMed[0])
                    pointCloudY = (0.9*pointCloudY) + 0.1*(GTMed[1])
                    pointCloudZ = (0.9*pointCloudZ) + 0.1*(GTMed[2])

                #Ok noise is now added
                #now I can reproject with this as my shared point cloud
                #All that is left is making sure I fill holes with x_mod, and that I updated x_indices as necessary
                #Add 1 so range is 1:Max
                #This way I can subtract 1 later to have empty pixels be -1
                newPointCloudIndices = torch.arange(pointCloudX.size(dim=0)).to(x_mod.device) + 1
                sharedCloud = torch.unsqueeze(torch.stack((pointCloudX,pointCloudY,pointCloudZ,pointCloudI)).view(4,-1),0)
                #now I need to get the resultant distance for EVERY origin
                bigCloud = torch.tile(sharedCloud[:,:3],(batchSize,1,1))
                #the 0 in originlist is necessary to reduce it from Ox3x1x1 to Ox3x1
                relativePoints = torch.subtract(bigCloud,originList[:batchSize,:,0])

                xy = torch.square(relativePoints[:,0]) + torch.square(relativePoints[:,1])
                newDepth = torch.sqrt(xy + torch.square(relativePoints[:,2]))
                # print("depths")
                # print(torch.max(x_mod))
                # print(torch.min(x_mod))
                # print(torch.max(realDistance))
                # print(torch.min(realDistance))
                # print(torch.max(newDepth))

                newDepth = ((torch.log2(newDepth+1)) / 6 * sigmaMod)
                # newDepth = newDepth / 2000
                # print("sanity check")
                #this passes
                # print(torch.sum(torch.abs(newDepth)) - torch.sum(torch.abs(x_mod[:,0,:,:])))
                # #this also passes
                # print(torch.sum(newDepth * modifierArray.flatten()) - torch.sum(x_mod[:,0,:,:]))
                # #now the real test
                # print(torch.sum(newDepth * modifierArray.flatten() - x_mod[:,0,:,:].flatten()))
                #This also passes because it is the new depth for each pixel
                #I have the row and column already
                #If I multiply depth by modifierArray I get the original depth value before I absoluted it
                #So now I have for that row & column, all the depths as they should be
                #oh no this fails outside batch=1
                #because what about for other viewpoints idiot
                #yeah stick with current method
                #So we're back to drawing board
                #on why the fuck I get these white patches of super far points
                # newDepth = newDepth / 2000
                # print(torch.max(newDepth))
                # newDepth = torch.clip(newDepth,min=0)
                # print(torch.max(newDepth))
                horizontal = torch.atan2(relativePoints[:,1], relativePoints[:,0])
                # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
                xy = torch.sqrt(xy)
                vertical = torch.atan2(relativePoints[:,2], xy)

                newCol = torch.round(torch.divide((horizontal-horizontalMin),horizontalAngles)).int()
                newRow = torch.round(torch.divide((vertical-bigRowMin),verticalAngles)).int()
                #Ok so here I have the row and column values
                #I need to flip them to make it work though
                #
                #Now I need to unflip it *again* because... reasons. The projection hates me that's why. I don't know why it had to be reversed in the raw data but I pay for that consequence now.
                newCol = newCol * -1 + colCount -1
                newRow = newRow * -1 + bigRowCount- 1
                #Post flip needs to be 0 -> bigRow
                #I think it should work
                #but clearly it doesn't
                #ugh

                # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.less(newRow,rowCount)))
                # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.logical_and(torch.less(newRow,rowCount),torch.greater(torch.unsqueeze(realDistance.flatten(),0),0))))
                # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.less(newRow,rowCount)))
                
                #now including all points, including Sky & fourth viewpoint, because I lowkey have to :/
                #If I really want I could try priorising non-sky points first in a two-step process but I'll do that Monday
                # isFour = torch.stack((torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.ones(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount)),0)
                # isFour = torch.unsqueeze(isFour,0)
                # isFour = torch.cat((torch.tile(torch.logical_not(isFour),(4,1,1)),torch.ones_like(isFour),torch.tile(torch.logical_not(isFour),(3,1,1))),0).view(8,-1).to(x_mod.device)
                # print(isFour.shape)
                inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,-1),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,-1),torch.less(newRow,bigRowCount))).to(x_mod.device)
                #Ok now that i have inGrid it's time to ditch the indices I don't use
                usedPoints = torch.any(inGrid,0).to(x_mod.device)
                newPointCloudIndices = torch.where(usedPoints,newPointCloudIndices,torch.tensor(-2).to(x_mod.device))
                tempUnique, newCloudInverse = torch.unique(newPointCloudIndices, return_inverse = True)
                newCloudSize = tempUnique.size(dim=0)
                newPointCloudIndices = torch.arange(newCloudSize)[newCloudInverse].to(x_mod.device)
                #Ok we finally have it
                #And because the first value from the torch.arange(), being 0, corresponds to -2, we can simply treat it as 1-size+1
                #which is exactly what we want to be able to subtract 1 from it later so empty pixels are -1 and not 0

                #Do not include sky points when sharing because it fucks shit up
                # inGrid = torch.logical_and(inGrid,torch.unsqueeze(sky.flatten(),0))
                # # print(inGrid.shape)
                # inGrid = torch.logical_and(isFour,inGrid)
                        
                del bigCloud
                newStack = []
                depthStack = []
                intensityStack = []
                imageMaskStack = []
                imageIndexStack = []
                for origin in range(batchSize):
                    new_ind = torch.argsort(torch.flatten(newDepth[origin][inGrid[origin]]),dim=0, descending = True)
                    # print("depth sort sanity check")
                    # print(torch.flatten(newDepth[origin][inGrid[origin]])[new_ind])
                    # print(torch.max(new_ind))
                    # print("the shapes")
                    # print(torch.min(x_mod))
                    # print(torch.min(pointX))
                    # print(torch.min(pointY))
                    # print(torch.min(pointZ))
                    # print(torch.min(horizontal))
                    # print(torch.min(vertical))
                    #Because pytorch is stupid and sorts before any torch.unique() I need to do the depth sorting AFTER getting unique values
                    #It's very dumb I know
                    #but if I take reduced_ind and sort it by new_ind
                    #then I run unique() on it again to remove duplicates
                    newRowTemp =torch.flatten(newRow[origin][inGrid[origin]])[new_ind]
                    # print(newRowTemp.shape)
                    newColTemp =torch.flatten(newCol[origin][inGrid[origin]])[new_ind]
                    #At this point merged is a row,column list sorted by depth
                    #I need to remove duplicates, but with removal based on depth
                    newRowTemp, theSortProjectionOne = torch.sort(newRowTemp, stable = True)
                    newColTemp = newColTemp[theSortProjectionOne]
                    newColTemp, theSortProjectionTwo = torch.sort(newColTemp, stable = True)
                    newRowTemp = newRowTemp[theSortProjectionTwo]
                    merged = torch.stack((newRowTemp,newColTemp),0)

                    # sortedMerged, theSortProjectionOne  = torch.sort(merged, stable = True, dim = 1)
                    # print(theSortProjectionOne)
                    #the inverse is useless, but because it's sorted, the count can be cumsum'd into what I need
                    _,duplicate_count= torch.unique_consecutive(merged, return_counts=True, dim=1)
                    # print(reduced_ind.shape)
                    reduced_ind = torch.cumsum(duplicate_count,0)-1
                    # print(reduced_ind.shape)

                    #this reduced_ind is for the depth_sorted row and col values
                    #It will take the values in any depth-sorted array, and convert them to a unique row/column sorted array



                    #Because Pytorch is garbage, I have no choice but to sort arrays before using unique on them. It's a huge fucking pain.
                    #This code can be massively cleaned up when pytorch finally gets around to making sorted=False work on CUDA.
                    #But until then, here we are, suffering together :)
                    # reduced_ind, theSortProjectionThree = torch.sort(reduced_ind,stable=True)
                    # print(torch.flatten(newDepth[origin][inGrid[origin]])[new_ind][theSortProjection])


                    final_ind = torch.arange(len(newDepth[0]),device=x_mod.device)
                    
                    gridInd = inGrid[origin]
                    
                    final_ind = final_ind[gridInd]
                    final_ind = final_ind[new_ind]
                    mergedReduced = torch.stack((newRowTemp[reduced_ind],newColTemp[reduced_ind]),0)
                    #ok so after doing this merged is no longer sorte by depth but by row/col
                    #so by definition any unique() function will not remove depth first
                    #except I use reduced_ind as a mask
                    #reduced_ind which is..... also sorted
                    #ughhhhh
                    #I need a fucking mask
                    # print(merged.shape)
                    #And with this we have merged and final_ind, both ready to be used to make new images
                    final_ind = final_ind[theSortProjectionOne][theSortProjectionTwo]

                    #for average across shared point cloud
                    # imageDepth = torch.sparse_coo_tensor(merged,torch.flatten(newDepth[origin][final_ind]), size=(bigRowCount,colMax)).to_dense()
                    # imageIntensity = torch.sparse_coo_tensor(merged,torch.flatten(sharedCloud[0,3,:])[final_ind], size=(bigRowCount,colMax)).to_dense()
                    # scaling = torch.sparse_coo_tensor(mergedReduced,duplicate_count, size=(bigRowCount,colMax)).to_dense() + 0.000000001
                    # imageDepth = torch.div(imageDepth,scaling)
                    # imageIntensity = torch.div(imageIntensity,scaling)
                    # #for minimum
                    final_ind = final_ind[reduced_ind]
                    imageDepth = torch.sparse_coo_tensor(mergedReduced,torch.flatten(newDepth[origin])[final_ind], size=(bigRowCount,colMax)).to_dense()
                    imageIntensity = torch.sparse_coo_tensor(mergedReduced,torch.flatten(sharedCloud[0,3,:])[final_ind], size=(bigRowCount,colMax)).to_dense()
                    imageIndex = torch.sparse_coo_tensor(mergedReduced,torch.flatten(newPointCloudIndices)[final_ind], size=(bigRowCount,colMax)).to_dense()

                    #in case some pixels have rounding errors and end up empty just ignore them
                    imageMask = torch.sparse_coo_tensor(mergedReduced,torch.ones_like(mergedReduced[0], device=x_mod.device), size=(bigRowCount,colMax)).to_dense()
                    
                    #ok now lets get the negs, since during diffusion some pixels will have a negative distance. For these we simply take the value of the opposite pixel and multiply by -1
                    #to keep it simple - if a pixel was negative before the shared pointcloud, it shall be negative after the shared point cloud.
                    # print("is it the neg")
                    # print(isNeg.shape)
                    # print(imageDepth.shape)
                    imageIntensity = torch.add(imageIntensity[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageIntensity,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])
                    imageIndex = torch.add(imageIndex[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageIndex,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])
                    imageMask = torch.add(imageMask[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageMask,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])
                    # imageDepth = torch.add(imageDepth[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageDepth,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin] * -1)
                    #Don't multiply by -1 for second method or else half the pixels are doomed to forever be negative
                    imageDepth = torch.add(imageDepth[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageDepth,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])

                    #Now it's time to subtract 1 from index
                    imageIndex = imageIndex - 1
                    

                    depthStack.append(imageDepth)
                    intensityStack.append(imageIntensity)
                    imageMaskStack.append(torch.unsqueeze(imageMask.int(),0))
                    imageIndexStack.append(imageIndex)
                newImageDepths = torch.stack(depthStack).float()
                newImageIntensities = torch.stack(intensityStack).float()
                maskImages = torch.stack(imageMaskStack).float()
                #Update X_indices
                x_indices = torch.stack(imageIndexStack)
                #replace any empty pixels with updated x_mod
                newImageIntensities = torch.where(newImageDepths == 0, x_mod[:,1],newImageIntensities)
                newImageDepths = torch.where(newImageDepths == 0, x_mod[:,0],newImageDepths)
                #Update X_mod
                newImages = torch.stack((newImageDepths,newImageIntensities),dim=1)
                if(c == 0 or c == 20 or c == 110):
                    sharedImages.append(newImages.to('cpu'))
                x_mod = newImages
                #Aaaand that's it. Comment out the rest
                #At this point we have our new x_mod, but if setting is 4, hard reset known points to the ground truth here
                # print("shape before")
                # print(x_mod.shape)
                # print(x_indices.shape)
                # print(refer_indices.shape)
                # print("Index Shapes")
                # print(x_indices.shape)
                # print(refer_indices.shape)
                # print(mask.shape))
                if(setting >= 4):
                    x_mod = torch.where(mask!=0, refer_image + noise * np.sqrt(step_size * 2),x_mod)
                    x_indices = x_indices + torch.max(refer_indices)
                    x_indices = torch.squeeze(torch.where(torch.unsqueeze(mask[:,0],1) !=0, refer_indices, torch.unsqueeze(x_indices,1)))
                # print("shape after")
                # print(x_mod.shape)
                # print(x_indices.shape)

                #Left to do:
                #Am I ensuring ground truth is reset every timestep?
                #I am, but I'm including it in the noiseRemoval per pixel
                #So that means it's currently being ADDED
                #let's divide the noise as well I guess to be an average of each pixel's reccomended noise
                #that way we're averaging each point's return to GT
                #ummmm
                #where is best to do this huh
                #
                
                # # print(torch.min(newImages))
                # #ok now I have my new image and mask
                # # print(newImages.shape)
                # # print(maskImages.shape)
                # #This needs to not overwrite any pixel which has an actual known groundtruth, so multiply by not(mask) as well to only update unknown points
                # #Don't bother updating sky points either
                # maskImages = torch.logical_and(maskImages, sky).int() 
                # correlation_correction = -maskImages * torch.logical_not(mask).int() * torch.sub(x_mod, newImages) # - 0.05*(1-mask)*(x_mod - raw_interp)
                # #if the distances are too large right now, don't even bother with this shit
                # correlation_correction = torch.where(tooHigh, torch.tensor(0, device=x_mod.device).float(), correlation_correction)
                
                # # print("why do you hurt meeeee")
                # # print(x_mod.shape)
                # # print(newImages.shape)
                # # testVal = x_mod[:,0,:,:] - newImages[:,0,:,:]
                # # print(testVal.shape)
                # # testOne = torch.argmax(testVal)
                # # testTwo = torch.argmin(testVal)
                # # print("ugh")
                # # print(testOne)
                # # print(x_mod[:,0,:,:].flatten()[testOne])
                # # print(newImages[:,0,:,:].flatten()[testOne])
                # # print(newDepth.flatten()[testOne])
                # # print(isNeg.flatten()[testOne])
                # # print(newRow.flatten()[testOne])
                # # predRow = testOne//colCount
                # # print(predRow)
                # # print(newCol.flatten()[testOne])
                # # print(testOne % colCount)


                # # print("yet this one fails??")
                # # print(testTwo)
                # # print(x_mod[:,0,:,:].flatten()[testTwo])
                # # print(newImages[:,0,:,:].flatten()[testTwo])
                # # print(newDepth.flatten()[testTwo])
                # # print(isNeg.flatten()[testTwo])
                # # print(newRow.flatten()[testTwo])
                # # predRow = testTwo//colCount
                # # print(predRow)
                # # print(newCol.flatten()[testTwo])
                # # print(testTwo % colCount)



                # # print("it's probably my roll or flip being slightly off")
                # # #a one pixel difference could account for the issues
                # # #because in correlation correction a negative x_mod minus a positive newImages results in double the absolute value
                # # #It's not that because the newImage itself has the white pixels with super large values
                # # print(torch.max(torch.abs(x_mod[:,0,:,:]))*6)
                # # print(torch.sum(correlation_correction) == 0)
                # # print(torch.sum(torch.abs(x_mod)) - torch.sum(torch.abs(newImages)))
                # # print(torch.sum(torch.abs(x_mod) - torch.abs(newImages)))
                # # print(torch.sum((x_mod - (newImages))))
                # # print(torch.max(torch.abs(newImages)))
                # #somehow this is... extremely high
                # #despite batch size being 1
                # #so in theory it should ALWAYS be 0
                # #shared should always be the only one
                # #Lazy option is just put some code in to set it to 0 when it's stupid high
                # #Good option is work out why it is usually 0 for batch 1 but sometimes 100286.1406 or 3514.5759 etc
                # #It's like it decided every single pixel should just be way off
                # #or maybe one pixel is insanely off
                # #Small error could be explained as when a negative pixel overlaps with a positive causing one of them to differ even with batch size 1
                # #but unless my coefficient 9999999999...
                # #
                # #who fucking knows
                # #It being 0 MOST of the time means the rest of my code is correct
                # #with the generated shared cloud being identical to the OG 
                # x_mod = x_mod + correlation_coefficient * correlation_correction

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                images.append(x_mod.to('cpu'))
            if verbose and c % 20 == 0:
                print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))
                print("level: {}, step_size: {}, grad_norm: {}, grad_likelihood_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), grad_likelihood_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise) + step_refer * grad_likelihood
            # images.append(x_mod.to('cpu'))
            print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))
        #This is what ensures the final image has the original Ground Truth pixels exactly correct (assuming step_refer = 1)
        #I am going to ditch it and just.... see what happens
        # grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)
        # x_mod = x_mod + step_refer * grad_likelihood
        images.append(x_mod.to('cpu'))
        # #Remove once I've fixed it
        # # x_mod = refer_image
        # # print("uhhh")
        # # print(torch.sum(refer_image == 0))
        # images.append(x_mod.to('cpu'))
        # print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))
        # #get the shared images
        # #get the shared image version
        # isNeg = (x_mod[:,0,:,:] < 0).int()
        # tooHigh = torch.max(torch.abs(x_mod[:,0,:,:]))*6 > 50
        # #ok so this needs to handle negatives as well
        # modifierArray = torch.ones_like(x_mod[:,0,:,:]) - (isNeg*2)
        # realDistance = (torch.pow(2,(torch.abs(x_mod[:,0,:,:])*6))-1) * modifierArray
        # #for now just outright skip this when distances too large to overlap
        # #smart


        # pointX = torch.add(realDistance * torch.cos(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,0])
        # pointY = torch.add(realDistance * torch.sin(azimuth).view(1,1,-1) * torch.cos(elevation).view(1,-1,1),originList[:batchSize,1])
        # pointZ = torch.add(realDistance * torch.sin(elevation).view(1,-1,1),originList[:batchSize,2])
        # sharedCloud = torch.unsqueeze(torch.stack((pointX,pointY,pointZ,x_mod[:,1,:,:])).view(4,-1),0)
        # #now I need to get the resultant distance for EVERY origin
        # bigCloud = torch.tile(sharedCloud[:,:3],(batchSize,1,1))
        # #the 0 in originlist is necessary to reduce it from Ox3x1x1 to Ox3x1
        # relativePoints = torch.subtract(bigCloud,originList[:batchSize,:,0])

        # xy = torch.square(relativePoints[:,0]) + torch.square(relativePoints[:,1])
        # newDepth = torch.sqrt(xy + torch.square(relativePoints[:,2]))
        # # print("depths")
        # # print(torch.max(x_mod))
        # # print(torch.min(x_mod))
        # # print(torch.max(realDistance))
        # # print(torch.min(realDistance))
        # # print(torch.max(newDepth))
        # newDepth = ((torch.log2(newDepth+1)) / 6)
        # newDepth = ((torch.log2(newDepth+1)) / 6)
        # # newDepth = newDepth / 2000
        # # print(torch.max(newDepth))
        # # newDepth = torch.clip(newDepth,min=0)
        # # print(torch.max(newDepth))
        # horizontal = torch.atan2(relativePoints[:,1], relativePoints[:,0])
        # # whatTextbookSaysVerticalShouldBe = np.arctan2(np.sqrt(xy),translatedPoint[2])
        # xy = torch.sqrt(xy)
        # vertical = torch.atan2(relativePoints[:,2], xy)
        # newCol = torch.round(torch.divide((horizontal-horizontalMin),horizontalAngles)).int()
        # newRow = torch.round(torch.divide((vertical-bigRowMin),verticalAngles)).int()

        # #Now I need to unflip it *again* because... reasons. The projection hates me that's why. I don't know why it had to be reversed in the raw data but I pay for that consequence now.
        # newCol = newCol * -1 + colCount -1
        # newRow = newRow * -1 + bigRowCount - 1

        # # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.less(newRow,rowCount)))
        # # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.logical_and(torch.less(newRow,rowCount),torch.greater(torch.unsqueeze(realDistance.flatten(),0),0))))
        # # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,0),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,0),torch.less(newRow,rowCount)))
        # #edit this to not include the high Z origin
        # isFour = torch.stack((torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.ones(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount),torch.zeros(rowCount*colCount)),0)
        # isFour = torch.unsqueeze(isFour,0)
        # isFour = torch.cat((torch.tile(torch.logical_not(isFour),(4,1,1)),torch.ones_like(isFour),torch.tile(torch.logical_not(isFour),(3,1,1))),0).view(8,-1).to(x_mod.device)
        # # # print(isFour.shape)
        # inGrid = torch.logical_and(torch.logical_and(torch.greater(newCol,-1),torch.less(newCol,colCount)),torch.logical_and(torch.greater(newRow,-1),torch.less(newRow,bigRowCount)))
        
        # inGrid = torch.logical_and(inGrid,torch.unsqueeze(sky.flatten(),0))# # print(inGrid.shape)
        # inGrid = torch.logical_and(isFour,inGrid)
                
        # del bigCloud
        # newStack = []
        # imageMaskStack = []
        # for origin in range(batchSize):
        #     new_ind = torch.argsort(torch.flatten(newDepth[origin][inGrid[origin]]),dim=0, descending = True)
        #     # print("depth sort sanity check")
        #     # print(torch.flatten(newDepth[origin][inGrid[origin]])[new_ind])
        #     # print(torch.max(new_ind))
        #     # print("the shapes")
        #     # print(torch.min(x_mod))
        #     # print(torch.min(pointX))
        #     # print(torch.min(pointY))
        #     # print(torch.min(pointZ))
        #     # print(torch.min(horizontal))
        #     # print(torch.min(vertical))
        #     #Because pytorch is stupid and sorts before any torch.unique() I need to do the depth sorting AFTER getting unique values
        #     #It's very dumb I know
        #     #but if I take reduced_ind and sort it by new_ind
        #     #then I run unique() on it again to remove duplicates
        #     newRowTemp =torch.flatten(newRow[origin][inGrid[origin]])[new_ind]
        #     # print(newRowTemp.shape)
        #     newColTemp =torch.flatten(newCol[origin][inGrid[origin]])[new_ind]
        #     #At this point merged is a row,column list sorted by depth
        #     #I need to remove duplicates, but with removal based on depth
        #     newRowTemp, theSortProjectionOne = torch.sort(newRowTemp, stable = True)
        #     newColTemp = newColTemp[theSortProjectionOne]
        #     newColTemp, theSortProjectionTwo = torch.sort(newColTemp, stable = True)
        #     newRowTemp = newRowTemp[theSortProjectionTwo]
        #     merged = torch.stack((newRowTemp,newColTemp),0)

        #     # sortedMerged, theSortProjectionOne  = torch.sort(merged, stable = True, dim = 1)
        #     # print(theSortProjectionOne)
        #     #the inverse is useless, but because it's sorted, the count can be cumsum'd into what I need
        #     _,duplicate_count= torch.unique_consecutive(merged, return_counts=True, dim=1)
        #     # print(reduced_ind.shape)
        #     reduced_ind = torch.cumsum(duplicate_count,0)-1
        #     # print(reduced_ind.shape)

        #     #this reduced_ind is for the depth_sorted row and col values
        #     #It will take the values in any depth-sorted array, and convert them to a unique row/column sorted array



        #     #Because Pytorch is garbage, I have no choice but to sort arrays before using unique on them. It's a huge fucking pain.
        #     #This code can be massively cleaned up when pytorch finally gets around to making sorted=False work on CUDA.
        #     #But until then, here we are, suffering together :)
        #     # reduced_ind, theSortProjectionThree = torch.sort(reduced_ind,stable=True)
        #     # print(torch.flatten(newDepth[origin][inGrid[origin]])[new_ind][theSortProjection])


        #     final_ind = torch.arange(len(newDepth[0]),device=x_mod.device)
            
        #     gridInd = inGrid[origin]
            
        #     final_ind = final_ind[gridInd]
        #     final_ind = final_ind[new_ind]
        #     mergedReduced = torch.stack((newRowTemp[reduced_ind],newColTemp[reduced_ind]),0)
        #     #ok so after doing this merged is no longer sorte by depth but by row/col
        #     #so by definition any unique() function will not remove depth first
        #     #except I use reduced_ind as a mask
        #     #reduced_ind which is..... also sorted
        #     #ughhhhh
        #     #I need a fucking mask
        #     # print(merged.shape)
        #     #And with this we have merged and final_ind, both ready to be used to make new images
        #     final_ind = final_ind[theSortProjectionOne][theSortProjectionTwo]

        #     #for average across shared point cloud
        #     #I need this to somehow remove outliers
        #     #Ok if we assume all points are <500 or >500
        #     #no we can do better than that
        #     #calculate min
        #     #get total
        #     #get average
        #     #if abs(average - min) > 20
        #     #just fucking set it to min + 20
        #     #no we can still do better
        #     #
        #     #Average only points within ~20 metres of 
        #     # imageDepth = torch.sparse_coo_tensor(merged,torch.flatten(newDepth[origin][final_ind]), size=(bigRowCount,colMax)).to_dense()
        #     # imageIntensity = torch.sparse_coo_tensor(merged,torch.flatten(sharedCloud[0,3,:])[final_ind], size=(bigRowCount,colMax)).to_dense()
        #     # scaling = torch.sparse_coo_tensor(mergedReduced,duplicate_count, size=(bigRowCount,colMax)).to_dense() + 0.000000001
        #     # imageDepth = torch.div(imageDepth,scaling)
        #     # imageIntensity = torch.div(imageIntensity,scaling)
        #     # #for minimum
        #     final_ind = final_ind[reduced_ind]
        #     imageDepth = torch.sparse_coo_tensor(mergedReduced,torch.flatten(newDepth[origin])[final_ind], size=(bigRowCount,colMax)).to_dense()
        #     imageIntensity = torch.sparse_coo_tensor(mergedReduced,torch.flatten(sharedCloud[0,3,:])[final_ind], size=(bigRowCount,colMax)).to_dense()

        #     #in case some pixels have rounding errors and end up empty just ignore them
        #     imageMask = torch.sparse_coo_tensor(mergedReduced,torch.ones_like(mergedReduced[0], device=x_mod.device), size=(bigRowCount,colMax)).to_dense()
            
        #     #ok now lets get the negs, since during diffusion some pixels will have a negative distance. For these we simply take the value of the opposite pixel and multiply by -1
        #     #to keep it simple - if a pixel was negative before the shared pointcloud, it shall be negative after the shared point cloud.
        #     # print("is it the neg")
        #     # print(isNeg.shape)
        #     # print(imageDepth.shape)
        #     sharedImages.append(torch.stack((imageDepth,imageIntensity)).to('cpu'))
        #     # print("THE SHAPE")
        #     # print(sharedImages[-1].shape)
        #     imageIntensity = torch.add(imageIntensity[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageIntensity,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])
        #     imageMask = torch.add(imageMask[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageMask,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin])
        #     imageDepth = torch.add(imageDepth[(bigRowCount - rowCount):] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageDepth,(colCount//2),(1,)),(0,))[(bigRowCount - rowCount):] * isNeg[origin] * -1)
        #     #Intensity the way to check actually
        #     #in case some pixels have rounding errors and end up empty just ignore them
        #     # imageMask = torch.sparse_coo_tensor(mergedReduced,torch.ones_like(mergedReduced[0], device=x_mod.device), size=(bigRowCount,colMax)).to_dense()
        #     # print("why shape like this 1")
        #     # print(imageDepth.shape)
        #     # print(imageIntensity.shape)
        #     # print(isNeg.shape)
            
        #     #ok now lets get the negs, since during diffusion some pixels will have a negative distance. For these we simply take the value of the opposite pixel and multiply by -1
        #     #to keep it simple - if a pixel was negative before the shared pointcloud, it shall be negative after the shared point cloud.
        #     imageIntensityUpdated = torch.add(imageIntensity[:rowCount] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageIntensity,(colCount//2),(1,)),(0,))[:rowCount] * isNeg[origin])
        #     imageMaskUpdated = torch.add(imageMask[:rowCount] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageMask,(colCount//2),(1,)),(0,))[:rowCount] * isNeg[origin])
        #     imageDepthUpdated = torch.add(imageDepth[:rowCount] * torch.logical_not(isNeg[origin]).int(),torch.flip(torch.roll(imageDepth,(colCount//2),(1,)),(0,))[:rowCount] * isNeg[origin] * -1)
        #     # print("why shape like this 2")
        #     # print(imageDepth.shape)
        #     # print(imageIntensity.shape)
            
            

        #     newStack.append(torch.stack((imageDepthUpdated,imageIntensityUpdated)))
        #     # imageMaskStack.append(torch.unsqueeze(imageMask.int(),0))
        # newImages = torch.stack(newStack).float()

        # #images.append(refer_image.to('cpu'))
        # targets.append(refer_image.to('cpu'))
        # # sharedImages.append(x_mod.to('cpu'))
        # sharedImages.append(newImages.to('cpu'))
        return images, targets, sharedImages


def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, refer_mask, scorenet, sigmas,
                                        n_steps_each=100, step_lr=0.000008, denoise = True, verbose = True, grad_ref = 0.1, sampling_step = 16):
    images = []
    targets = []
    mask = refer_mask
    raw = refer_image
    # raw_interp = torch.nn.functional.interpolate(raw, size=(64, 1024), mode='bilinear')
    # mask[:, :, 0:64:sampling_step, :] = 1
    step_refer = grad_ref# 0.001

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)



                # used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                # expectedNoise = torch.randn_like(X) * used_sigmas
                # #X = X + noise * mask

                # grad_likelihood = -mask * (x_mod - (refer_image+expectedNoise)) # - 0.05*(1-mask)*(x_mod - raw_interp)
                grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                grad_likelihood_norm = torch.norm(grad_likelihood.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + step_refer * grad_likelihood + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                images.append(x_mod.to('cpu'))
            if verbose and c % 20 == 0:
                print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))
                print("level: {}, step_size: {}, grad_norm: {}, grad_likelihood_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                    c, step_size, grad_norm.item(), grad_likelihood_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise) + step_refer * grad_likelihood
            images.append(x_mod.to('cpu'))
            print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))

        grad_likelihood = -mask * (x_mod - refer_image) # - 0.05*(1-mask)*(x_mod - raw_interp)
        x_mod = x_mod + step_refer * grad_likelihood
        images.append(x_mod.to('cpu'))
        print("grad_ref: {}, mean: {}, median: {}".format(grad_ref, torch.mean(torch.abs(x_mod - refer_image)), torch.median(torch.abs(x_mod - refer_image))))

        #images.append(refer_image.to('cpu'))
        targets.append(refer_image.to('cpu'))
        return images, targets

# @torch.no_grad()
# def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
#                                         n_steps_each=100, step_lr=0.000008):
#     """
#     Currently only good for 32x32 images. Assuming the right half is missing.
#     """

#     images = []

#     refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
#     refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)
#     x_mod = x_mod.view(-1, 3, image_size, image_size)
#     cols = image_size // 2
#     half_refer_image = refer_image[..., :cols]
#     with torch.no_grad():
#         for c, sigma in enumerate(sigmas):
#             labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
#             labels = labels.long()
#             step_size = step_lr * (sigma / sigmas[-1]) ** 2

#             for s in range(n_steps_each):
#                 images.append(x_mod.to('cpu'))
#                 corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
#                 x_mod[:, :, :, :cols] = corrupted_half_image
#                 noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
#                 grad = scorenet(x_mod, labels)
#                 x_mod = x_mod + step_size * grad + noise
#                 print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
#                                                                          grad.abs().max()))

#         return images
