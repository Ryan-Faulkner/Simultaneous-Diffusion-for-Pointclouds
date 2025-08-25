import os, glob, pickle
import sys
from glob import glob
sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import metrics.iou as lidargen_iou
import numpy as np
import cv2
if __name__ == '__main__':
    expOG = "DGXDataLiDARGenSettings/*"
    print("hwy")
    listOfExperiments = glob(expOG)
    print(listOfExperiments)
    for experiment in listOfExperiments:
        exp = experiment
        expInput = os.path.join(exp, "Input")
        expBILINEAROrganised = os.path.join(exp, "Input/BILINEAR")
        expNNOrganised = os.path.join(exp, "Input/NN")
        expBICUBICOrganised = os.path.join(exp, "Input/BICUBIC")
        expNSOrganised = os.path.join(exp, "Input/NS")
        os.system("rm -r " + str(expBICUBICOrganised))
        os.system("mkdir " + str(expBICUBICOrganised))

        os.system("rm -r " + str(expBILINEAROrganised))
        os.system("mkdir " + str(expBILINEAROrganised))

        os.system("rm -r " + str(expNNOrganised))
        os.system("mkdir " + str(expNNOrganised))

        os.system("rm -r " + str(expNSOrganised))
        os.system("mkdir " + str(expNSOrganised))
        counterMax = 1
        if(experiment.split('/')[-1] == "Densification"):
            counterMax = 3
        for counter in range(counterMax):
            current_index = 0
            toGlob = ""
            # if(counter == 0):
            #     toGlob = expGT
            # elif(counter == 1):
            #     toGlob = expSimultaneous
            # else:
            #     toGlob = expLiDARGen
            toGlob = expInput
            for file in np.sort(glob(toGlob + '/*.npy')):
                file = np.load(file)
                distanceFULL = file[:file.shape[0]//2,0]
                intensityFULL = file[file.shape[0]//2:,0]
                loggedIgnore = 0.1
                loggedIgnore = ((np.log2(loggedIgnore+1)) / 6)
                saveSpot = ""
                # print(distance.shape)
                kNums = distanceFULL.shape[0] // 6
                for sample in range(kNums * 6):
                    distance = distanceFULL[sample].astype(np.float32)
                    intensity = intensityFULL[sample].astype(np.float32)
                    if(experiment.split('/')[-1] != "Densification"):
                        distance = cv2.inpaint(distance,(distance <= loggedIgnore).astype(np.uint8),3, flags=0)
                        intensity = cv2.inpaint(intensity,(distance <= loggedIgnore).astype(np.uint8),3, flags=0)
                        saveSpot = expNSOrganised
                    else:
                        if(counter == 0): #nearest neighbour
                            distance = cv2.resize(distance[0::4],(0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
                            intensity = cv2.resize(intensity[0::4], (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
                            saveSpot = expNNOrganised
                        elif(counter == 1): #bicubic
                            distance = cv2.resize(distance[0::4],(0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
                            intensity = cv2.resize(intensity[0::4], (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
                            saveSpot = expBICUBICOrganised
                        else: #linear     
                            distance = cv2.resize(distance[0::4],(0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_LINEAR)
                            intensity = cv2.resize(intensity[0::4], (0, 0), fx=1.0, fy=4.0, interpolation=cv2.INTER_LINEAR)
                            saveSpot = expBILINEAROrganised
                    combined = np.stack((distance,intensity),0)
                    # os.system("rm -r " + str(expSimultaneousOrganised))
                    toSave = os.path.join(saveSpot, "k_" + str(sample % kNums))
                    # expSimultaneousKVal = os.path.join(expSimultaneousOrganised, "k_" + str(sample % kNums))
                    # expLiDARGenKVal = os.path.join(expLiDARGenOrganised, "k_" + str(sample % kNums))
                    # toSave = ""
                    os.system("mkdir " + str(toSave))
                    # elif(counter == 1):
                    #     toSave = expSimultaneousKVal
                    #     os.system("mkdir " + str(expSimultaneousKVal))
                    # else:
                    #     toSave = expLiDARGenKVal
                    #     os.system("mkdir " + str(expLiDARGenKVal))
                    np.save(os.path.join(toSave, str(sample//kNums + current_index) + '.npy'),combined)
                    # np.save(sample_target[i], str(args.exp) + '/densification_target/' + str(current_index) + '.pth')
                current_index = current_index + 6


#To DO
#Make program that just fucking shifts all the results into the format the existing LiDARGen evaluation code expects - that means batch size 1, etc etc. Check the original iou code to see how it wants it's shit saved