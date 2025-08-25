import os, glob, pickle
import sys
from glob import glob
sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import metrics.iou as lidargen_iou
import numpy as np
if __name__ == '__main__':
    expOG = "DGXDataLiDARGenSettings/*"
    print("hwy")
    listOfExperiments = glob(expOG)
    print(listOfExperiments)
    for experiment in listOfExperiments:
        exp = experiment
        expGT = os.path.join(exp, "GroundTruth")
        expSimultaneous = os.path.join(exp, "Simultaneous")
        expLiDARGen = os.path.join(exp, "LiDARGen")
        expGTOrganised = os.path.join(exp, "GroundTruth/Organised")
        expSimultaneousOrganised = os.path.join(exp, "Simultaneous/Organised")
        expLiDARGenOrganised = os.path.join(exp, "LiDARGen/Organised")
        os.system("rm -r " + str(expGTOrganised))
        os.system("mkdir " + str(expGTOrganised))

        os.system("rm -r " + str(expLiDARGenOrganised))
        os.system("mkdir " + str(expLiDARGenOrganised))

        os.system("rm -r " + str(expSimultaneousOrganised))
        os.system("mkdir " + str(expSimultaneousOrganised))
        for counter in range(3):
            current_index = 0
            toGlob = ""
            if(counter == 0):
                toGlob = expGT
            elif(counter == 1):
                toGlob = expSimultaneous
            else:
                toGlob = expLiDARGen
            for file in np.sort(glob(toGlob + '/*.npy')):
                file = np.load(file)
                distance = file[:file.shape[0]//2]
                intensity = file[file.shape[0]//2:]
                combined = np.stack((distance,intensity),1)
                kNums = distance.shape[0] // 6
                for sample in range(kNums * 6):
                    # os.system("rm -r " + str(expSimultaneousOrganised))
                    expGTKVal = os.path.join(expGTOrganised, "k_" + str(sample % kNums))
                    expSimultaneousKVal = os.path.join(expSimultaneousOrganised, "k_" + str(sample % kNums))
                    expLiDARGenKVal = os.path.join(expLiDARGenOrganised, "k_" + str(sample % kNums))
                    toSave = ""
                    if(counter == 0):
                        toSave = expGTKVal
                        os.system("mkdir " + str(expGTKVal))
                    elif(counter == 1):
                        toSave = expSimultaneousKVal
                        os.system("mkdir " + str(expSimultaneousKVal))
                    else:
                        toSave = expLiDARGenKVal
                        os.system("mkdir " + str(expLiDARGenKVal))
                    np.save(os.path.join(toSave, str(sample//kNums + current_index) + '.npy'),combined[sample])
                    # np.save(sample_target[i], str(args.exp) + '/densification_target/' + str(current_index) + '.pth')
                current_index = current_index + 6


#To DO
#Make program that just fucking shifts all the results into the format the existing LiDARGen evaluation code expects - that means batch size 1, etc etc. Check the original iou code to see how it wants it's shit saved