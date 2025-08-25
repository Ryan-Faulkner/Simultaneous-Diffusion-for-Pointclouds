import os, glob, pickle
import sys
from glob import glob
sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import metrics.iou as lidargen_iou
import numpy as np
if __name__ == '__main__':
    expOG = "DGXDataLiDARGenSettings/SceneCompletionW*"
    print("hwy")
    listOfExperiments = glob(expOG)
    print(listOfExperiments)
    for experiment in listOfExperiments:
        exp = experiment
        # expGT = os.path.join(exp, "GroundTruth")
        expSimultaneous = os.path.join(exp, "image_samples/images")
        # expLiDARGen = os.path.join(exp, "LiDARGen")
        # expGTOrganised = os.path.join(exp, "GroundTruth/Organised")
        expSimultaneousOrganised = os.path.join(exp, "Organised")
        # expLiDARGenOrganised = os.path.join(exp, "LiDARGen/Organised")
        # os.system("rm -r " + str(expGTOrganised))
        # os.system("mkdir " + str(expGTOrganised))

        # os.system("rm -r " + str(expLiDARGenOrganised))
        # os.system("mkdir " + str(expLiDARGenOrganised))

        os.system("rm -r " + str(expSimultaneousOrganised))
        os.system("mkdir " + str(expSimultaneousOrganised))
        for counter in range(1,2):
            current_index = 0
            finalFolder = expSimultaneousOrganised
            os.system("rm -r " + str(finalFolder))
            os.system("mkdir " + str(finalFolder))
            toGlob = os.path.join(exp, "image_samples/images/" + str(1) + "*Masked_completion_897.pth.npy")
            fileCounter = 0
            for file in np.sort(glob(toGlob)):
                filename = file.split('/')[-1]
                origins = filename[:-len("Masked_completion_897.pth.npy")] + "ORIGINS_897.pth.npy"
                filestart = file[:-len(filename)]
                originLoad = filestart + str(0) + origins[1:]
                finalName = filestart + str(counter) + filename[1:]
                print(finalName)
                file = np.load(finalName)
                originFile = np.load(originLoad)
                distance = file[:file.shape[0]//2]
                intensity = file[file.shape[0]//2:]
                combined = np.stack((distance,intensity),1)
                kNums = distance.shape[0]
                for sample in range(kNums):
                    # os.system("rm -r " + str(expSimultaneousOrganised))
                    # expGTKVal = os.path.join(expGTOrganised, "k_" + str(sample % kNums))
                    # expSimultaneousKVal = os.path.join(expSimultaneousOrganised, "k_" + str(sample % kNums))
                    # expLiDARGenKVal = os.path.join(expLiDARGenOrganised, "k_" + str(sample % kNums))
                    toSave = os.path.join(finalFolder,filename[2:-len("_Masked_completion_897.pth.npy")])
                    # toSaveOrigin = os.path.join(finalFolder,str(fileCounter))
                    os.system("mkdir " + str(toSave))
                    os.system("mkdir " + str(toSave) + "/Origins")
                    np.save(os.path.join(toSave, str(sample % kNums) + '.npy'),combined[sample])
                    np.save(os.path.join(toSave + '/Origins', str(sample % kNums) + '.npy'),originFile[sample])
                    # np.save(sample_target[i], str(args.exp) + '/densification_target/' + str(current_index) + '.pth')
                current_index = current_index + 6
                fileCounter += 1


#To DO
#Make program that just fucking shifts all the results into the format the existing LiDARGen evaluation code expects - that means batch size 1, etc etc. Check the original iou code to see how it wants it's shit saved