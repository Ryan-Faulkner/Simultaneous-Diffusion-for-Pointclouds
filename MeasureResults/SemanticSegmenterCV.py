import os, glob, pickle
import sys
from glob import glob
sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import metrics.iou as lidargen_iou

if __name__ == '__main__':
    expOG = "DGXDataLiDARGenSettings/*"
    print("hwy")
    listOfExperiments = glob(expOG)
    print(listOfExperiments)
    for experiment in listOfExperiments:
        exp = experiment
        expGTO = os.path.join(exp, "GroundTruth/Organised")
        expBILINEAROrganised = os.path.join(exp, "Input/BILINEAR")
        expNNOrganised = os.path.join(exp, "Input/NN")
        expBICUBICOrganised = os.path.join(exp, "Input/BICUBIC")
        expNSOrganised = os.path.join(exp, "Input/NS")
        expLiDARGenO = os.path.join(exp, "LiDARGen/Organised")
        #How many K's are of relevance?
        numK = len(glob(expLiDARGenO + '/*'))
        for k in range(numK):
            expBILINEAR = os.path.join(expBILINEAROrganised, 'k_'+ str(k))
            expNN = os.path.join(expNNOrganised, 'k_'+ str(k))
            expBICUBIC = os.path.join(expBICUBICOrganised, 'k_'+ str(k))
            expNS = os.path.join(expNSOrganised, 'k_'+ str(k))
            expGT = os.path.join(expGTO, 'k_'+ str(k))
            numSamp = len(glob(expGT + '/*.npy'))
            # os.system("rm -r " + str(expGT) + '/target_rangenet_segmentations')
            # os.system("rm -r " + str(expGT) + '/target_rangenet_fid')
            # os.system("mkdir " + str(expGT) + "/target_rangenet_segmentations")
            # os.system("mkdir " + str(expGT) + "/target_rangenet_fid")

            os.system("rm -r " + str(expBICUBIC) + '/result_rangenet_segmentations')
            os.system("rm -r " + str(expBICUBIC) + '/result_rangenet_fid')
            os.system("mkdir " + str(expBICUBIC) + "/result_rangenet_segmentations")
            os.system("mkdir " + str(expBICUBIC) + "/result_rangenet_fid")

            os.system("rm -r " + str(expBILINEAR) + '/result_rangenet_segmentations')
            os.system("rm -r " + str(expBILINEAR) + '/result_rangenet_fid')
            os.system("mkdir " + str(expBILINEAR) + "/result_rangenet_segmentations")
            os.system("mkdir " + str(expBILINEAR) + "/result_rangenet_fid")

            os.system("rm -r " + str(expNN) + '/result_rangenet_segmentations')
            os.system("rm -r " + str(expNN) + '/result_rangenet_fid')
            os.system("mkdir " + str(expNN) + "/result_rangenet_segmentations")
            os.system("mkdir " + str(expNN) + "/result_rangenet_fid")

            os.system("rm -r " + str(expNS) + '/result_rangenet_segmentations')
            os.system("rm -r " + str(expNS) + '/result_rangenet_fid')
            os.system("mkdir " + str(expNS) + "/result_rangenet_segmentations")
            os.system("mkdir " + str(expNS) + "/result_rangenet_fid")
            counterMax = 1
            if(experiment.split('/')[-1] == "Densification"):
                counterMax = 3
            for counter in range(counterMax):
                targetEXP = ""
                if(counterMax == 1):
                    targetEXP = expNS
                elif(counter == 0):
                    targetEXP = expNN
                elif(counter == 1):
                    targetEXP = expBICUBIC
                else:
                    targetEXP = expBILINEAR

                #This segments the scans in densification_result
                rangenetpp.main("--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/ --output_dir {exp}/result_rangenet_segmentations --kitti_count {numSamples} --frd_dir {exp}/result_rangenet_fid".format(exp=targetEXP, numSamples = numSamp))
                #This segments the scans in densification_target
                #Already done GT so no need to do so again
                # rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/ --output_dir {exp}/target_rangenet_segmentations --kitti_count {numSamples} --frd_dir {exp}/target_rangenet_fid'.format(exp=expGT, numSamples = numSamp))

                print(experiment)
                print(k)
                print(counter)
                iou = lidargen_iou.calculate_iou("{exp}/result_rangenet_segmentations".format(exp=targetEXP), "{exp}/target_rangenet_segmentations".format(exp=expGT))
                print('-------------------------------------------------')
                print('IOU Score: ' + str(iou))
                print('-------------------------------------------------')
                # iou = lidargen_iou.calculate_iou("{exp}/result_rangenet_segmentations".format(exp=expSimultaneous), "{exp}/target_rangenet_segmentations".format(exp=expGT))
                # print('-------------------------------------------------')
                # print('IOU Score: ' + str(iou))
                # print('-------------------------------------------------')

#To DO
#Make program that just fucking shifts all the results into the format the existing LiDARGen evaluation code expects - that means batch size 1, etc etc. Check the original iou code to see how it wants it's shit saved