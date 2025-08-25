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
        expSimultaneousO = os.path.join(exp, "Simultaneous/Organised")
        expLiDARGenO = os.path.join(exp, "LiDARGen/Organised")
        #How many K's are of relevance?
        numK = len(glob(expLiDARGenO + '/*'))
        for k in range(numK):
            expGT = os.path.join(expGTO, 'k_'+ str(k))
            expSimultaneous = os.path.join(expSimultaneousO, 'k_'+ str(k))
            expLiDARGen = os.path.join(expLiDARGenO, 'k_'+ str(k))
            numSamp = len(glob(expGT + '/*.npy'))
            # os.system("rm -r " + str(expGT) + '/target_rangenet_segmentations')
            # os.system("rm -r " + str(expGT) + '/target_rangenet_fid')
            # os.system("mkdir " + str(expGT) + "/target_rangenet_segmentations")
            # os.system("mkdir " + str(expGT) + "/target_rangenet_fid")

            # os.system("rm -r " + str(expLiDARGen) + '/result_rangenet_segmentations')
            # os.system("rm -r " + str(expLiDARGen) + '/result_rangenet_fid')
            # os.system("mkdir " + str(expLiDARGen) + "/result_rangenet_segmentations")
            # os.system("mkdir " + str(expLiDARGen) + "/result_rangenet_fid")

            # os.system("rm -r " + str(expSimultaneous) + '/result_rangenet_segmentations')
            # os.system("rm -r " + str(expSimultaneous) + '/result_rangenet_fid')
            # os.system("mkdir " + str(expSimultaneous) + "/result_rangenet_segmentations")
            # os.system("mkdir " + str(expSimultaneous) + "/result_rangenet_fid")
            # #This segments the scans in densification_result
            # rangenetpp.main("--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/ --output_dir {exp}/result_rangenet_segmentations --kitti_count {numSamples} --frd_dir {exp}/result_rangenet_fid".format(exp=expLiDARGen, numSamples = numSamp))
            # rangenetpp.main("--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/ --output_dir {exp}/result_rangenet_segmentations --kitti_count {numSamples} --frd_dir {exp}/result_rangenet_fid".format(exp=expSimultaneous, numSamples = numSamp))
            # #This segments the scans in densification_target
            # rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/ --output_dir {exp}/target_rangenet_segmentations --kitti_count {numSamples} --frd_dir {exp}/target_rangenet_fid'.format(exp=expGT, numSamples = numSamp))

            print(experiment)
            print(k)
            iou = lidargen_iou.calculate_iou("{exp}/result_rangenet_segmentations".format(exp=expLiDARGen), "{exp}/target_rangenet_segmentations".format(exp=expGT))
            print('-------------------------------------------------')
            print('IOU Score: ' + str(iou))
            print('-------------------------------------------------')
            iou = lidargen_iou.calculate_iou("{exp}/result_rangenet_segmentations".format(exp=expSimultaneous), "{exp}/target_rangenet_segmentations".format(exp=expGT))
            print('-------------------------------------------------')
            print('IOU Score: ' + str(iou))
            print('-------------------------------------------------')

#To DO
#Make program that just fucking shifts all the results into the format the existing LiDARGen evaluation code expects - that means batch size 1, etc etc. Check the original iou code to see how it wants it's shit saved