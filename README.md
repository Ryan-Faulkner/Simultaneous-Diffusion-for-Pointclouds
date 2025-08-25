# simultaneous-diffusion-for-pointclouds

## Running experiments

Built on top of the LiDARGen framework, simply run the main.py within the LiDARGen folder, calling the config file for the experiments desired.

Line.yml generates novel views following the road (not necessarily a perfectly straight line).

Inpainting.yml is used for inpainting, creating synthetic views surrounding the original view and then filling in empty pixels.

Densification.yml runs densification.

There are runners for different variants:

ncsn_runner_basic_simultaneous.py makes no effort to combine or enforce consistency between views, and is the basic method compared against in tables

ncsn_runner_kitti_simultaneous.py enforces all kitti range images are generated with consistency between them, and is for when multiple synthetic viewpoints are desired

ncsn_runner_AllForOne.py does the same thing but is for tasks which only want a single view, so all additional viewpoints exist purely to supplement results in the main one desired (hence the name)

For example, to run experiments for generating novel views:

```
python main.py --ni --sample --config Line.yml
```
The configs

All tests configuration produce results using both our simultaneous sampling, and single-view sampling.

We ran tests with the data in the folder:
'/data/'

Within which was 

1) The raw point clouds in 'KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/'. The data from KITTI-360 can be downloaded from their website https://www.cvlibs.net/datasets/kitti-360/index.php 

2) The mask of frequently empty pixels (converting point clouds to images using the same settings as LiDARGen, e.g. 64x1024 pixel images): 'existTotalLiDARGenSettings'

3) The calibrations for the KITTI-360 dataset, also downloadable from https://www.cvlibs.net/datasets/kitti-360/index.php. These are used to set future vehicle poses as the poses for the synthetic scans during novel view generation (to then use the future scan as a groundtruth comparison).

This code will later be uploaded to github, to be publicly available, both for reproduceability and so others can build off our work as we have the work of LiDARGen (which itself builds upon "Improved Techniques for Training Score-Based Generative Models"). 

The visualisations of the produced point clouds shown in our work were generated using proprietary pointcloud visualisation software, so the code presented here simply generates pointcloud scans in the form of .npy files. Feel free to use whatever visualisation approach is most convenient.
