import numpy as np
from glob import glob
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.dsm import anneal_dsm_score_estimation_with_mask

import torch.nn.functional as F
import logging
import torch
import os
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest, NCSN_LiDAR, NCSN_LiDAR_small
from models.ncsn import NCSN, NCSNdeeper
from datasets import get_dataset, data_transform, inverse_data_transform
from losses import get_optimizer
from models import (anneal_Langevin_dynamics,
                    anneal_Langevin_dynamics_inpainting,
                    anneal_Langevin_dynamics_densification,
                    anneal_Langevin_dynamics_inpainting_simultaneous_basic,
                    anneal_Langevin_dynamics_inpainting_simultaneous_second_method)
from models import get_sigmas
from models.ema import EMAHelper
#from .nvs import KITTINVS, novel_view_synthesis

__all__ = ['NCSNRunnerBasicSimultaneous']


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == 'KITTI' or config.data.dataset == 'lidar':
        #return NCSN_LiDAR(config).to(config.device)
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'KITTI360':
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'HDVMineGenerate':
        #return NCSN_LiDAR(config).to(config.device)
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'HDVMinePreGenerated':
        #return NCSN_LiDAR(config).to(config.device)
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'HDVMineGenerateFromInvidivualScans':
        return NCSN_LiDAR_small(config).to(config.device)
    elif config.data.dataset == 'HDVMinePreGenerated8Batch':
        return NCSN_LiDAR_small(config).to(config.device)

class MySampler():
    def __init__(self, num_batches, batch_size, random = True):
        self.n_batches = num_batches
        self.batch_size = batch_size
        self.random = random

    def __iter__(self):
        # print("the fucking number")
        # print(self.n_batches)
        numbers = np.arange(self.n_batches)
        if(self.random):
            np.random.shuffle(numbers)
        # print(numbers)
        batches = []
        for chosenNum in range(self.n_batches):
            for i in range(self.batch_size):
                batches.append((numbers[chosenNum] * self.batch_size) + i)
            # batches.append(batch)
        return iter(batches)

class NCSNRunnerBasicSimultaneous():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):
        dataset, test_dataset = get_dataset(self.args, self.config)
        # trainingSize = len(glob('PreGenFinal/Depth/*')) #* 6 // 10
        # valSize = len(glob('PreGenFinalValWithSky/Depth/*'))# * 2 // 10
        trainingSize = len(glob('/data/PreGenFinal/PreGenFinal/Depth/*')) * 6 // 10
        valSize = len(glob('/data/PreGenFinalVal/Depth/*')) * 2 // 10

        trainingSampler = MySampler(num_batches = trainingSize ,batch_size=self.config.training.batch_size)
        valSampler = MySampler(num_batches = valSize ,batch_size=self.config.training.batch_size)
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, sampler=trainingSampler,
                                num_workers=self.config.data.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, sampler=valSampler,
                                 num_workers=self.config.data.num_workers, drop_last=True)
        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size * self.config.data.image_width * self.config.data.channels

        tb_logger = self.config.tb_logger

        score = get_model(self.config)
        # print(score)

        score = torch.nn.DataParallel(score)
        optimizer = get_optimizer(self.config, score.parameters())

        start_epoch = 0
        step = 0
        trueStep = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        if self.args.resume_training:
            # states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'))
            # states = torch.load('diffusionNet/checkpoint_Sigma1_NoLog.pth')
            states = torch.load('diffusionNet/checkpoint_148.pth')
            # print(len(states[0]))
            # print(len(score.state_dict()))
            statesToLoad = {}
            for key in states[0]:
                # print(states[0][key].shape)
                # print(score.state_dict()[key].shape)
                if(states[0][key].shape== score.state_dict()[key].shape):
                    statesToLoad[key] = states[0][key]
            #strict=False lets me ignore that some layers are missing and remain randomised
            score.load_state_dict(statesToLoad, strict=False)
            # ### Make sure we can resume with different eps
            # states[1]['param_groups'][0]['eps'] = self.config.optim.eps
            # optimizer.load_state_dict(states[1])
            # start_epoch = states[2]
            # step = states[3]
            # if self.config.model.ema:
            #     ema_helper.load_state_dict(states[4])

        sigmas = get_sigmas(self.config)

        if self.config.training.log_all_sigmas:
            ### Commented out training time logging to save time.
            test_loss_per_sigma = [None for _ in range(len(sigmas))]

            def hook(loss, labels):
                # for i in range(len(sigmas)):
                #     if torch.any(labels == i):
                #         test_loss_per_sigma[i] = torch.mean(loss[labels == i])
                pass

            def tb_hook():
                # for i in range(len(sigmas)):
                #     if test_loss_per_sigma[i] is not None:
                #         tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                #                              global_step=step)
                pass

            def test_hook(loss, labels):
                for i in range(len(sigmas)):
                    if torch.any(labels == i):
                        test_loss_per_sigma[i] = torch.mean(loss[labels == i])

            def test_tb_hook():
                for i in range(len(sigmas)):
                    if test_loss_per_sigma[i] is not None:
                        tb_logger.add_scalar('test_loss_sigma_{}'.format(i), test_loss_per_sigma[i],
                                             global_step=step)

        else:
            hook = test_hook = None

            def tb_hook():
                pass

            def test_tb_hook():
                pass

        maxTimeStepReachable = 1
        for epoch in range(start_epoch, self.config.training.n_epochs):
            #X is point cloud, mask is mask
            for i, (X, mask, sky) in enumerate(dataloader):
                step += 1
                X = X.to(self.config.device)
                mask = mask.to(self.config.device)
                sky = sky.to(self.config.device)
                X = data_transform(self.config, X)
                originalX = torch.clone(X)
                labels = torch.full((X.shape[0],),0, device=X.device)
                used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                noise = torch.randn_like(X) * used_sigmas
                #Add max noise to any image sections I don't trust to effectively intialise them
                X = X + noise*torch.logical_not(mask).int() 
                for timestep in range(maxTimeStepReachable):
                    labels = torch.full((X.shape[0],),timestep, device=X.device) 
                    used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
                    noise = torch.randn_like(X) * used_sigmas
                    X = X + noise * mask
                    trueStep += 1
                    X = X.detach().requires_grad_()
                    score.train()
                    # labels = torch.full((X.shape[0],),timestep, device=X.device)
                    #No need to data_tranform the mask, as that shouldn't have noise added, etc

                    loss, grad = anneal_dsm_score_estimation_with_mask(score, X, used_sigmas, noise, mask, sky, sigmas, labels,
                                                       self.config.training.anneal_power,
                                                       hook)
                    for s in range(self.config.sampling.n_steps_each):
                    
                        step_size = self.config.sampling.step_lr * (sigmas[timestep] / sigmas[-1]) ** 2    
                        # grad_likelihood = -mask * (X - originalX) # - 0.05*(1-mask)*(x_mod - raw_interp)

                        noise2 = torch.randn_like(X)
                        # grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                        # grad_likelihood_norm = torch.norm(grad_likelihood.view(grad.shape[0], -1), dim=-1).mean()
                        # noise_norm = torch.norm(noise2.view(noise2.shape[0], -1), dim=-1).mean()
                        prediction = X + step_size * grad + noise2 * torch.sqrt(step_size * 2)

                        # image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                        # snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                        # grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                        # images.append(x_mod.to('cpu'))

                        X = originalX * mask + prediction * torch.logical_not(mask).int() 
     
                    tb_logger.add_scalar('loss', loss, global_step=trueStep)
                    tb_hook()

                    logging.info("step: {}, timestep: {}, loss: {}".format(step, timestep, loss.item()))

                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(score.parameters(), 10.0, 'inf')
                    optimizer.step()

                    if self.config.model.ema:
                        ema_helper.update(score)

                    if step >= self.config.training.n_iters:
                        return 0

                    if step % 100 == 0 and timestep == 0: #100
                        if self.config.model.ema:
                            test_score = ema_helper.ema_copy(score)
                        else:
                            test_score = score

                        test_score.eval()
                        try:
                            test_X, test_y, test_sky = next(test_iter)
                        except StopIteration:
                            test_iter = iter(test_loader)
                            test_X, test_y, test_sky = next(test_iter)

                        test_X = test_X.to(self.config.device)
                        test_y = test_y.to(self.config.device)
                        test_sky = test_sky.to(self.config.device)
                        test_X = data_transform(self.config, test_X) 
                        lossTotal = 0
                        with torch.no_grad():
                            original_test_X = torch.clone(test_X)
                            test_labels = torch.full((test_X.shape[0],),0, device=test_X.device)
                            test_used_sigmas = sigmas[test_labels].view(test_X.shape[0], *([1] * len(test_X.shape[1:])))
                            test_noise = torch.randn_like(test_X) * test_used_sigmas
                            #Add max noise to any image sections I don't trust to effectively intialise them
                            test_X = test_X + test_noise*torch.logical_not(test_y).int() 
                            for testTimestep in range(maxTimeStepReachable):
                                test_labels = torch.full((test_X.shape[0],),testTimestep, device=test_X.device) 
                                test_used_sigmas = sigmas[test_labels].view(test_X.shape[0], *([1] * len(test_X.shape[1:])))
                                test_noise = torch.randn_like(test_X) * test_used_sigmas
                                test_X = test_X + test_noise * test_y
                                test_dsm_loss, grad = anneal_dsm_score_estimation_with_mask(test_score, test_X, test_used_sigmas, test_noise, test_y, test_sky, sigmas, test_labels,
                                                                            self.config.training.anneal_power,
                                                                            hook=test_hook)
                                # test_dsm_loss, grad = anneal_dsm_score_estimation_with_mask(test_score, original_test_X, None, None, test_y, test_sky, sigmas, None,
                                #                                             self.config.training.anneal_power,
                                #                                             hook=test_hook)
                                lossTotal += test_dsm_loss
                                for testStep in range(self.config.sampling.n_steps_each):
                    
                                    step_size = self.config.sampling.step_lr * (sigmas[testTimestep] / sigmas[-1]) ** 2    

                                    noise2 = torch.randn_like(test_X)
                                    prediction = test_X + step_size * grad + noise2 * torch.sqrt(step_size * 2)

                                    test_X = original_test_X * test_y + prediction * torch.logical_not(test_y).int() 
                            test_dsm_loss = lossTotal / maxTimeStepReachable
                            tb_logger.add_scalar('test_loss', test_dsm_loss, global_step=trueStep)
                            test_tb_hook()
                            logging.info("step: {}, test_loss: {}".format(step, test_dsm_loss.item()))

                            del test_score

                    if trueStep % 20 == 0:
                        if(maxTimeStepReachable < len(sigmas)):
                            maxTimeStepReachable += 1
                    if trueStep % self.config.training.snapshot_freq == 0:
                        states = [
                            score.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        torch.save(states, os.path.join(self.args.log_path, 'checkpoint_{}.pth'.format(step)))
                        torch.save(states, os.path.join(self.args.log_path, 'checkpoint.pth'))

                        if self.config.training.snapshot_sampling:
                            if self.config.model.ema:
                                test_score = ema_helper.ema_copy(score)
                            else:
                                test_score = score

                            test_score.eval()

                            ## Different part from NeurIPS 2019.
                            ## Random state will be affected because of sampling during training time.
                            init_samples = torch.rand(36, self.config.data.channels,
                                                      self.config.data.image_size, self.config.data.image_width,
                                                      device=self.config.device)
                            init_samples = data_transform(self.config, init_samples)

                            all_samples = anneal_Langevin_dynamics(init_samples, test_score, sigmas.cpu().numpy(),
                                                                   self.config.sampling.n_steps_each,
                                                                   self.config.sampling.step_lr,
                                                                   final_only=True, verbose=True,
                                                                   denoise=self.config.sampling.denoise)

                            sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                          self.config.data.image_size,
                                                          self.config.data.image_width)

                            sample = inverse_data_transform(self.config, sample)

                            torch.save(sample, os.path.join(self.args.log_sample_path, 'samples_{}.pth'.format(step)))
                            torch.save(all_samples, os.path.join(self.args.log_sample_path, 'samples_all_{}.pth'.format(step)))

                            if sample.dim() == 4 and sample.size(1) == 2:  # two-channel images
                                sample = sample.transpose(1, 0)
                                sample = sample.reshape((sample.size(1)*sample.size(0), 1, sample.size(2), sample.size(3)))
                                sample = torch.cat((sample, sample, sample), 1)

                            image_grid = make_grid(sample, 6)
                            save_image(image_grid,
                                       os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(step)))

                            del test_score
                            del all_samples

    def nvs(self):
        '''
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        dataset, _ = get_dataset(self.args, self.config)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                num_workers=4)
        score.eval()

        dataset = KITTINVS('/mnt/data/KITTI-360', seq_number = 0)

        # compute common mask
        range_sum = np.zeros((64, 1024))
        for idx in range(100):
            lidar_range_src, v2w_src = dataset[idx]
            range_sum = range_sum + lidar_range_src[0]
        common_mask = range_sum < 1e-2

        for src_idx in [100, 200, 300, 400, 500, 1000, 1500, 2000]:
            batch_size = 24
            batch = []
            batch_gt = []
            for tgt_idx in range(src_idx + 1, src_idx + batch_size + 1):
                lidar_range_src, v2w_src = dataset[src_idx]
                lidar_range_tgt, v2w_tgt = dataset[tgt_idx]
                real, real_log, xyz_src_tgt, xyz_src_tgt_world = novel_view_synthesis(lidar_range_src, v2w_src, lidar_range_tgt, v2w_tgt, common_mask)
                batch.append(real_log)
                batch_gt.append(lidar_range_tgt)
                # pcd_src_tgt = o3d.geometry.PointCloud()
                # pcd_src_tgt.points = o3d.utility.Vector3dVector(xyz_src_tgt_world)
                # pcd_src_tgt.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([pcd_src_tgt])

            batch = np.stack(batch, axis = 0)
            samples = torch.tensor(batch)
            batch_gt = np.stack(batch_gt, axis = 0)
            samples_gt = torch.tensor(batch_gt)

            samples = samples.to(self.config.device)
            samples = data_transform(self.config, samples)

            samples_gt = samples_gt.to(self.config.device)
            samples_gt = data_transform(self.config, samples_gt)

            init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                        self.config.data.image_size, self.config.data.image_width,
                                        device=self.config.device)
            init_samples = data_transform(self.config, init_samples)


            for grad_ref in [1, 2, 0.5, 0.2]:

                all_samples, targets = anneal_Langevin_dynamics_nvs(init_samples, samples, samples_gt, score, sigmas,
                                                    self.config.sampling.n_steps_each,
                                                    self.config.sampling.step_lr,
                                                    denoise=self.config.sampling.denoise,
                                                    grad_ref=grad_ref,
                                                    sampling_step=4)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples[-3:]), total=len(all_samples[-3:]),
                                            desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'nvs_image_grid_{}_{}_{}.png'.format(grad_ref, src_idx, i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'nvs_samples_{}_{}_{}.pth'.format(grad_ref, src_idx, i)))

                    sample = targets[0]
                    sample = sample.view(sample.shape[0], self.config.data.channels,
                                            self.config.data.image_size,
                                            self.config.data.image_width)
                    sample = inverse_data_transform(self.config, sample)
                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    save_image(image_grid, os.path.join(self.args.image_folder, 'nvs_ref_grid_{}.png'.format(src_idx)))
                    torch.save(sample, os.path.join(self.args.image_folder, 'nvs_ref_{}.pth'.format(src_idx)))

            sample = samples_gt
            sample = sample.view(sample.shape[0], self.config.data.channels,
                                    self.config.data.image_size,
                                    self.config.data.image_width)
            sample = inverse_data_transform(self.config, sample)
            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
            save_image(image_grid, os.path.join(self.args.image_folder, 'nvs_gt_grid_{}.png'.format(src_idx)))
            torch.save(sample, os.path.join(self.args.image_folder, 'nvs_gt_{}.pth'.format(src_idx)))
        '''
        return


    def sample(self):
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            # states = torch.load('secondSession.pth',
            # states = torch.load('diffusionNet/checkpoint_Sigma1_NoLog.pth',
            #                     map_location=self.config.device)
            #Normal Settings - Log and Sigma 50
            states = torch.load('diffusionNet/checkpoint_148.pth',
                                map_location=self.config.device)
            #Log and Sigma 1
            # states = torch.load('diffusionNet/LogSigma1.pth',
            #                     map_location=self.config.device)
            # states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
            #                     map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        train_dataset, test_dataset = get_dataset(self.args, self.config)
        dataset = test_dataset

        # valSize = len(glob('PreGenFinalValWithSkyAndIndices/Depth/*'))# * 2 // 10
        #So this is the actual size but
        # valSize = len(glob(os.path.join('RawScans', 'Scans/*.npy'))) * 2 // 10 - 1
        #for my purposes it needs to be the number of mods I'm doing
        valSize = 8

        valSampler = MySampler(num_batches = valSize ,batch_size=self.config.sampling.batch_size, random=False)
        dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, sampler=valSampler,
                                 num_workers=self.config.data.num_workers, drop_last=True)

        # dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
        #                         num_workers=self.config.data.num_workers)

        score.eval()

        if not self.config.sampling.fid:
            if self.config.sampling.inpainting:
                data_iter = iter(dataloader)
                gradRef = 1
                scanCounter = 1
                for batchesToDo in range(8):
                    scanCounter += 1
                    # refer_images_full,refer_mask_full, sky, refer_indices, modScale = next(data_iter)
                    #do this again because I don't like the first image for quick testing
                    refer_images_full,refer_mask_full, sky, refer_indices, modScale = next(data_iter)
                    #Just do this same 8 the whole way down plz
                    for doThis in range(6):
                        #First 3, first method:
                        #Second 3, second method: 
                        subPart = 0
                        sample_full = []
                        shared_full = []
                        shared_intial = []
                        shared_second = []
                        shared_third = []
                        numSubsections = 1

                        # gradRef = gradRef - 0.1
                        #correlation starts at 1 and decreases until it reaches 0 at doThis = 10
                        # correlation_coArray = [0.1,0.5,0,0.05]
                        # gradRefArray = [1,0.5,0.25]
                        # minStart = [0,200,150,0,200,150,0,200,150,0,200,150]
                        # settingArray = [4,4,4,4,4,1,1,1,1,1]
                        # correlation_co = correlation_coArray[doThis]
                        # gradRef = gradRefArray[doThis]
                        # startStep = minStart[doThis]
                        #Keep it simple first few
                        startStep = 20
                        correlation_co = 0.05
                        # setting = settingArray[doThis]
                        setting = 4
                        # if(doThis > 2):
                        # if(doThis == 0):
                            # The proper correlation_co = 0.05 I should've done last time
                        #Now it's time for the GradRef Testing~~~~~~~~~~~~~~~~~~~~~~
                        if(doThis == 1):
                            correlation_co = 0.05
                            gradRef = 0.9
                            #Scaled correlation co up to 1
                            # setting = 5
                        if(doThis == 2):
                            correlation_co = 0.1
                            gradRef = 0.9
                            #Scaled correlation co up to 0.5
                            # setting = 6
                        if(doThis == 3):
                            correlation_co = 0.05
                            gradRef = 0.5
                            #Default method - no correlation method done
                            correlation_co = 0
                        if(doThis == 4): 
                            correlation_co = 0.1
                            gradRef = 0.5
                            #Second Method with Hard GroundTruth Reset, this time with the logarithmic Unet
                            # setting = 4
                        if(doThis == 5):
                            correlation_co = 0.1
                            gradRef = 0.25
                            #Second Method better suited for aerial origins... somehow
                            #I guess let's add something to push the median of predicted points closer to median of GT points
                            #Let's literally just shift ALL the predicted points ~20% of the way towards the real median at each timestep and see what happens
                            #With a hard GroundTruth otherwise
                            # setting = 7
                        #need to multiply generator Too
                        # originMultiplierArray = [1,2,3,4,5,6,7,8]
                        # originMultiplierArray = [1,1,1,1,1,1,1,1]
                        if(torch.max(modScale) != torch.min(modScale)):
                            print("scales are fucked")
                            continue
                        modifiersToGive = (torch.from_numpy(np.array(self.config.data.modifications)) * modScale[0]).to(self.config.device)
                        print("scale used")
                        print(modScale)

                        subsectionSize = self.config.data.image_size // numSubsections
                        refer_images_for_diffusion = refer_images_full[:,:,subPart*subsectionSize:(subPart+1)*subsectionSize,:].float().to(self.config.device)
                        refer_mask = refer_mask_full[:,:,subPart*subsectionSize:(subPart+1)*subsectionSize,:].int().to(self.config.device)
                        width = int(self.config.sampling.batch_size)
                        init_samples = torch.rand(width, self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_width,
                                                  device=self.config.device)
                        init_samples = data_transform(self.config, init_samples)
                        # print("why do you hate me")
                        # print(init_samples.shape)
                        # print(refer_images_for_diffusion.shape)
                        # print(refer_mask.shape)

                        refer_images = refer_images_full * refer_mask_full
                        refer_images = inverse_data_transform(self.config, refer_images)

                        if refer_images.dim() == 4 and refer_images.size(1) == 2:  # two-channel images
                            refer_images = refer_images.transpose(1, 0)
                            refer_images = refer_images.reshape((refer_images.size(1)*refer_images.size(0), 1, refer_images.size(2), refer_images.size(3)))
                            refer_images = torch.cat((refer_images, refer_images, refer_images), 1)

                        image_grid = make_grid(refer_images, int(np.sqrt(self.config.sampling.batch_size)))
                        if(doThis == 0):
                            save_image(image_grid, os.path.join(self.args.image_folder,
                                                                str(doThis) + '_' + str(batchesToDo) + '_GT_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                            np.save(os.path.join(self.args.image_folder,
                                                            str(doThis) + '_' + str(batchesToDo) + '_GT_completion_{}.pth'.format(self.config.sampling.ckpt_id)),refer_images.cpu().detach().numpy())
                        #Ok so the crux of it is that for each batch of size 8 I need to create a shared point cloud, reproject to 2D, and add a gradient based on where they differ.
                        #init_samples starts as pure random noise
                        #the function incrementally calculates gradients, and checks against the masked reference image to ensure output matches the known pixels
                        #Really I should do this all inside of the inpainting function itself
                        #First method
                        #This continue is for if just confirming the Ground Truths look correct before running it
                        # continue
                        if(doThis < 8):
                            all_outputs, all_targets, all_shared = anneal_Langevin_dynamics_inpainting_simultaneous_basic(init_samples, refer_images_for_diffusion, refer_mask , sky,refer_indices, startStep, setting, score, sigmas, modifiersToGive,
                                                                    self.config.sampling.n_steps_each,
                                                                    self.config.sampling.step_lr,
                                                                    denoise=self.config.sampling.denoise,
                                                                    grad_ref=gradRef,
                                                                    correlation_coefficient = correlation_co,
                                                                    sampling_step=4)
                        #second method
                        else:
                            all_outputs, all_targets, all_shared = anneal_Langevin_dynamics_inpainting_simultaneous_second_method(init_samples, refer_images_for_diffusion, refer_mask , sky,refer_indices, startStep, setting, score, sigmas, modifiersToGive,
                                                                    self.config.sampling.n_steps_each,
                                                                    self.config.sampling.step_lr,
                                                                    denoise=self.config.sampling.denoise,
                                                                    grad_ref=gradRef,
                                                                    correlation_coefficient = correlation_co,
                                                                    sampling_step=4)
                        # print("whyyyy")
                        # print(refer_images.shape)
                        # print(sum(refer_images))
                        # torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
                        # refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1,
                        #                                                                                      *refer_images.shape[
                        #                                                                                       1:])
                        sample_full.append(all_outputs[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_width))
                        # shared_full.append(all_shared[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                        #                               self.config.data.image_size,
                        #                               self.config.data.image_width))
                        shared_intial.append(all_outputs[-2].view(self.config.sampling.batch_size, self.config.data.channels,
                                                      self.config.data.image_size,
                                                      self.config.data.image_width))
                        # shared_second.append(all_shared[-3].view(1, self.config.data.channels,
                        #                               240,#self.config.data.image_size,
                        #                               self.config.data.image_width))
                        # shared_third.append(all_shared[-2].view(self.config.sampling.batch_size, self.config.data.channels,
                        #                               self.config.data.image_size,
                        #                               self.config.data.image_width))


                        sample = torch.cat(sample_full,2)
                        # sample_shared = torch.cat(shared_full,2)
                        shared_intial = torch.cat(shared_intial,2)# * sigmas[-1] / sigmas[0]
                        # shared_second = torch.cat(shared_second,2)# * sigmas[-1] / sigmas[20]
                        # shared_third = torch.cat(shared_third,2)# * sigmas[-1] / sigmas[110] 
                        # sample = sample_full[-1]

                        # save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'))

                        if not self.config.sampling.final_only:
                            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                                sample = sample.view(self.config.sampling.batch_size, self.config.data.channels,
                                                     self.config.data.image_size,
                                                     self.config.data.image_width)

                                sample = inverse_data_transform(self.config, sample)

                                if sample.dim() == 4 and sample.size(1) == 2:  # two-channel images
                                    sample = sample.transpose(1, 0)
                                    sample = sample.reshape((sample.size(1)*sample.size(0), 1, sample.size(2), sample.size(3)))
                                    sample = torch.cat((sample, sample, sample), 1)

                                image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                                save_image(image_grid, os.path.join(self.args.image_folder, str(doThis) + '_' + str(batchesToDo) + '_image_grid_{}.png'.format(i)))
                                np.save(sample.cpu().detach().numpy(), os.path.join(self.args.image_folder, str(doThis) +  '_' + str(batchesToDo) +'_completion_{}.pth'.format(i)))
                        else:
                            # sample = all_outputs[-1].view(self.config.sampling.batch_size, self.config.data.channels,
                            #                               self.config.data.image_size,
                            #                               self.config.data.image_width)

                            sample = inverse_data_transform(self.config, sample)
                            # sample_shared = inverse_data_transform(self.config, sample_shared) 
                            sample_initial = inverse_data_transform(self.config, shared_intial) 
                            # sample_second = inverse_data_transform(self.config, shared_second)
                            # sample_third = inverse_data_transform(self.config, shared_third)
                            maskedSample = sample * sky
                            # maskedShared = sample_shared * sky

                            if sample.dim() == 4 and sample.size(1) == 2:  # two-channel images
                                sample = sample.transpose(1, 0)
                                sample = sample.reshape((sample.size(1)*sample.size(0), 1, sample.size(2), sample.size(3)))
                                sample = torch.cat((sample, sample, sample), 1)
                                # sample_shared = sample_shared.transpose(1, 0)
                                # sample_shared = sample_shared.reshape((sample_shared.size(1)*sample_shared.size(0), 1, sample_shared.size(2), sample_shared.size(3)))
                                # sample_shared = torch.cat((sample_shared, sample_shared, sample_shared), 1)

                                sample_initial = sample_initial.transpose(1, 0)
                                sample_initial = sample_initial.reshape((sample_initial.size(1)*sample_initial.size(0), 1, sample_initial.size(2), sample_initial.size(3)))
                                sample_initial = torch.cat((sample_initial, sample_initial, sample_initial), 1)

                                # sample_second = sample_second.transpose(1, 0)
                                # sample_second = sample_second.reshape((sample_second.size(1)*sample_second.size(0), 1, sample_second.size(2), sample_second.size(3)))
                                # sample_second = torch.cat((sample_second, sample_second, sample_second), 1)

                                # sample_third = sample_third.transpose(1, 0)
                                # sample_third = sample_third.reshape((sample_third.size(1)*sample_third.size(0), 1, sample_third.size(2), sample_third.size(3)))
                                # sample_third = torch.cat((sample_third, sample_third, sample_third), 1)

                                maskedSample = maskedSample.transpose(1, 0)
                                maskedSample = maskedSample.reshape((maskedSample.size(1)*maskedSample.size(0), 1, maskedSample.size(2), maskedSample.size(3)))
                                maskedSample = torch.cat((maskedSample, maskedSample, maskedSample), 1)

                                # maskedShared = maskedShared.transpose(1, 0)
                                # maskedShared = maskedShared.reshape((maskedShared.size(1)*maskedShared.size(0), 1, maskedShared.size(2), maskedShared.size(3)))
                                # maskedShared = torch.cat((maskedShared, maskedShared, maskedShared), 1)

                            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                            save_image(image_grid, os.path.join(self.args.image_folder,
                                                                str(doThis) + '_' + str(batchesToDo) + '_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                            np.save(os.path.join(self.args.image_folder,
                                                            str(doThis) + '_' + str(batchesToDo) + '_completion_{}.pth'.format(self.config.sampling.ckpt_id)),sample.cpu().detach().numpy())
                            image_grid = make_grid(maskedSample, int(np.sqrt(self.config.sampling.batch_size)))
                            save_image(image_grid, os.path.join(self.args.image_folder,
                                                                str(doThis) + '_' + str(batchesToDo) + '_Masked_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                            np.save(os.path.join(self.args.image_folder,
                                                            str(doThis) + '_' + str(batchesToDo) + '_Masked_completion_{}.pth'.format(self.config.sampling.ckpt_id)),maskedSample.cpu().detach().numpy())
                            # #shared image
                            # image_grid = make_grid(sample_shared, int(np.sqrt(self.config.sampling.batch_size)))
                            # save_image(image_grid, os.path.join(self.args.image_folder,
                            #                                     str(doThis) + '_' + str(batchesToDo) + '_Shared_image_grid_final{}.png'.format(self.config.sampling.ckpt_id)))
                            # np.save(os.path.join(self.args.image_folder,
                            #                                 str(doThis) + '_' + str(batchesToDo) + '_Shared_completion_final{}.pth'.format(self.config.sampling.ckpt_id)),sample_shared.cpu().detach().numpy())

                            # image_grid = make_grid(sample_initial, int(np.sqrt(self.config.sampling.batch_size)))
                            # save_image(image_grid, os.path.join(self.args.image_folder,
                            #                                     str(doThis) + '_' + str(batchesToDo) + '_Shared_image_grid_initial{}.png'.format(self.config.sampling.ckpt_id)))
                            # np.save(os.path.join(self.args.image_folder,
                            #                                 str(doThis) + '_' + str(batchesToDo) + '_Shared_completion_initial{}.pth'.format(self.config.sampling.ckpt_id)),sample_initial.cpu().detach().numpy())

                            # image_grid = make_grid(sample_second, int(np.sqrt(self.config.sampling.batch_size)))
                            # save_image(image_grid, os.path.join(self.args.image_folder,
                            #                                     str(doThis) + '_' + str(batchesToDo) + '_Shared_image_grid_second{}.png'.format(self.config.sampling.ckpt_id)))
                            # np.save(os.path.join(self.args.image_folder,
                            #                                 str(doThis) + '_' + str(batchesToDo) + '_Shared_completion_second{}.pth'.format(self.config.sampling.ckpt_id)),sample_second.cpu().detach().numpy())

                            # image_grid = make_grid(sample_third, int(np.sqrt(self.config.sampling.batch_size)))
                            # save_image(image_grid, os.path.join(self.args.image_folder,
                            #                                     str(doThis) + '_Shared_image_grid_third{}.png'.format(self.config.sampling.ckpt_id)))
                            # np.save(os.path.join(self.args.image_folder,
                            #                                 str(doThis) + '_Shared_completion_third{}.pth'.format(self.config.sampling.ckpt_id)),sample_third.cpu().detach().numpy())

                            # image_grid = make_grid(maskedShared, int(np.sqrt(self.config.sampling.batch_size)))
                            # save_image(image_grid, os.path.join(self.args.image_folder,
                            #                                     str(doThis) + '_' + str(batchesToDo) + '_Masked_Shared_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                            # np.save(os.path.join(self.args.image_folder,
                            #                                 str(doThis) + '_' + str(batchesToDo) + '_Masked_Shared_completion_{}.pth'.format(self.config.sampling.ckpt_id)),maskedShared.cpu().detach().numpy())

            
            elif self.config.sampling.densification:
                data_iter = iter(dataloader)
                samples, masks, _ = next(data_iter)
                samples = samples.to(self.config.device)
                samples = data_transform(self.config, samples)

                init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                            self.config.data.image_size, self.config.data.image_width,
                                            device=self.config.device)
                init_samples = data_transform(self.config, init_samples)

                #if self.config.sampling.diverse:
                #    samples[1:] = samples[2]
                #    print("same init guidance")


                for grad_ref in [1]:

                    all_samples, targets = anneal_Langevin_dynamics_densification(init_samples, samples, score, sigmas,
                                                        self.config.sampling.n_steps_each,
                                                        self.config.sampling.step_lr,
                                                        denoise=self.config.sampling.denoise,
                                                        grad_ref=grad_ref,
                                                        sampling_step=4)

                    if not self.config.sampling.final_only:
                        for i, sample in tqdm.tqdm(enumerate(all_samples[-3:]), total=len(all_samples[-3:]),
                                                desc="saving image samples"):
                            sample = sample.view(sample.shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_width)

                            sample = inverse_data_transform(self.config, sample)

                            image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                            save_image(image_grid, os.path.join(self.args.image_folder, 'densify_image_grid_{}_{}.png'.format(grad_ref, i)))
                            torch.save(sample, os.path.join(self.args.image_folder, 'densify_samples_{}_{}.pth'.format(grad_ref, i)))

                        sample = targets[0]
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                                self.config.data.image_size,
                                                self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'densify_ref_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'densify_ref_{}.pth'.format(i)))

                    else:
                        sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                    self.config.data.image_size,
                                                    self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder,
                                                            'densify_image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                        #torch.save(sample, os.path.join(self.args.image_folder,
                        #                                'densify_samples_{}.pth'.format(self.config.sampling.ckpt_id)))
                        torch.save(sample, os.path.join(self.args.image_folder,
                                                        'densify_samples_result.pth'))
                        torch.save(samples, os.path.join(self.args.image_folder,
                                                        'densify_samples_target.pth'))

            else:
                if self.config.sampling.data_init:
                    data_iter = iter(dataloader)
                    samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    init_samples = samples + sigmas_th[0] * torch.randn_like(samples)

                else:
                    init_samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                              self.config.data.image_size, self.config.data.image_width,
                                              device=self.config.device)
                    init_samples = data_transform(self.config, init_samples)

                all_samples = anneal_Langevin_dynamics(init_samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=True,
                                                       final_only=self.config.sampling.final_only,
                                                       denoise=self.config.sampling.denoise)

                if not self.config.sampling.final_only:
                    for i, sample in tqdm.tqdm(enumerate(all_samples), total=len(all_samples),
                                               desc="saving image samples"):
                        sample = sample.view(sample.shape[0], self.config.data.channels,
                                             self.config.data.image_size,
                                             self.config.data.image_width)

                        sample = inverse_data_transform(self.config, sample)

                        image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                        save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
                        torch.save(sample, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(i)))
                else:
                    sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                                  self.config.data.image_size,
                                                  self.config.data.image_width)

                    sample = inverse_data_transform(self.config, sample)

                    image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
                    #save_image(image_grid, os.path.join(self.args.image_folder,
                    #                                    'image_grid_{}.png'.format(self.config.sampling.ckpt_id)))
                    #torch.save(sample, os.path.join(self.args.image_folder,
                    #                                'samples_{}.pth'.format(self.config.sampling.ckpt_id)))
                    save_image(image_grid, os.path.join(self.args.image_folder,
                                                        'image_grid.png'))
                    torch.save(sample, os.path.join(self.args.image_folder,
                                                    'samples.pth'))

        else:
            total_n_samples = self.config.sampling.num_samples4fid
            n_rounds = total_n_samples // self.config.sampling.batch_size
            if self.config.sampling.data_init:
                dataloader = DataLoader(dataset, batch_size=self.config.sampling.batch_size, shuffle=True,
                                        num_workers=4)
                data_iter = iter(dataloader)

            img_id = 0
            for _ in tqdm.tqdm(range(n_rounds), desc='Generating image samples for FID/inception score evaluation'):
                if self.config.sampling.data_init:
                    try:
                        samples, _ = next(data_iter)
                    except StopIteration:
                        data_iter = iter(dataloader)
                        samples, _ = next(data_iter)
                    samples = samples.to(self.config.device)
                    samples = data_transform(self.config, samples)
                    samples = samples + sigmas_th[0] * torch.randn_like(samples)
                else:
                    samples = torch.rand(self.config.sampling.batch_size, self.config.data.channels,
                                         self.config.data.image_size,
                                         self.config.data.image_width, device=self.config.device)
                    samples = data_transform(self.config, samples)

                all_samples = anneal_Langevin_dynamics(samples, score, sigmas,
                                                       self.config.sampling.n_steps_each,
                                                       self.config.sampling.step_lr, verbose=False,
                                                       denoise=self.config.sampling.denoise)

                samples = all_samples[-1]
                for img in samples:
                    img = inverse_data_transform(self.config, img)
                    torch.save(img, os.path.join(self.args.image_folder, 'samples_{}.pth'.format(img_id)))
                    save_image(img, os.path.join(self.args.image_folder, 'image_{}.png'.format(img_id)))
                    img_id += 1

    def test(self):
        score = get_model(self.config)
        score = torch.nn.DataParallel(score)

        sigmas = get_sigmas(self.config)

        dataset, test_dataset = get_dataset(self.args, self.config)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.test.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, drop_last=True)

        verbose = False
        for ckpt in tqdm.tqdm(range(self.config.test.begin_ckpt, self.config.test.end_ckpt + 1, 5000),
                              desc="processing ckpt:"):
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{ckpt}.pth'),
                                map_location=self.config.device)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(score)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(score)
            else:
                score.load_state_dict(states[0])

            score.eval()

            step = 0
            mean_loss = 0.
            mean_grad_norm = 0.
            average_grad_scale = 0.
            for x, y in test_dataloader:
                step += 1

                x = x.to(self.config.device)
                x = data_transform(self.config, x)

                with torch.no_grad():
                    test_loss = anneal_dsm_score_estimation(score, x, sigmas, None,
                                                            self.config.training.anneal_power)
                    if verbose:
                        logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))

                    mean_loss += test_loss.item()

            mean_loss /= step
            mean_grad_norm /= step
            average_grad_scale /= step

            logging.info("ckpt: {}, average test loss: {}".format(
                ckpt, mean_loss
            ))

