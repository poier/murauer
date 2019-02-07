"""
Copyright 2018 ICG, Graz University of Technology

This file is part of MURAUER.

MURAUER is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MURAUER is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MURAUER.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import util.losses as losses
import util.iterators as iters

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np

import copy
import time
import os.path


mse = nn.MSELoss(size_average=True)
sse = nn.MSELoss(size_average=False)
recon_l1 = nn.L1Loss(size_average=False)


class HaPeTrainerMapPreviewAdversarial(object):
    """
    """
    
    def __init__(self, train_loaders, val_loader, logger, trainer_parameters=None):
        """
        Initialize trainer
        
        Arguments:
            train_loaders (data.basetypes.TrainLoaders): train data loaders
            val_loader (torch.utils.data.DataLoader): validation data loader
            logger (CrayonExperiment): crayon logger, 
                i.e., created by CrayonClient.create_experiment()
            trainer_parameters: object containing all parameters
        """
        self.pretrain_loader = train_loaders.loader_pretrain
        self.train_loader_corr = train_loaders.loader_corr
        self.train_loader_synth = train_loaders.loader_synth
        self.train_loader_preview = train_loaders.loader_preview
        self.train_loader_wc_ul_r = train_loaders.loader_real_weakcorr_ul
        self.train_loader_wc_ul_s = train_loaders.loader_synth_weakcorr_ul
        self.val_loader = val_loader
        self.log = logger
        self.train_params = trainer_parameters
        
    
    def loss_function_train_map_preview_adv(self, joints_pred_r, joints_target_r, 
                                            joints_pred_s, joints_target_s, 
                                            embedding_pred, embedding_target,
                                            joints_pred_rest_s, joints_target_rest_s, 
                                            joints_pred_prev_r, joints_target_prev_r,
                                            img_pred_prev_r, img_target_prev_r,
                                            joints_pred_prev_s, joints_target_prev_s,
                                            img_pred_prev_s, img_target_prev_s,
                                            embedding_prev_pred, embedding_prev_target,
                                            discriminator_pred, discriminator_target,
                                            step_id):
        gamma = self.train_params.lambda_embedding_loss
        beta = self.train_params.lambda_realdata_loss
        lam_p = self.train_params.lambda_preview_loss
        lam_adv = self.train_params.lambda_adversarial_loss
        
        loss_joints_r       = sse(joints_pred_r.view_as(joints_target_r), joints_target_r)
        loss_joints_s       = sse(joints_pred_s.view_as(joints_target_s), joints_target_s)
        
        loss_joints_rest_s = 0.0
        if joints_pred_rest_s.size():
            loss_joints_rest_s  = sse(joints_pred_rest_s.view_as(joints_target_rest_s), joints_target_rest_s)
            
        # To have real and synth. loss corresponding despite diff. numbers of 
        # labeled real samples (in each batch), 
        # compute mse and scale by num. synthetic-samples
        loss_joints_prev_r = 0.0
        if joints_pred_prev_r.size():
            loss_joints_prev_r = mse(joints_pred_prev_r.view_as(joints_target_prev_r), joints_target_prev_r)
            loss_joints_prev_r = loss_joints_prev_r * joints_target_prev_s.size(0)
        loss_joints_prev_s  = sse(joints_pred_prev_s.view_as(joints_target_prev_s), joints_target_prev_s)
        
        loss_emb            = sse(embedding_pred.view_as(embedding_target), embedding_target)
        loss_emb_prev = 0.0
        if embedding_prev_pred.size():
            loss_emb_prev   = mse(embedding_prev_pred.view_as(embedding_prev_target), embedding_prev_target)
            loss_emb_prev = loss_emb_prev * embedding_target.size(0)    # same weight as orig. embedding loss
        loss_emb = loss_emb + loss_emb_prev
        
        loss_emb_adv = self.loss_function_discriminator(
                                            discriminator_pred, discriminator_target)
        
        loss_preview_r      = recon_l1(img_pred_prev_r, img_target_prev_r)
        loss_preview_s      = recon_l1(img_pred_prev_s, img_target_prev_s)
        sum_prev_r = float(img_target_prev_r.size(0))
        sum_prev_s = float(img_target_prev_s.size(0))
        loss_preview = loss_preview_r + loss_preview_s
        
        loss_real   = loss_joints_r + loss_joints_prev_r
        loss_synth  = loss_joints_s + loss_joints_rest_s + loss_joints_prev_s
                
        loss = loss_synth \
                + beta * loss_real \
                + gamma * loss_emb \
                + lam_p * loss_preview \
                + lam_adv * loss_emb_adv
              
        try:
            self.log.add_scalar_value("train-loss-full", loss.item(), 
                                      wall_time=time.clock(), step=step_id)
            self.log.add_scalar_value("train-loss-real", loss_real.item(), 
                                      wall_time=time.clock(), step=step_id)
            self.log.add_scalar_value("train-loss-synth", loss_synth.item(), 
                                      wall_time=time.clock(), step=step_id)
            self.log.add_scalar_value("train-loss-emb", loss_emb.item(), 
                                      wall_time=time.clock(), step=step_id)
            self.log.add_scalar_value("train-loss-preview", loss_preview.item(), 
                                      wall_time=time.clock(), step=step_id)
            self.log.add_scalar_value("train-loss-preview-real", loss_preview_r.item() / sum_prev_r, 
                                      wall_time=time.clock(), step=step_id)
            self.log.add_scalar_value("train-loss-preview-synth", loss_preview_s.item() / sum_prev_s, 
                                      wall_time=time.clock(), step=step_id)
            self.log.add_scalar_value("train-loss-emb-adv", loss_emb_adv.item(), 
                                      wall_time=time.clock(), step=step_id)
        except:
            print("Logging to server failed.")
            pass
        
        return loss
        
    
    def loss_function_pretrain(self, joints_pred, joints_target, step_id):
        loss_joints = mse(joints_pred.view_as(joints_target), joints_target)
        num_samples = joints_pred.size()[0]
        
        loss = loss_joints * num_samples
                    
        try:
            self.log.add_scalar_value("pretrain-loss-step", loss.item(), 
                                      wall_time=time.clock(), step=step_id)
        except:
            print("Logging to server failed.")
            pass
        
        return loss

        
    def loss_function_discriminator(self, prediction, target):
        return sse(prediction.view_as(target), target)
        
    
    def loss_function_test(self, estimate, target):
        return losses.joint_pos_distance_loss(estimate, target)
    
    
    def train(self, model, optim_params, model_discriminator, num_epochs, 
              lr=1e-3, weight_decay=0.001, optim_type=0, do_pretrain=True):
        """
        Arguments:
            optim_params: can be an iterable of Variables (e.g., 
                model.parameters()), or an iterable of dicts 
                Specified separately here in order to, 
                e.g., define specific optimization parameters 
                for each parameter of model (see [1])
                This is ignored for pre-training, i.e., model.parameters() is
                used for pretraining
                
        [1] http://pytorch.org/docs/optim.html#per-parameter-options
        """
                  
        if not self.train_params.do_load_pretrained_model:
            self.pretrain_using_synth(model, self.train_params.num_epochs_pretrain, 
                                      self.train_params.lr_pretrain, 
                                      weight_decay=weight_decay, 
                                      optim_type=optim_type)
                                  
        if optim_type == 0:
            optimizer = optim.Adam(optim_params, 
                                   lr=lr, weight_decay=weight_decay)
            optimizer_d = optim.Adam(model_discriminator.parameters(), 
                                   lr=lr, weight_decay=weight_decay)
        elif optim_type == 1:
            optimizer = optim.RMSprop(optim_params, 
                                      lr=lr, weight_decay=weight_decay)
            optimizer_d = optim.RMSprop(model_discriminator.parameters(), 
                                      lr=lr, weight_decay=weight_decay)
        elif optim_type == 2:
            optimizer = optim.SGD(optim_params, 
                                  lr=lr, momentum=0.9, weight_decay=weight_decay)
            optimizer_d = optim.SGD(model_discriminator.parameters(), 
                                  lr=lr, momentum=0.9, weight_decay=weight_decay)
                                  
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.train_params.lr_lambda)
        scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=self.train_params.lr_lambda)
        
        num_all_samples_train = self.train_params.id_end_train - self.train_params.id_start_train
        num_iterations_per_epoch = int(np.ceil(float(num_all_samples_train) / self.train_params.batch_size))
        
        if self.train_params.do_save_model \
                and not os.path.exists(
                os.path.dirname(self.train_params.model_filepath)):
            os.makedirs(os.path.dirname(self.train_params.model_filepath))
                                      
        model.train()
        model_discriminator.train()
        
        t_0 = time.time()
        
        val_error_best = np.inf
        epoch_best = np.NaN
        for epoch in range(1, num_epochs + 1):
            scheduler.step()
            scheduler_d.step()
            
            t_0_epoch = time.time()
            self.train_epoch(epoch, model, model_discriminator, 
                             optimizer, optimizer_d, num_iterations_per_epoch)
            t_1_epoch = time.time()
            t_epoch = t_1_epoch - t_0_epoch
            print("Time (wall) for training epoch: {:.1f} sec.".format(t_epoch))
            
            val_error = self.test_epoch(epoch, model, num_iterations_per_epoch)
            
            # Store current model
            if self.train_params.do_save_model:
                print("Storing current model...")
                torch.save(model.state_dict(), self.train_params.model_filepath)
            # Store (intermediate) model
            if ((epoch % self.train_params.save_model_epoch_interval) == 0) \
                    and self.train_params.do_save_intermediate_models:
                print("Storing current model (permanently, as intermediate model)...")
                filename = self.train_params.model_filepath + "_epoch{}".format(epoch)
                torch.save(model.state_dict(), filename)
                
            if self.train_params.do_use_best_model and (val_error < val_error_best):
                print("Currently, best val. error; copying model...")
                model_best = copy.deepcopy(model)
                val_error_best = val_error
                epoch_best = epoch
                
        t_1 = time.time()
        
        if self.train_params.do_use_best_model:
            model = copy.deepcopy(model_best)
            print("Best model from epoch {}/{} (val. error: {})".format(
                epoch_best, num_epochs, val_error_best))
        if self.train_params.do_save_model:
            torch.save(model.state_dict(), self.train_params.model_filepath)
    
        t_train = t_1 - t_0
        print("Time (wall) for train: {:.1f} sec. ({:.1f} hours)".format(t_train, t_train / 3600.))
        
        
    def train_epoch(self, epoch, model, model_d, optimizer, optimizer_d, 
                    num_iterations):
        model.train()
        model_d.train()
        
        label_real_d = 1
        label_synth_d = 0
        
        target_d = torch.FloatTensor(self.train_params.batch_size)
        target_d = target_d.to(self.train_params.used_device)
        
        do_bp_feat_emb = not self.train_params.no_backprop_through_featext_for_emb
        
        train_loss = 0
        num_samples_done = 0
        
        iter_corr = iters.cycle_no_memory(self.train_loader_corr)
        iter_synth = iters.cycle_no_memory(self.train_loader_synth)
        iter_prev = iters.cycle_no_memory(self.train_loader_preview)
        iter_wc_ul_r = iters.cycle_no_memory(self.train_loader_wc_ul_r)
        iter_wc_ul_s = iters.cycle_no_memory(self.train_loader_wc_ul_s)
        
        for batch_idx in range(num_iterations):
            # Load data
            data_corr_r, _,_, joints_corr_r,_,_, _,_,_, com_corr_r,_,_, _,_,_, \
            data_corr_s, _,_, joints_corr_s,_,_, _,_,_, com_corr_s,_,_, _,_,_, _ = \
                iter_corr.next()
            data_s, joints_s, _, com_s, _ = iter_synth.next()
            data_prev_r, data_prev_c2_r, data_prev_c3_r, joints_prev_r,_,_, _,_,_, com_prev_r,_,_, _,_,_, \
            data_prev_s, data_prev_c2_s, data_prev_c3_s, joints_prev_s,_,_, _,_,_, com_prev_s,_,_, _,_,_, is_labeled_prev \
                = iter_prev.next()
            data_wc_ul_r, joints_wc_ul_r, _, com_wc_ul_r, _ = iter_wc_ul_r.next()
            data_wc_ul_s, joints_wc_ul_s, _, com_wc_ul_s, _ = iter_wc_ul_s.next()
                
            # Select cam views
            list_camviews_prev_r = [data_prev_r, data_prev_c2_r, data_prev_c3_r]
            data_target_prev_r = list_camviews_prev_r[self.train_params.output_cam_ids_train[0] - 1] # 1-based IDs
            list_camviews_prev_s = [data_prev_s, data_prev_c2_s, data_prev_c3_s]
            data_target_prev_s = list_camviews_prev_s[self.train_params.output_cam_ids_train[0] - 1] # 1-based IDs
            
            # Normalize and jitter coms
            com_prev_r = torch.from_numpy(
                self.train_loader_preview.dataset.normalize_and_jitter_3D(com_prev_r.numpy()))
            com_prev_s = torch.from_numpy(
                self.train_loader_preview.dataset.normalize_and_jitter_3D(com_prev_s.numpy()))
            
            # Data preparation
            # input/joints from (corresponding) real data
            data_input_corr_r       = self.prepare_data(data_corr_r)
#            data_target_corr_r      = self.prepare_data(data_target_corr_r)
            com_corr_r              = self.prepare_data(com_corr_r)
            joints_target_corr_r    = self.prepare_data(joints_corr_r)
            # input/joints from (corresponding) synth data
            data_input_corr_s       = self.prepare_data(data_corr_s)
#            data_target_corr_s      = self.prepare_data(data_target_corr_s)
            com_corr_s              = self.prepare_data(com_corr_s)
            joints_target_corr_s    = self.prepare_data(joints_corr_s)
            # input/joints from synthetic data (without correspondences)
            data_input_rest_s       = self.prepare_data(data_s)
            joints_target_rest_s    = self.prepare_data(joints_s)
            # input/joints from view prediction real data
            data_input_prev_r       = self.prepare_data(data_prev_r)
            data_target_prev_r      = self.prepare_data(data_target_prev_r)
            com_prev_r              = self.prepare_data(com_prev_r)
            joints_target_prev_r    = self.prepare_data(joints_prev_r)
            # input/joints from view prediction synth data
            data_input_prev_s       = self.prepare_data(data_prev_s)
            data_target_prev_s      = self.prepare_data(data_target_prev_s)
            com_prev_s              = self.prepare_data(com_prev_s)
            joints_target_prev_s    = self.prepare_data(joints_prev_s)
            # input/joints from (weakly corresponding, unlabeled) real data
            data_input_wc_ul_r      = self.prepare_data(data_wc_ul_r)
            com_wc_ul_r             = self.prepare_data(com_wc_ul_r)
            # input/joints from (weakly corresponding, unlabeled) synth data
            data_input_wc_ul_s      = self.prepare_data(data_wc_ul_s)
            com_wc_ul_s             = self.prepare_data(com_wc_ul_s)
            
            # fprop
            optimizer.zero_grad()
            joints_pred_corr_r, emb_corr_r, _ \
                                    = model(data_input_corr_r, com_corr_r, do_map=True, do_preview=False)
            joints_pred_corr_s, emb_corr_s, _ \
                                    = model(data_input_corr_s, com_corr_s, do_map=False, do_preview=False)
            joints_pred_rest_s, _,_ = model(data_input_rest_s, com_s,      do_map=False, do_preview=False)
            joints_pred_prev_r, emb_prev_r, data_pred_prev_r \
                                    = model(data_input_prev_r, com_prev_r, do_map=True)
            joints_pred_prev_s, emb_prev_s, data_pred_prev_s \
                                    = model(data_input_prev_s, com_prev_s, do_map=False)
            _, emb_wc_ul_r, _       = model(data_input_wc_ul_r, com_wc_ul_r, do_map=True, do_preview=False, 
                                            do_backprop_through_feature_extr_before_map=do_bp_feat_emb)
            _, emb_wc_ul_s, _       = model(data_input_wc_ul_s, com_wc_ul_s, do_map=False, do_preview=False)
            
            # Update discriminator
            optimizer_d.zero_grad()
            # Mini-batch with real data
            # Create label vector for batch with real data
            batch_size_adv = data_input_wc_ul_r.size(0)
            target_d.resize_(batch_size_adv).fill_(label_real_d)
            # fprop/backprop
            prediction_d = model_d(emb_wc_ul_r.detach())     # only backprop through discriminator
            errD_r = self.loss_function_discriminator(prediction_d, target_d)
            errD_r.backward()
            # for log
            D_r = prediction_d.data.mean()
            # Mini-batch with synthetic data
            target_d.fill_(label_synth_d)
            # fprop/backprop
            prediction_d = model_d(emb_wc_ul_s.detach())     # only backprop through discriminator
            errD_s = self.loss_function_discriminator(prediction_d, target_d)
            errD_s.backward()
            # for log
            D_s = prediction_d.data.mean()
            errD = errD_r + errD_s
            # Update
            optimizer_d.step()
            
            # Update generator
            target_d.fill_(label_synth_d)    # targets for mapped real samples are "synthetic" for "mapping cost"
            # fprop through discriminator on ("mapped") real samples again 
            # but want to backprop through mapping this time, i.e., no detach()
            prediction_d = model_d(emb_wc_ul_r)
                                    
            # Select labeled data (from "preview loader") for joint loss
            is_labeled_prev = is_labeled_prev.to(self.train_params.used_device)
            joints_pred_prev_rl = joints_pred_prev_r[is_labeled_prev.view(-1)]
            joints_target_prev_rl = joints_target_prev_r[is_labeled_prev.view(-1)]
            # Select labeled data for embedding loss (we do not assume correspondence for unlabeled data)
            emb_prev_rl = emb_prev_r[is_labeled_prev.view(-1)]
            emb_prev_sl = emb_prev_s[is_labeled_prev.view(-1)]
            
            step = (epoch-1) * num_iterations + batch_idx   # for logging
            
            loss = self.loss_function_train_map_preview_adv(joints_pred_corr_r, joints_target_corr_r,
                                                            joints_pred_corr_s, joints_target_corr_s,
                                                            emb_corr_r, emb_corr_s.detach(),
                                                            joints_pred_rest_s, joints_target_rest_s,
                                                            joints_pred_prev_rl, joints_target_prev_rl,
                                                            data_pred_prev_r, data_target_prev_r,
                                                            joints_pred_prev_s, joints_target_prev_s,
                                                            data_pred_prev_s, data_target_prev_s,
                                                            emb_prev_rl, emb_prev_sl.detach(),
                                                            prediction_d, target_d,
                                                            step)
                                                        
            loss.backward()
            # For log
            D_r2 = prediction_d.data.mean()
            train_loss += loss.item()
            # Update
            optimizer.step()
            
            # Logging
            batch_size_curr = len(data_input_corr_r) \
                              + len(data_input_rest_s) \
                              + len(data_input_prev_r) \
                              + len(data_input_prev_s) \
                              + len(data_input_wc_ul_r)
            num_samples_done += batch_size_curr
            if (batch_idx+1) % self.train_params.log_interval == 0:
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss/Sample: {:.6f}, critic: {:.6f}; D_s: {:.6f}, D_r: {:.6f}, D_r2: {:.6f}'.format(
                    epoch, num_samples_done, 
                    100. * (batch_idx+1) / num_iterations,
                    loss.item() / batch_size_curr,
                    errD.item() / batch_size_curr, D_s, D_r, D_r2)) 
                try:
                    self.log.add_scalar_value("train-loss", loss.item(), 
                                              wall_time=time.clock(), step=step)
                    self.log.add_scalar_value("discriminator-loss", errD.item(), 
                                              wall_time=time.clock(), step=step)
                    self.log.add_scalar_value("disc-pred-real", D_r.item(), 
                                              wall_time=time.clock(), step=step)
                    self.log.add_scalar_value("disc-pred-synth", D_s.item(), 
                                              wall_time=time.clock(), step=step)
                except:
                    print("Logging to server failed.")
                    pass
    
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / num_samples_done ))
            
    
    def pretrain_using_synth(self, model, num_epochs, 
              lr=1e-3, weight_decay=0.001, optim_type=0):
        """
        Pre-train the model
        """
        do_use_real_val_data = False
        val_loss_name = "val-loss-pretrain"
        
        if optim_type == 0:
            optimizer = optim.Adam(model.parameters(), 
                                   lr=lr, weight_decay=weight_decay)
        elif optim_type == 1:
            optimizer = optim.RMSprop(model.parameters(), 
                                      lr=lr, weight_decay=weight_decay)
        elif optim_type == 2:
            optimizer = optim.SGD(model.parameters(), 
                                  lr=lr, momentum=0.9, weight_decay=weight_decay)
                                  
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.train_params.lr_lambda_pretrain)
        
        num_train_iterations_per_epoch = len(self.pretrain_loader)
        
        if self.train_params.do_save_model \
                and not os.path.exists(
                os.path.dirname(self.train_params.model_filepath)):
            os.makedirs(os.path.dirname(self.train_params.model_filepath))
                                      
        model.train()
        
        t_0 = time.time()
        
        val_error_best = np.inf
        epoch_best = np.NaN
        for epoch in range(1, num_epochs + 1):
            scheduler.step()
            
            t_0_epoch = time.time()
            self.pretrain_epoch(epoch, model, optimizer)
            t_1_epoch = time.time()
            val_error = self.test_epoch(epoch, model, 
                                        num_train_iterations_per_epoch, 
                                        do_use_real_data=do_use_real_val_data, 
                                        loss_name=val_loss_name)
            
            t_epoch = t_1_epoch - t_0_epoch
            print("Time (wall) for pre-training epoch: {:.2f} sec.".format(t_epoch))
            
            if self.train_params.do_use_best_model_pretrain and (val_error < val_error_best):
                model_best = copy.deepcopy(model)
                val_error_best = val_error
                epoch_best = epoch
                
        t_1 = time.time()
        
        if self.train_params.do_use_best_model_pretrain:
            model = copy.deepcopy(model_best)
            print("Best model from (pre-training) epoch {}/{} (val. error: {})".format(
                epoch_best, num_epochs, val_error_best))
           
        if self.train_params.do_save_model:
            filepath = self.train_params.model_filepath + "_pretrain"
            torch.save(model.state_dict(), filepath)
    
        print("Time (wall) for pre-train: {:.2f} sec.".format(t_1 - t_0))
        
        
    def pretrain_epoch(self, epoch, model, optimizer):
        model.train()
        train_loss = 0
        num_samples_done = 0
        for batch_idx, (data, joints, _, com, _) \
                in enumerate(self.pretrain_loader):
            
            # Input/joints from synthetic data
            data_input    = self.prepare_data(data)
            joints_target = self.prepare_data(joints)
            com = self.prepare_data(com)
                
            optimizer.zero_grad()
            joints_pred, _,_ = model(data_input, com, do_map=False, classifier_id=0, do_preview=False)
            
            step = (epoch-1) * len(self.pretrain_loader) + batch_idx # for logging
            
            loss = self.loss_function_pretrain(joints_pred, joints_target, step)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # Logging
            batch_size_curr = len(data_input)
            num_samples_done += batch_size_curr
            if (batch_idx+1) % (self.train_params.log_interval-1) == 0:
                print('Pre-Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss/Sample: {:.6f}'.format(
                    epoch, num_samples_done, len(self.pretrain_loader.sampler),
                    100. * (batch_idx+1) / len(self.pretrain_loader),
                    loss.item() / batch_size_curr))
                try:
                    self.log.add_scalar_value("pretrain-loss", 
                                              loss.item() / batch_size_curr, 
                                              wall_time=time.clock(), step=step)
                except:
                    print("Logging to server failed.")
                    pass
    
        print('====> Pre-Train Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.pretrain_loader.sampler) ))


    def test_epoch(self, epoch, model, num_train_iterations_per_epoch, 
                   do_use_real_data=True, loss_name_prefix="val-loss"):
        with torch.no_grad():
            loss_name_postfix = "-real" if do_use_real_data else "-synth"
            loss_name = loss_name_prefix + loss_name_postfix
            
            model.eval()
            test_loss = 0
            for batch_idx, (data_cam1_r,_,_, joints_target_r,_,_, _,_,_, com_cam1_r,_,_, _,_,_, \
                data_cam1_s,_,_, joints_target_s,_,_, _,_,_, com_cam1_s,_,_, _,_,_, _) \
                    in enumerate(self.val_loader):
                if do_use_real_data:
                    data_input      = self.prepare_data(data_cam1_r)
                    joints_target   = self.prepare_data(joints_target_r)
                    com             = self.prepare_data(com_cam1_r)
                    do_map = True
                else:
                    data_input      = self.prepare_data(data_cam1_s)
                    joints_target   = self.prepare_data(joints_target_s)
                    com             = self.prepare_data(com_cam1_s)
                    do_map = False
                    
                joints_pred, _,_ = model(data_input, com, do_map=do_map, do_preview=False)
                test_loss += (self.loss_function_test(joints_pred, joints_target).item()
                    * joints_target.size()[0])
        
            test_loss /= len(self.val_loader.sampler)
            # Denormalize the loss, to have it corresponding to test error       
            # Note, this is wrong as soon as different cube/crop-sizes are used per sample 
            # (not sure if we want to account for that during training)
            test_loss *= self.val_loader.dataset.args_data.crop_size_3d_tuple[2]
            if not self.val_loader.dataset.args_data.do_norm_zero_one:
                test_loss /= 2.0
            # Output
            print('====> Validation set loss: {:.4f}'.format(test_loss))
            step = epoch * num_train_iterations_per_epoch
            try:
                self.log.add_scalar_value(loss_name, test_loss, 
                                          wall_time=time.clock(), step=step)
            except:
                print("Logging to server failed.")
                pass
                                    
            return test_loss
        
        
    def prepare_data(self, data):
        """
        Prepare for computation (Put to desired device)
        """
        return data.to(self.train_params.used_device)
        
        