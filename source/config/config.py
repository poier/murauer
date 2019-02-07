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

# Project specific
import data.basetypes as basetypes
# Other libs
import numpy as np


args = basetypes.Arguments()

# How many (CPU) workers for loading data
args.num_loader_workers = 5

# Model architecture specific parameters
# Number of (base-)feature channels (usually increased after downsampling)
args.num_features = 32
# Number of latent representation/embedding dimensions
args.num_bottleneck_dim = 1024
# Number of additional input dimensions for the view prediction decoder
args.num_cond_dims = 3
# Architecture of the discriminator
args.discriminator_type = basetypes.DiscriminatorNetType.RESNET

# Learning-rate decay
args.lr_lambda = lambda epoch: (0.33 ** max(0, 2-epoch//2)) if epoch < 4 else np.exp(-0.04*epoch)   # starts with 0.1, after 2 epochs 0.33, after 4 epochs multiplier is exp(-0.04*epoch)
# Use the model with best val. error (over epochs)?
args.do_use_best_model = False

# Pre-training
args.num_epochs_pretrain = 150
args.lr_pretrain = 3.3*1e-4
args.lr_lambda_pretrain = args.lr_lambda
args.do_use_best_model_pretrain = False
args.pretrained_model_filepath = "../results/pretrained_models/model_synth_only.mdl_pretrain"

# Output parameters
args.do_save_model = True
args.do_save_intermediate_models = False    # save "checkpoints" at fixed intervals?
args.save_model_epoch_interval = 20
args.out_filename_model = "model.mdl"
args.out_filename_args = "args.pkl"
args.out_filename_log = "crayon_results_log"
args.out_filename_result = "results.txt"
args.out_filename_joint_positions_estimated = "results_joint_pos_estimated.pkl"
args.out_filename_joint_positions_groundtruth = "results_joint_pos_gt.pkl"
args.out_filename_joint_positions_estimated_xyz_txt = "results_joint_pos_estimated_xyz.txt"
args.out_filename_joint_positions_estimated_uvd_txt = "results_joint_pos_estimated_uvd.txt"
args.out_fmt_txt = "%.4f"   # Number format specification for joint pos. textfiles
args.step_result_overlay = 200
args.out_subdir_model = "model"
args.out_subdir_images = "images"
args.out_subdir_config = "config"
args.out_subdir_results_synthetic = "results_on_synthetic_data"
args.do_show_images = False
args.do_write_qualitative_results = False
args.crayon_logger_port = 8889
