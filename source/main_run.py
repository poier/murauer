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
import cv2 # to be imported before torch (for me; so that libgomp is loaded from the system installation)

import os
import matplotlib
try:
    os.environ["DISPLAY"]
except:
    matplotlib.use("Agg")   # for machines without display    

# PyTorch
import torch
import torch.utils.data

# Project specific
import nets.netfactory as netfactory
from trainer.supertrainer import SuperTrainer
import eval.handpose_evaluation as hape_eval
import data.loaderfactory as loaderfactory
from data.basetypes import LoaderMode
from util.argparse_helper import parse_arguments_generic
import util.output as out
from util.helper import prepare_output_dirs_files, save_params

import numpy as np
import os.path


#%% Set configuration
from config.config import args                  # general parameters
args = parse_arguments_generic(args)            # command-line parameters
from config.config_data_nyu import args_data    # dataset specific parameters
args.__dict__ = dict(args.__dict__.items() + args_data.__dict__.items()) # merge into single object

args.used_device = torch.device("cuda:{}".format(args.gpu_id) if args.cuda else "cpu")

with torch.cuda.device(args.gpu_id):
    print("Running on {}".format(args.used_device))
    # Set seed for random number generators
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    args.rng = np.random.RandomState(args.seed)
    
    prepare_output_dirs_files(args, os.path.dirname(os.path.realpath(__file__)))
    save_params(args.filepath_args, args)
        
    
    #%% Create data loaders
    if args.do_train:
        train_loaders = loaderfactory.create_train_loaders(args)
        val_loader = loaderfactory.create_dataloader(LoaderMode.VAL, args)
    if args.do_test:
        test_loader = loaderfactory.create_dataloader(LoaderMode.TEST, args)
                                                 
    
    #%% Train/Load model
    model, model_d = netfactory.create_net(args.net_type, args)
    # Load pretrained model if required
    if args.do_load_pretrained_model:
        print("Loading pre-trained model from file {}...".format(
            args.pretrained_model_filepath))
        # First load to CPU (no matter where it was before)
        model.load_state_dict(torch.load(args.pretrained_model_filepath, 
                                         map_location=lambda storage, loc: storage))
        model.to(args.used_device)
    
    if args.do_train:
        tb_log = out.create_crayon_logger(args.exp_name, args.crayon_logger_port)
        print("Training...")
        trainer = SuperTrainer()
        trainer.train(model, model_d, train_loaders, val_loader, args, tb_log)
        
        # Backup experiment logs as zip file
        filename = tb_log.to_zip(args.log_filepath)
        print("Stored log in file {}".format(filename))
        
    # Load stored (best/last) model file
    descr_str = "best" if args.do_use_best_model else "last"
    print("Loading {} model...".format(descr_str))
    print("  from file {}".format(args.model_filepath))
    # First load to CPU (no matter where it was before)
    model.load_state_dict(torch.load(args.model_filepath, 
                                     map_location=lambda storage, loc: storage))
    model.to(args.used_device)
        
        
    #%% Evaluate
    if args.do_test:
        print("Evaluating model on test set...")
        targets, predictions, crop_transforms, coms, data \
            = hape_eval.evaluate_model(model, test_loader, args.used_device)
        print("Computing metrics/Writing to files...")
        hape_eval.compute_and_output_results(targets, predictions, test_loader.dataset,
                                             args, args.out_path_results)
        # Write some qualitative results
        if args.do_write_qualitative_results:
            print("Writing qualitative results...")
            hpe = hape_eval.NYUHandposeEvaluation(targets, predictions, args.num_joints)
            hpe.outputPath = args.out_path_results_images
            hpe.writeQualitativeResults(data, crop_transforms, 
                                        predictions, targets, test_loader, args)
        
print("Finished experiment.")
    