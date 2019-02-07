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

import os
import cPickle


def prepare_output_dirs_files(args, default_root_dir):
    """
    Asembles (file-)paths and creates directories if necessary
    """
    # Assemble (output-) paths
    root_dir = default_root_dir if args.out_base_path == "" else args.out_base_path
    args.out_path_results = os.path.join(root_dir, args.out_path, args.exp_name)
    args.out_path_results_config    = os.path.join(args.out_path_results, args.out_subdir_config)
    args.out_path_results_images    = os.path.join(args.out_path_results, args.out_subdir_images)
    # Assemble some paths with filenames
    args.filepath_args = os.path.join(args.out_path_results_config, args.out_filename_args)
    args.log_filepath = os.path.join(args.out_path_results, args.out_filename_log)
    if args.model_filepath == "":
        args.model_filepath = os.path.join(args.out_path_results, args.out_subdir_model, args.out_filename_model)
    # Create directories (if necessary)
    if not os.path.exists(args.out_path_results):
        os.makedirs(args.out_path_results)
    if not os.path.exists(args.out_path_results_config):
        os.makedirs(args.out_path_results_config)
    if not os.path.exists(args.out_path_results_images):
        os.makedirs(args.out_path_results_images)
        
        
def save_params(filepath, args):
    """
    Saves the specified parameters args in a binary file specified by filepath
    """
    # Temporarily reset variables which cannot be straightforwardly written
    lambdafunction          = args.lr_lambda
    lambdafunction_pretrain = args.lr_lambda_pretrain
    used_torch_device       = args.used_device
    dummy_lambda_string = "lambda function can not be pickled"
    args.lr_lambda          = dummy_lambda_string    
    args.lr_lambda_pretrain = dummy_lambda_string
    args.used_device        = "{}".format(args.used_device)
    with open(filepath, "wb") as f:
        cPickle.dump(args, f, protocol=cPickle.HIGHEST_PROTOCOL)
    args.lr_lambda          = lambdafunction
    args.lr_lambda_pretrain = lambdafunction_pretrain
    args.used_device        = used_torch_device
    