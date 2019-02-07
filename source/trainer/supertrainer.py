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

from nets.netfactory import NetType
from trainer.hapetrainer_map_preview_adversarial import HaPeTrainerMapPreviewAdversarial
from data.basetypes import TrainingType
        

#%%
class SuperTrainer(object):
    """
    Wrapper for network trainers
    """
    
    def __init__(self):
        """
        Initialize trainer
        """
        
    
    def train(self, model, model_discriminator, train_loaders, val_loader, args, tb_log):
        """
        Train the given model(s) in the specified way
        """
        # Sanity check
        if (not args.net_type == NetType.HAPE_RESNET50_MAP_PREVIEW) \
                or (not args.training_type == TrainingType.ADVERSARIAL):
            raise UserWarning("Specified training-/net-type not implemented.")
                
        # Create trainer
        trainer = HaPeTrainerMapPreviewAdversarial(train_loaders,
                                                   val_loader=val_loader, 
                                                   logger=tb_log, 
                                                   trainer_parameters=args)
            
        # Train model
        trainer.train(model, model.parameters(), model_discriminator, 
                      args.epochs, lr=args.lr, weight_decay=args.weight_decay, 
                      optim_type=args.optim_type)
                      