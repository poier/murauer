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

from nets.resnet import resnet50, DiscriminatorResNet
from data.basetypes import TrainingType, NetType, DiscriminatorNetType
        
    
#%%
def create_net(net_type, args):
    """
    Create a network model(s) according to parameters
    """
    
    model, model_d = None, None
    # Create "prediction" model (pose and view prediction)
    if net_type == NetType.HAPE_RESNET50_MAP_PREVIEW:
        model = resnet50(net_type='MapPreview', 
                         num_classes=(args.num_joints * 3),
                         num_features=args.num_features, 
                         num_bottleneck_dim=args.num_bottleneck_dim)
    else:
        print("NetType (={}) unknown!".format(net_type))
        raise UserWarning("NetType unknown.")
        
    # Create discriminator
    if args.training_type == TrainingType.ADVERSARIAL:
        if args.discriminator_type == DiscriminatorNetType.RESNET:
            model_d = DiscriminatorResNet(num_in_planes=args.num_bottleneck_dim)
        else:
            raise UserWarning("DiscriminatorNetType not known/implemented.")
        
    # Put model(s) to used device
    model.to(args.used_device)
    if args.training_type == TrainingType.ADVERSARIAL:
        model_d.to(args.used_device)
        
    return model, model_d
    
    
    