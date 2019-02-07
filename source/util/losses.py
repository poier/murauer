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

import torch

from enum import IntEnum


#%% General definitions
class LossType(IntEnum):
    L1 = 0
    L2 = 1
    HUBER = 2


#%%
def joint_pos_distance_loss(estimate, target, size_average=True):
    """
    Joint distances (L2-norm per joint)
    
    Arguments:
        estimate (torch.autograd.Variable) estimated positions, BxDx1x1,
            where D is the number of dimensions, i.e., N*3, 
            where N is the number of joint positions
        target (torch.autograd.Variable) target positions, BxNx3, 
            where N is the number of joint positions
    """
    divisor = (target.size()[0] * target.size()[1]) if size_average else 1.0
    
    est = estimate.view(target.size())
    return torch.sum(torch.sqrt(torch.pow((est - target), 2).sum(dim=2, keepdim=False))) / divisor
    