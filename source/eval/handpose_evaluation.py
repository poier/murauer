"""
Copyright 2015, 2018 ICG, Graz University of Technology

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

from data.NyuHandPoseDataset import NyuAnnoType
from data.loaderfactory import DatasetType
from util.transformations import transformPoint2D
import nets

import numpy
import os
import time
import matplotlib.pyplot as plt
import cPickle

# Pytorch
import torch


#%% Helper
def evaluate_model(model, testset_loader, used_device):
    """
    Evaluates a learned model on the data provided by the loader
    
    Arguments:
        model (torch.nn.Module): the trained model
        testset_loader (torch.utils.data.DataLoader): loader for test samples;
            need to be a loader for a NyuHandPoseMultiViewDataset
        used_device (torch.device)
    """
    with torch.no_grad():
        numJoints = testset_loader.dataset.args_data.num_joints
        sample_size = testset_loader.dataset[0][0].numpy().shape
            
        model = model.to(used_device)
        model.eval()
        
        data_all = numpy.zeros((0, sample_size[0], sample_size[1], sample_size[2]))
        targets_all = numpy.zeros((0, numJoints, 3))
        estimates_all = numpy.zeros((0, numJoints, 3))
        transforms_all = numpy.zeros((0, 3, 3))
        coms_all = numpy.zeros((0, 3))
        
        t0 = time.time()
        time_sum = 0.0
        for data,_,_, targets,_,_, transforms,_,_, coms,_,_, sizes,_,_, \
            data_s,_,_, targets_s,_,_, transforms_s,_,_, coms_s,_,_, sizes_s,_,_, _ \
                in testset_loader:
            # Compute prediction
            data = data.to(used_device)
            t0_ = time.time()
            if isinstance(model, nets.resnet.ResNet_Map_PreView):
                estimates = model(data, [], do_preview=False)
            else:
                estimates = model(data)
            t1_ = time.time()
            time_sum += t1_ - t0_
            if isinstance(estimates, tuple):    # a model which returns a tuple shall return the pose first
                estimates = estimates[0]
            
            # Prepare output transform result space, to numpy arrays, ...
            data_all = numpy.append(data_all, data.data.cpu().numpy(), axis=0)
            com_n = coms.numpy()
            targets = targets.numpy()
            sizes = sizes.numpy()
            targets = testset_loader.dataset.denormalize_joint_pos(targets, sizes)
            targets = targets + com_n.reshape(com_n.shape[0], 1, com_n.shape[1])
            targets_all = numpy.append(targets_all, targets, axis=0)
            
            estimates = estimates.data.cpu().view(-1,numJoints,3).numpy()
            estimates = testset_loader.dataset.denormalize_joint_pos(estimates, sizes)
            estimates = estimates + com_n.reshape(com_n.shape[0], 1, com_n.shape[1])
            estimates_all = numpy.append(estimates_all, estimates, axis=0)
            
            transforms_all = numpy.append(transforms_all, transforms.numpy(), axis=0)
            
            coms_all = numpy.append(coms_all, com_n, axis=0)
        
        t1 = time.time()
        runtime = t1 - t0
        num_images = len(testset_loader.sampler)
        print("  Time for evaluating on {} images: {:.2f} sec. ({:.1f} msec./image)".format(
            num_images, runtime, runtime / float(num_images) * 1000.0))
        print("  Time measured only for fprops: {:.2f} sec. ({:.1f} msec./image)".format(
            time_sum, time_sum / float(num_images) * 1000.0))
        
        return targets_all, estimates_all, transforms_all, coms_all, data_all


def compute_and_output_results(targets, predictions, dataset, args, out_path):
    hpe = get_evaluator(targets, predictions, args)
    hpe.outputPath = out_path
    num_joints = targets.shape[1]
    
    # Compute (and print) basic errors
    mean_error = hpe.getMeanError()
    print("  Mean error: {:.2f} mm".format(mean_error))
    print("  FS80: {:.3f}".format(hpe.getMDscore(80)))
            
    # Write evaluation results to textfile
    out_result_filepath = os.path.join(out_path, args.out_filename_result)
    hpe.writeResults2Textfile(out_result_filepath)
    
    # Write predicted joint positions (binary)
    filepath_est = os.path.join(out_path, args.out_filename_joint_positions_estimated)
    cPickle.dump(predictions, open(filepath_est, "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
    # Write ground truth joint positions (binary)
    filepath_gt = os.path.join(out_path, args.out_filename_joint_positions_groundtruth)
    cPickle.dump(targets, open(filepath_gt, "wb"), protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Write predicted joint positions (xyz, to text-file)
    jts_out = predictions.reshape(-1, num_joints * 3)
    filepath_txt_xyz = os.path.join(out_path, args.out_filename_joint_positions_estimated_xyz_txt)
    numpy.savetxt(filepath_txt_xyz, jts_out, args.out_fmt_txt)
    
    # Convert predicted joint positions to uvd and write to text-file
    predictions_uvd = dataset.points3DToImg(predictions.reshape(-1, 3))
    predictions_uvd = predictions_uvd.reshape(-1, num_joints * 3)
    filepath_txt_uvd = os.path.join(out_path, args.out_filename_joint_positions_estimated_uvd_txt)
    numpy.savetxt(filepath_txt_uvd, predictions_uvd, args.out_fmt_txt)
    
    hpe.plotEvaluation(args.exp_name, methodName='Ours')
                
                
def get_evaluator(targets, predictions, args):
    """
    Create an evaluator for the dataset specified by args.dataset_type
    """
    if args.dataset_type == DatasetType.NYU:
        hpe = NYUHandposeEvaluation(targets, predictions, joints=args.num_joints)
    elif args.dataset_type == DatasetType.ICG:
        hpe = ICGHandposeEvaluation(targets, predictions)
    return hpe


class HandposeEvaluation(object):
    """
    Different evaluation metrics for handpose, L2 distance used
    """

    def __init__(self, gt, joints):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        if not (isinstance(gt, numpy.ndarray) or isinstance(gt, list)) or not (
                isinstance(joints, list) or isinstance(joints, numpy.ndarray)):
            raise ValueError("Params must be list or ndarray")

        if len(gt) != len(joints):
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gt), len(joints)))
            raise ValueError("Params must be the same size")

        if len(gt) == len(joints) == 0:
            print("Error: groundtruth has {} elements, eval data has {}".format(len(gt), len(joints)))
            raise ValueError("Params must be of non-zero size")

        if gt[0].shape != joints[0].shape:
            print("Error: groundtruth has {} dims, eval data has {}".format(gt[0].shape, joints[0].shape))
            raise ValueError("Params must be of same dimensionality")

        self.gt = numpy.asarray(gt)
        self.joints = numpy.asarray(joints)
        assert (self.gt.shape == self.joints.shape)

        self.colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'brown', 'gray', 'indigo', 'pink',
                       'lightgreen', 'darkorange', 'peru', 'steelblue', 'turquoise']
        self.linestyles = ['-']  # , '--', '-.', ':', '-', '--', '-.', ':']
        self.lineWidth = 2.0
        self.jointcolors = [(0.0, 0.0, 1.0), (0.0, 0.5, 0.0), (1.0, 0.0, 0.0), (0.0, 0.75, 0.75),
                            (0.75, 0, 0.75), (0.75, 0.75, 0), (0.0, 0.0, 0.0)]

        self.outputPath = './results'
        self.visiblemask = numpy.ones((self.gt.shape[0], self.gt.shape[1], 3))

        self.jointNames = None
        self.jointConnections = []
        self.jointConnectionColors = []
        self.plotMaxJointDist = 80
        self.plotMeanJointDist = 80
        self.plotMedianJointDist = 80
        self.VTKviewport = [0, 0, 0, 0, 0]


    def getMeanError(self):
        """
        get average error over all joints, averaged over sequence
        :return: mean error
        """
        return numpy.nanmean(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1).mean()


    def getStdError(self):
        """
        get standard deviation of error over all joints, averaged over sequence
        :return: standard deviation of error
        """
        return numpy.nanstd(numpy.sqrt(
            numpy.square(self.gt - self.joints).sum(axis=2)), axis=1).mean()


    def getMaxError(self):
        """
        get max error over all joints, averaged over sequence
        :return: maximum error
        """
        return numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2))).mean()


    def getMeanErrorPerFrame(self):
        """
        get average error over all joints, for each frame
        :return: mean error per frame
        """
        return numpy.nanmean(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1)


    def getMaxErrorPerFrame(self):
        """
        get max error over all joints for each image of sequence
        :return: maximum error per frame
        """
        return numpy.nanmax(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1)


    def getJointMeanError(self, jointID):
        """
        get error of one joint, averaged over sequence
        :param jointID: joint ID
        :return: mean joint error
        """
        return numpy.nanmean(numpy.sqrt(numpy.square(self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))


    def getJointStdError(self, jointID):
        """
        get standard deviation of one joint, averaged over sequence
        :param jointID: joint ID
        :return: standard deviation of joint error
        """
        return numpy.nanstd(numpy.sqrt(
            numpy.square(
            self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))


    def getJointMaxError(self, jointID):
        """
        get maximum error of one joint
        :param jointID: joint ID
        :return: maximum joint error
        """
        return numpy.nanmax(numpy.sqrt(numpy.square(
            self.gt[:, jointID, :] - self.joints[:, jointID, :]).sum(axis=1)))


    def getNumFramesWithinMaxDist(self, dist):
        """
        calculate the number of frames where the maximum difference of a joint is within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.nanmax(numpy.sqrt(
            numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()


    def getNumFramesWithinMeanDist(self, dist):
        """
        calculate the number of frames where the mean difference over all joints of a hand are within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.nanmean(numpy.sqrt(
            numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()


    def getNumFramesWithinMedianDist(self, dist):
        """
        calculate the number of frames where the median difference over all joints of a hand are within dist mm
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.median(numpy.sqrt(
            numpy.square(self.gt - self.joints).sum(axis=2)), axis=1) <= dist).sum()
        
        
    def getNumJointsWithinDist(self, dist):
        """
        calculate the number of joints within dist mm to ground truth
        :param dist: distance between joint and GT
        :return: number of frames
        """
        return (numpy.sqrt(
            numpy.square(self.gt - self.joints).sum(axis=2)) <= dist).sum()


    def getMDscore(self, dist):
        """
        Calculate the max dist score, ie. MD=\int_0^d{\frac{|F<x|}{|F|}dx = \sum
        That is the area-under-the-curve (AuC) for the frame based success rate 
        up to the given dist.
        :param dist: distance between joint and GT
        :return: score value [0-1]
        """
        vals = [(numpy.nanmax(numpy.sqrt(numpy.square(
            self.gt - self.joints).sum(axis=2)), axis=1) <= j).sum() 
            / float(self.joints.shape[0]) for j in range(0, dist)]
        return numpy.asarray(vals).sum() / float(dist)
        
        
    def getJointSuccesAuC(self, dist):
        """
        Compute the area-under-the-curve (AuC) score for the joint based 
        success rate (i.e., AuC for 'joints within distance' plot up to given 
        dist). This is the 'MD-score' for individual joints (instead of frames)
        :param dist: distance between joint and GT
        :return: score value [0-1]
        """
        vals = [(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)) <= j).sum() \
            / float(self.joints.shape[0] * self.joints.shape[1]) for j in range(0, dist)]
        return numpy.asarray(vals).sum() / float(dist)


    def plotEvaluation(self, basename, methodName='Our method', baseline=None, 
                       basePath=None, curveColors=None, lineWidth=None):
        """
        plot and save standard evaluation plots
        :param basename: file basename, i.e., prefix for (output) filenames
        :param methodName: our method name
        :param baseline: list of baselines as tuple (Name,evaluation object)
        :param basePath: if not None this path is used as base-path instead of 
                         the default path (self.outputPath)
        :param curveColors: list of colors for each method (see self.colors for default)
        :param lineWidth: line width for the curves
        :return: None
        """
        # Sanity check
        if baseline is not None:
            for bs in baseline:
                if not (isinstance(bs[1], self.__class__)):
                    raise TypeError('baseline must be of type {} but {} provided'.format(
                        self.__class__.__name__, bs[1].__class__.__name__))
                                                                                                                                                                                  
        if basePath is None:
            basePath = self.outputPath
            
        if curveColors is None:
            curveColors = self.colors
            
        if lineWidth is None:
            lineWidth = self.lineWidth
            
        if not os.path.exists(basePath):
            os.makedirs(basePath)

        # plot number of frames within max distance
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([self.getNumFramesWithinMaxDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMaxJointDist)],
                label=methodName, c=curveColors[0], linestyle=self.linestyles[0], linewidth=lineWidth)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.plot([bs[1].getNumFramesWithinMaxDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMaxJointDist)],
                        label=bs[0], c=curveColors[bs_idx % len(curveColors)], 
                        linestyle=self.linestyles[bs_idx % len(self.linestyles)], 
                        linewidth=lineWidth)
                bs_idx += 1
        plt.xlabel('Distance threshold / mm')
        plt.ylabel('Fraction of frames with max distance within threshold / %')
        plt.ylim([0.0, 100.0])
        ax.grid(True)
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig(os.path.join(basePath,'{}_framesMaxWithin.pdf'.format(basename)), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        
        # plot number of frames within; mean distance
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([self.getNumFramesWithinMeanDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMeanJointDist)],
                label=methodName, c=curveColors[0], linestyle=self.linestyles[0], 
                linewidth=lineWidth)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.plot([bs[1].getNumFramesWithinMeanDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMeanJointDist)],
                        label=bs[0], c=curveColors[bs_idx % len(curveColors)], 
                        linestyle=self.linestyles[bs_idx % len(self.linestyles)], 
                        linewidth=lineWidth)
                bs_idx += 1
        plt.xlabel('Distance threshold / mm')
        plt.ylabel('Fraction of frames with mean distance within threshold / %')
        plt.ylim([0.0, 100.0])
        ax.grid(True)
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig(os.path.join(basePath,'{}_framesMeanWithin.pdf'.format(basename)), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        
        # plot number of frames within; median distance
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([self.getNumFramesWithinMedianDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMedianJointDist)],
                label=methodName, c=curveColors[0], linestyle=self.linestyles[0], 
                linewidth=lineWidth)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.plot([bs[1].getNumFramesWithinMedianDist(j) / float(self.joints.shape[0]) * 100. for j in range(0, self.plotMedianJointDist)],
                        label=bs[0], c=curveColors[bs_idx % len(curveColors)], 
                        linestyle=self.linestyles[bs_idx % len(self.linestyles)], 
                        linewidth=lineWidth)
                bs_idx += 1
        plt.xlabel('Distance threshold / mm')
        plt.ylabel('Fraction of frames with median distance within threshold / %')
        plt.ylim([0.0, 100.0])
        ax.grid(True)
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig(os.path.join(basePath,'{}_framesMedianWithin.pdf'.format(basename)), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        
        # plot number of joints within
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([self.getNumJointsWithinDist(j) / float(self.joints.shape[0] * self.joints.shape[1]) * 100. for j in range(0, self.plotMaxJointDist)],
                label=methodName, c=curveColors[0], linestyle=self.linestyles[0], 
                linewidth=lineWidth)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.plot([bs[1].getNumJointsWithinDist(j) / float(self.joints.shape[0] * self.joints.shape[1]) * 100. for j in range(0, self.plotMaxJointDist)],
                        label=bs[0], c=curveColors[bs_idx % len(curveColors)], 
                        linestyle=self.linestyles[bs_idx % len(self.linestyles)], 
                        linewidth=lineWidth)
                bs_idx += 1
        plt.xlabel('Distance threshold / mm')
        plt.ylabel('Fraction of joints within threshold to ground truth / %')
        plt.ylim([0.0, 100.0])
        ax.grid(True)
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig(os.path.join(basePath,'{}_jointsWithin.pdf'.format(basename)), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        
        # plot mean error per frame
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(numpy.nanmean(numpy.sqrt(numpy.square(self.gt - self.joints).sum(axis=2)), axis=1),
                label=methodName, c=curveColors[0], linestyle=self.linestyles[0], 
                linewidth=lineWidth)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.plot(numpy.nanmean(numpy.sqrt(numpy.square(self.gt - bs[1].joints).sum(axis=2)), axis=1),
                        label=bs[0], c=curveColors[bs_idx % len(curveColors)], 
                        linestyle=self.linestyles[bs_idx % len(self.linestyles)], 
                        linewidth=lineWidth)
                bs_idx += 1
        plt.xlabel('frame ID')
        plt.ylabel('Error / mm')
        plt.ylim([0.0, 150.0])
        ax.grid(True)
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='lower right', ncol=1) #, bbox_to_anchor=(0.5,-0.1)
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig(os.path.join(basePath,'{}_frameerror.pdf'.format(basename)), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        
        # plot mean error for each joint
        ind = numpy.arange(self.joints.shape[1]+1)  # the x locations for the groups, +1 for mean
        if baseline is not None:
            width = (1 - 0.33) / (1. + len(baseline))  # the width of the bars
        else:
            width = 0.67
        fig, ax = plt.subplots()
        mean = [self.getJointMeanError(j) for j in range(self.joints.shape[1])]
        mean.append(self.getMeanError())
        std = [self.getJointStdError(j) for j in range(self.joints.shape[1])]
        std.append(self.getStdError())
        ax.bar(ind, numpy.array(mean), width, label=methodName, color=curveColors[0])  # , yerr=std)
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                mean = [bs[1].getJointMeanError(j) for j in range(self.joints.shape[1])]
                mean.append(bs[1].getMeanError())
                std = [bs[1].getJointStdError(j) for j in range(self.joints.shape[1])]
                std.append(bs[1].getStdError())
                ax.bar(ind + width * float(bs_idx), numpy.array(mean), width,
                       label=bs[0], color=curveColors[bs_idx % len(curveColors)])  # , yerr=std)
                bs_idx += 1
        ax.set_xticks(ind + width)
        ll = list(self.jointNames)
        ll.append('Avg')
        label = tuple(ll)
        ax.set_xticklabels(label)
        plt.ylabel('Mean error of joint / mm')
        # plt.ylim([0.0,50.0])
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig(os.path.join(basePath,'{}_joint_mean.pdf'.format(basename)), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)

        # plot maximum error for each joint
        ind = numpy.arange(self.joints.shape[1])  # the x locations for the groups
        if baseline is not None:
            width = (1 - 0.33) / (1. + len(baseline))  # the width of the bars
        else:
            width = 0.67
        fig, ax = plt.subplots()
        ax.bar(ind, numpy.array([self.getJointMaxError(j) for j in range(self.joints.shape[1])]), 
               width, label=methodName, color=curveColors[0])
        bs_idx = 1
        if baseline is not None:
            for bs in baseline:
                ax.bar(ind + width * float(bs_idx),
                       numpy.array([bs[1].getJointMaxError(j) for j in range(self.joints.shape[1])]), 
                       width, label=bs[0], color=curveColors[bs_idx % len(curveColors)])
                bs_idx += 1
        ax.set_xticks(ind + width)
        ax.set_xticklabels(self.jointNames)
        plt.ylabel('Maximum error of joint / mm')
        plt.ylim([0.0, 200.0])
        # Put a legend below current axis
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
        plt.show(block=False)
        fig.savefig(os.path.join(basePath,'{}_joint_max.pdf'.format(basename)), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        
    
    def writeQualitativeResults(self, data, crop_transforms, predictions, targets, test_loader, args):
        """
        Plot qualitative results on image data and write to disk
        :param data: image crops
        :param crop_transforms: transformation matrices specifying tranformation 
            of each crop as returned by the data loader
        :param predictions: predicted joint positions
        :param targets: ground truth joint positions
        :param test_loader: dataset loader (only used for proper transformations)
        :param args: additional paramters
        """
        for ind in range(0, predictions.shape[0], args.step_result_overlay):
            jt = predictions[ind]
            gt = targets[ind]
            jtI = test_loader.dataset.points3DToImg(jt)
            gtI = test_loader.dataset.points3DToImg(gt)
            for joint in range(jt.shape[0]):
                t = transformPoint2D(jtI[joint], crop_transforms[ind])
                jtI[joint, 0] = t[0]
                jtI[joint, 1] = t[1]
                t_g = transformPoint2D(gtI[joint], crop_transforms[ind])
                gtI[joint, 0] = t_g[0]
                gtI[joint, 1] = t_g[1]
            self.plotResult(data[ind,0], gtI, jtI, "{}_i{}".format(args.exp_name, ind))


    def plotResult(self, dpt, gtcrop, joint, name=None, showGT=True, 
                   niceColors=False, showJoints=True, showDepth=True, 
                   basePath=None, connection_linewidth=2.0):
        """
        Show the annotated depth image
        :param dpt: depth image to show
        :param gtcrop: cropped 2D coordinates
        :param joint: joint data
        :param name: name of file to save, if None return image
        :param showGT: show groundtruth annotation
        :param niceColors: plot nice gradient colors for each joint
        :param basePath: if not None this path is used as base-path instead of 
                         the default path (self.outputPath)
        :return: None, or image if name = None
        """
                                                                                                                                                   
        if basePath is None:
            basePath = self.outputPath
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('off')
        # plot depth image with annotations
        if showDepth:
            imgcopy = dpt.copy()
            # display hack to hide nd depth
            msk = imgcopy > 0
            msk2 = imgcopy == 0
            if msk.max() == True:
                min = imgcopy[msk].min()
                max = imgcopy[msk].max()
                if (max - min) > 0:
                    imgcopy = (imgcopy - min) / (max - min) * 255.
            imgcopy[msk2] = 255.
            ax.imshow(imgcopy, cmap='gray')
        else:
            # same view as with image
            ax.set_xlim([0, dpt.shape[0]])
            ax.set_ylim([0, dpt.shape[1]])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gca().invert_yaxis()
        # use child class plots
        if showJoints:
            self.plotJoints(ax, joint, color=('r' if niceColors is False else 'nice'), 
                            jcolor=('r' if niceColors is False else None), 
                            connection_linewidth=connection_linewidth)  # ours
        if showGT:
            self.plotJoints(ax, gtcrop, color=('b' if niceColors is False else 'nice'), 
                            jcolor=('b' if niceColors is False else None), 
                            connection_linewidth=connection_linewidth)  # groundtruth
        plt.tight_layout(pad=0)
        plt.show(block=False)
        if name is not None:
            fig.savefig(os.path.join(basePath,'annotated_{}.png'.format(name)), bbox_inches='tight')
            plt.close(fig)
        else:
            # If we haven't already shown or saved the plot, then we need to draw the figure first...
            fig.patch.set_facecolor('w')
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return data
            
            
    def plotJoints(self, ax, joint, color='nice', jcolor=None, connection_linewidth=2.0):
        """
        Plot connected joints
        :type ax: axis to plot on
        :type joint: joints to connect
        :type color: line color
        """
        # Draw skeleton
        for i in range(len(self.jointConnections)):
            ax.plot(numpy.hstack((joint[self.jointConnections[i][0], 0], joint[self.jointConnections[i][1], 0])),
                    numpy.hstack((joint[self.jointConnections[i][0], 1], joint[self.jointConnections[i][1], 1])),
                    c=(color if color is not 'nice' else self.jointConnectionColors[i]), 
                    linewidth=connection_linewidth)
        # Draw joints                    
        color_index = 0
        for i in range(joint.shape[0]):
            idx = color_index % len(self.jointcolors)
            ax.scatter(joint[i, 0], joint[i, 1], 
                       c=(self.jointcolors[idx] if jcolor is None else jcolor), 
                       marker='.', s=400)
            color_index += 1
                    
                    
    def writeResults2Textfile(self, filepath):
        with open(filepath, 'w') as f:
            f.write("Mean-error(mm): {}\n".format(self.getMeanError()))
            f.write("Max-error(mm): {}\n".format(self.getMaxError()))
            f.write("Std-error(mm): {}\n".format(self.getStdError()))
            dists = [20, 30, 40, 50, 60, 70, 80]
            for d in dists:
                f.write("MD-score({}mm): {}\n".format(d, self.getMDscore(d)))
            f.write("Joint-Mean-error(mm): {}\n".format(
                [self.getJointMeanError(j) for j in range(self.gt[0].shape[0])]))
            f.write("Joint-Max-error(mm): {}\n".format(
                [self.getJointMaxError(j) for j in range(self.gt[0].shape[0])]))
            for d in dists:
                f.write("JSAuC({}mm): {}\n".format(d, self.getJointSuccesAuC(d)))


class NYUHandposeEvaluation(HandposeEvaluation):
    """
    Different evaluation metrics for handpose specific for NYU dataset
    """

    def __init__(self, gt, joint, joints=NyuAnnoType.EVAL_JOINTS_ORIGINAL):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joint: predicted joints
        """

        super(NYUHandposeEvaluation, self).__init__(gt, joint)
        import matplotlib

        # setup specific stuff
        if joints == NyuAnnoType.ALL_JOINTS:
            self.jointNames = ('P1', 'P2', 'P3', 'P4', 'P5', 'P6',
                               'R1', 'R2', 'R3', 'R4', 'R5', 'R6',
                               'M1', 'M2', 'M3', 'M4', 'M5', 'M6',
                               'I1', 'I2', 'I3', 'I4', 'I5', 'I6',
                               'T1', 'T2', 'T3', 'T4', 'T5', 
                               'C1', 'C2', 'C3',
                               'W1', 'W2', 'W3', 'W4')
            self.jointConnections = [[33, 5], [5, 4], [4, 3], [3, 2], [2, 1], [1, 0],
                                     [32, 11], [11, 10], [10, 9], [9, 8], [8, 7], [7, 6],
                                     [32, 17], [17, 16], [16, 15], [15, 14], [14, 13], [13, 12],
                                     [32, 23], [23, 22], [22, 21], [21, 20], [20, 19], [19, 18],
                                     [34, 29], [29, 28], [28, 27], [27, 26], [26, 25], [25, 24],
                                     [34, 32], [34, 33], [33, 32],
                                     [34, 30], [34, 31], [35, 30], [35, 31]]
            self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.2]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.3]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.4]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.0]]]))[0, 0],
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1.0]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1.0]]]))[0, 0]]

        elif joints == NyuAnnoType.EVAL_JOINTS_ORIGINAL:
            self.jointNames = ('P1', 'P2', 'R1', 'R2', 'M1', 'M2', 
                               'I1', 'I2', 'T1', 'T2', 'T3', 'W1', 'W2', 'C')
            # re-ordered joint conncections to better correspond to joints (fingers from tip to "mcp"-joint)
            self.jointConnections = [[1, 0], [13, 1], # pinky
                                     [3, 2], [13, 3], # ring
                                     [5, 4], [13, 5], # middle
                                     [7, 6], [13, 7], # index
                                     [9, 8], [10, 9], # thumb
                                     [13, 10], [13, 11], [13, 12]] # palm
            self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 1]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.7]]]))[0, 0], # pinky
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.7]]]))[0, 0], # ring
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.7]]]))[0, 0], # middle
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.7]]]))[0, 0], # index
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], # thumb
                                          matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 0.7]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.16, 1, 1]]]))[0, 0]]
            self.jointcolors = self.jointConnectionColors + [self.jointConnectionColors[-1]]

        else:
            raise ValueError("Unknown annotation type")
        self.plotMaxJointDist = 80
        self.VTKviewport = [0, 0, 0, 180, 40]
        self.fps = 25.0


class ICGHandposeEvaluation(HandposeEvaluation):
    """
    Evaluation for specific annotation (24 annotated joint positions)
    """

    def __init__(self, gt, joints):
        """
        Initialize class

        :type gt: groundtruth joints
        :type joints: calculated joints
        """

        super(ICGHandposeEvaluation, self).__init__(gt, joints)
        import matplotlib

        # setup specific stuff
        self.jointNames = ( 'PalmT',                    # palm (thumb)
                            'T1', 'T2', 'T3',           # thumb
                            'PalmI',                    # palm (index)
                            'I1', 'I2', 'I3', 'I4',     # index
                            'PalmM',                    # palm (middle)
                            'M1', 'M2', 'M3', 'M4',     # middle
                            'PalmR',                    # palm (ring)
                            'R1', 'R2', 'R3', 'R4',     # ring
                            'PalmP',                    # palm (pinky)
                            'P1', 'P2', 'P3', 'P4',     # pinky
                            )
        self.jointConnections = [[0, 1], [1, 2], [2, 3],                    # thumb
                                 [4, 5], [5, 6], [6, 7], [7, 8],            # index
                                 [9, 10], [10, 11], [11, 12], [12, 13],     # middle
                                 [14, 15], [15, 16], [16, 17], [17, 18],    # ring
                                 [19, 20], [20, 21], [21, 22], [22, 23]]    # pinky
        self.jointConnectionColors = [matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.00, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.33, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.50, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 0.8]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.66, 1, 1]]]))[0, 0],
                                      matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.6]]]))[0, 0], matplotlib.colors.hsv_to_rgb(numpy.asarray([[[0.83, 1, 0.8]]]))[0, 0]]

        self.plotMaxJointDist = 80
        self.VTKviewport = [0, 0, 180, 40, 40]
        self.fps = 10.0

