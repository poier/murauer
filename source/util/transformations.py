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

import cv2
import numpy


def getTransformationMatrix(center, rot, trans, scale):
    ca = numpy.cos(rot)
    sa = numpy.sin(rot)
    sc = scale
    cx = center[0]
    cy = center[1]
    tx = trans[0]
    ty = trans[1]
    t = numpy.array([ca * sc, -sa * sc, sc * (ca * (-tx - cx) + sa * ( cy + ty)) + cx,
                     sa * sc, ca * sc, sc * (ca * (-ty - cy) + sa * (-tx - cx)) + cy])
    return t


def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = numpy.asmatrix(M.reshape((3, 3))) * numpy.matrix([pt[0], pt[1], 1]).T
    return numpy.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoint3D(pt, M):
    """
    Transform point in 3D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt3 = numpy.asmatrix(M.reshape((4, 4))) * numpy.matrix([pt[0], pt[1], pt[2], 1]).T
    return numpy.array([pt3[0] / pt3[3], pt3[1] / pt3[3], pt3[2] / pt3[3]])
    

def pointsImgTo3D(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D
    :param sample: points in (x,y,z) with x,y in image coordinates and z in mm
    z is assumed to be the distance from the camera plane (i.e., not camera center)
    :return: normalized points in mm
    """
    ret = numpy.zeros((sample.shape[0], 3), numpy.float32)
    for i in range(sample.shape[0]):
        ret[i] = pointImgTo3D(sample[i], fx, fy, ux, uy)
    return ret
    

def pointImgTo3D(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D
    :param sample: point in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized points in mm
    """
    ret = numpy.zeros((3,), numpy.float32)
    # convert to metric using f
    ret[0] = (sample[0]-ux)*sample[2]/fx
    ret[1] = (sample[1]-uy)*sample[2]/fy
    ret[2] = sample[2]
    return ret
    

def points3DToImg(sample, fx, fy, ux, uy):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: points in (x,y,z) with x,y and z in mm
    :return: points in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = numpy.zeros((sample.shape[0], 3), numpy.float32)
    for i in range(sample.shape[0]):
        ret[i] = point3DToImg(sample[i], fx, fy, ux, uy)
    return ret
    

def point3DToImg(sample, fx, fy, ux, uy):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: points in (x,y,z) with x,y and z in mm
    :return: points in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = numpy.zeros((3,), numpy.float32)
    # convert to metric using f
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0]/sample[2]*fx+ux
    ret[1] = sample[1]/sample[2]*fy+uy
    ret[2] = sample[2]
    return ret
    
    
def pointsImgTo3D_NYU(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D (NYU dataset specific, see code from Tompson)
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    ret = numpy.zeros((sample.shape[0],3), numpy.float32)
    for i in range(sample.shape[0]):
        ret[i] = pointImgTo3D_NYU(sample[i], fx, fy, ux, uy)
    return ret


def pointImgTo3D_NYU(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D (NYU dataset specific, see code from Tompson)
    :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized joints in mm
    """
    ret = numpy.zeros((3,), numpy.float32)
    # convert to metric using f, see Thomson et al.
    ret[0] = (sample[0] - ux) * sample[2] / fx
    ret[1] = (uy - sample[1]) * sample[2] / fy
    ret[2] = sample[2]
    return ret


def points3DToImg_NYU(sample, fx, fy, ux, uy):
    """
    Denormalize sample from metric 3D to image coordinates (NYU dataset specific, 
    see code from Tompson)
    :param sample: joints in (x,y,z) with x,y and z in mm
    :return: joints in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = numpy.zeros((sample.shape[0], 3), numpy.float32)
    for i in range(sample.shape[0]):
        ret[i] = point3DToImg_NYU(sample[i], fx, fy, ux, uy)
    return ret


def point3DToImg_NYU(sample, fx, fy, ux, uy):
    """
    Denormalize sample from metric 3D to image coordinates (NYU dataset specific, 
    see code from Tompson)
    :param sample: joints in (x,y,z) with x,y and z in mm
    :return: joints in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = numpy.zeros((3,), numpy.float32)
    #convert to metric using f, see Thomson et.al.
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0]/sample[2]*fx+ux
    ret[1] = uy-sample[1]/sample[2]*fy
    ret[2] = sample[2]
    return ret


def rotateImageAndGt(imgDepth, gtUvd, gt3d, angle,
                     fx, fy, cx, cy, jointIdRotCenter, pointsImgTo3DFunction, bgValue=10000):
    """
    :param angle:   rotation angle
    :param pointsImgTo3DFunction:   function which transforms a set of points 
        from image coordinates to 3D coordinates
        like transformations.pointsImgTo3D() (from the same file).
        (To enable specific projections like for the NYU dataset)
    """
    # Rotate image around given joint
    jtId = jointIdRotCenter
    center = (gtUvd[jtId][0], gtUvd[jtId][1])
    rotationMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    sizeRotImg = (imgDepth.shape[1], imgDepth.shape[0])
    imgRotated = cv2.warpAffine(src=imgDepth, M=rotationMat, 
                                dsize=sizeRotImg, flags=cv2.INTER_NEAREST, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=bgValue)
    
    # Rotate GT
    gtUvd_ = gtUvd.copy()
    gtUvdRotated = numpy.ones((gtUvd_.shape[0], 3), dtype=gtUvd.dtype)
    gtUvdRotated[:,0:2] = gtUvd_[:,0:2]
    gtUvRotated = numpy.dot(rotationMat, gtUvdRotated.T)
    gtUvdRotated[:,0:2] = gtUvRotated.T
    gtUvdRotated[:,2] = gtUvd_[:,2]
    # normalized joints in 3D coordinates
    gt3dRotated = pointsImgTo3DFunction(gtUvdRotated, fx, fy, cx, cy)
    
    return imgRotated, gtUvdRotated, gt3dRotated
    