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

import math
import numpy
import cv2


class HandDetector(object):
    """
    Detect hand based on simple heuristic, centered at Center of Mass
    """

    def __init__(self):
        """
        """

    @staticmethod
    def bilinearResize(src, dsize, ndValue):
        """
        Bilinear resizing with sparing out not defined parts of the depth map
        :param src: source depth map
        :param dsize: new size of resized depth map
        :param ndValue: value of not defined depth
        :return:resized depth map
        """

        dst = numpy.zeros((dsize[1], dsize[0]), dtype=numpy.float32)

        x_ratio = float(src.shape[1] - 1) / dst.shape[1]
        y_ratio = float(src.shape[0] - 1) / dst.shape[0]
        for row in range(dst.shape[0]):
            y = int(row * y_ratio)
            y_diff = (row * y_ratio) - y  # distance of the nearest pixel(y axis)
            y_diff_2 = 1 - y_diff
            for col in range(dst.shape[1]):
                x = int(col * x_ratio)
                x_diff = (col * x_ratio) - x  # distance of the nearest pixel(x axis)
                x_diff_2 = 1 - x_diff
                y2_cross_x2 = y_diff_2 * x_diff_2
                y2_cross_x = y_diff_2 * x_diff
                y_cross_x2 = y_diff * x_diff_2
                y_cross_x = y_diff * x_diff

                # mathematically impossible, but just to be sure...
                if(x+1 >= src.shape[1]) | (y+1 >= src.shape[0]):
                    raise UserWarning("Shape mismatch")

                # set value to ND if there are more than two values ND
                numND = int(src[y, x] == ndValue) + int(src[y, x + 1] == ndValue) + int(src[y + 1, x] == ndValue) + int(
                    src[y + 1, x + 1] == ndValue)
                if numND > 2:
                    dst[row, col] = ndValue
                    continue
                # print y2_cross_x2, y2_cross_x, y_cross_x2, y_cross_x
                # interpolate only over known values, switch to linear interpolation
                if src[y, x] == ndValue:
                    y2_cross_x2 = 0.
                    y2_cross_x = 1. - y_cross_x - y_cross_x2
                if src[y, x + 1] == ndValue:
                    y2_cross_x = 0.
                    if y2_cross_x2 != 0.:
                        y2_cross_x2 = 1. - y_cross_x - y_cross_x2
                if src[y + 1, x] == ndValue:
                    y_cross_x2 = 0.
                    y_cross_x = 1. - y2_cross_x - y2_cross_x2
                if src[y + 1, x + 1] == ndValue:
                    y_cross_x = 0.
                    if y_cross_x2 != 0.:
                        y_cross_x2 = 1. - y2_cross_x - y2_cross_x2

                # print src[y, x], src[y, x+1],src[y+1, x],src[y+1, x+1]
                # normalize weights
                if not ((y2_cross_x2 == 0.) & (y2_cross_x == 0.) & (y_cross_x2 == 0.) & (y_cross_x == 0.)):
                    sc = 1. / (y_cross_x + y_cross_x2 + y2_cross_x + y2_cross_x2)
                    y2_cross_x2 *= sc
                    y2_cross_x *= sc
                    y_cross_x2 *= sc
                    y_cross_x *= sc
                # print y2_cross_x2, y2_cross_x, y_cross_x2, y_cross_x

                if (y2_cross_x2 == 0.) & (y2_cross_x == 0.) & (y_cross_x2 == 0.) & (y_cross_x == 0.):
                    dst[row, col] = ndValue
                else:
                    dst[row, col] = y2_cross_x2 * src[y, x] + y2_cross_x * src[y, x + 1] + y_cross_x2 * src[
                        y + 1, x] + y_cross_x * src[y + 1, x + 1]

        return dst
        
        
class HandDetectorICG(object):
    """
    Stump containing only the functionality needed here
    """
    
    RESIZE_BILINEAR = 0
    RESIZE_CV2_NN = 1
    RESIZE_CV2_LINEAR = 2

    def __init__(self):
        """
        Constructor
        """
        self.resizeMethod = self.RESIZE_CV2_NN


    def cropArea3D(self, imgDepth, com, fx, fy, minRatioInside=0.75, 
                   size=(250, 250, 250), dsize=(128, 128)):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """
        CROP_BG_VALUE = 0.0

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(math.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
        xend = int(math.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx))
        ystart = int(math.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy))
        yend = int(math.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy))
        
        # Check if part within image is large enough; otherwise stop
        xstartin = max(xstart,0)
        xendin = min(xend, imgDepth.shape[1])
        ystartin = max(ystart,0)
        yendin = min(yend, imgDepth.shape[0])        
        ratioInside = float((xendin - xstartin) * (yendin - ystartin)) / float((xend - xstart) * (yend - ystart))
        if (ratioInside < minRatioInside) \
                and ((com[0] < 0) \
                    or (com[0] >= imgDepth.shape[1]) \
                    or (com[1] < 0) or (com[1] >= imgDepth.shape[0])):
            print("Hand largely outside image (ratio (inside) = {})".format(ratioInside))
            raise UserWarning('Hand not inside image')

        # crop patch from source
        cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]), 
                           max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, imgDepth.shape[0])), 
                                      (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, imgDepth.shape[1]))), 
                            mode='constant', constant_values=int(CROP_BG_VALUE))
        msk1 = numpy.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = numpy.bitwise_and(cropped > zend, cropped != 0)
        # Backface is at 0, it is set later; 
        # setting anything outside cube to same value now (was set to zstart earlier)
        cropped[msk1] = CROP_BG_VALUE
        cropped[msk2] = CROP_BG_VALUE
        
        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = numpy.asmatrix(numpy.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        # Compute size of image patch for isotropic scaling 
        # where the larger side is the side length of the fixed size image patch (preserving aspect ratio)
        if wb > hb:
            sz = (dsize[0], int(round(hb * dsize[0] / float(wb))))
        else:
            sz = (int(round(wb * dsize[1] / float(hb))), dsize[1])

        # Compute scale factor from cropped ROI in image to fixed size image patch; 
        # set up matrix with same scale in x and y (preserving aspect ratio)
        roi = cropped
        if roi.shape[0] > roi.shape[1]: # Note, roi.shape is (y,x) and sz is (x,y)
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[1] / float(roi.shape[0]))
        else:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[0] / float(roi.shape[1]))
        scale[2, 2] = 1

        # depth resize
        if self.resizeMethod == self.RESIZE_CV2_NN:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)
        elif self.resizeMethod == self.RESIZE_BILINEAR:
            rz = HandDetector.bilinearResize(cropped, sz, CROP_BG_VALUE)
        elif self.resizeMethod == self.RESIZE_CV2_LINEAR:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")

        # Sanity check
        numValidPixels = numpy.sum(rz != CROP_BG_VALUE)
        if (numValidPixels < 40) or (numValidPixels < (numpy.prod(dsize) * 0.01)):
            print("Too small number of foreground/hand pixels: {}/{} ({}))".format(
                numValidPixels, numpy.prod(dsize), dsize))
            raise UserWarning("No valid hand. Foreground region too small.")

        # Place the resized patch (with preserved aspect ratio) 
        # in the center of a fixed size patch (padded with default background values)
        ret = numpy.ones(dsize, numpy.float32) * CROP_BG_VALUE  # use background as filler
        xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
        xend = int(xstart + rz.shape[1])
        ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        off = numpy.asmatrix(numpy.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, off * scale * trans, com
        