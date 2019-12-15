# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stat
import scipy.signal as signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def edge_dec(orig, mod):
    '''See modified picture and original side-by-side'''
    plt.clf()
    plt.subplot(122),plt.imshow(mod, cmap = "gray")
    plt.title('Modded Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(121),plt.imshow(orig, cmap = "gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

def show_stats(mod, stats1, stats2):
    '''Display the axes extracted from the picture on the picture'''
    plt.clf()
    plt.subplot(121), plt.imshow(mod, cmap = "gray")
    plt.subplot(121), plt.plot(stats1)

    plt.subplot(122), plt.imshow(np.rot90(mod), cmap = "gray")
    plt.subplot(122), plt.plot(stats2)

def get_size(img):
    """Return the size of the image in pixels."""
    ih, iw = img.shape[:2]
    return iw * ih

def crop(img, boundaries):
    """Crop the image to the given boundaries."""
    minx, miny, maxx, maxy = boundaries
    return img[miny:maxy, minx:maxx]

def check_for_rect(mat, edge_coverage = 3, thresh = 1.98):
    '''Checker function for controlling the presence of lines
    with z-scores over 3.5 in quarters of edge detected images'''
    dims = mat.shape

    ax_vert = stat.zscore(np.std(mat, axis=1))
    ax_vert_left = ax_vert[:int(dims[0]/edge_coverage)]
    ax_vert_right = ax_vert[int(dims[0] * (edge_coverage - 1)/edge_coverage):]

    ax_hori = stat.zscore(np.std(mat, axis=0))
    ax_hori_upper = ax_hori[:int(dims[1]/edge_coverage)]
    ax_hori_lower = ax_hori[int(dims[1] * (edge_coverage - 1)/edge_coverage):]

    # check for 3.5 (99.5 limit) in the corners and that
    # the area is smaller than third of axis length
    return (np.sum( ax_vert_left[ax_vert_left > thresh] ) / dims[0] > 0 and
       np.sum( ax_vert_right[ax_vert_right > thresh] ) / dims[0] > 0 and
       np.sum( ax_hori_upper[ax_hori_upper > thresh] ) / dims[1] > 0 and
       np.sum( ax_hori_lower[ax_hori_lower > thresh] ) / dims[1] > 0 and
       np.sum( ax_vert[ax_vert > thresh]) / dims[0] < 0.33 and
       np.sum( ax_hori[ax_hori > thresh]) / dims[1] < 0.33)

def detect_rect(mat, minlineprop):
    '''Detect lines from picture (mat) that are horizontal and rectangular
    and minimum length of defined proprtion (minlineprop) of picture axis
    length'''
    dims = mat.shape

    vertic_struct = cv2.getStructuringElement(cv2.MORPH_RECT,
                                              (1, int(dims[0]*minlineprop)))
    vertic = cv2.erode(mat, vertic_struct)
    vertic = cv2.dilate(vertic, vertic_struct)

    # detect horizontal lines
    horiz_struct = cv2.getStructuringElement(cv2.MORPH_RECT,
                                             (int(dims[1]*minlineprop), 1))
    horiz = cv2.erode(mat, horiz_struct)
    horiz = cv2.dilate(horiz, horiz_struct)
    return vertic + horiz

def get_rect_bounds(mat, edge_coverage = 3, thresh = 1.98):
    '''Get the minx, miny, maxx, maxy of detected rectangle on the picture,
    from the area from the edge of edge_coverage'th of picture measured from the
    edge'''
    dims = mat.shape
    ax_vert = stat.zscore(np.std(mat, axis=1))
    ax_vert_left = ax_vert[:int(dims[0] / edge_coverage)][::-1]
    ax_vert_right = ax_vert[int(dims[0] * ( edge_coverage-1 )/edge_coverage):]

    ax_hori = stat.zscore(np.std(mat, axis=0))
    ax_hori_upper = ax_hori[:int(dims[1] / edge_coverage)][::-1]
    ax_hori_lower = ax_hori[int(dims[1] * ( edge_coverage-1 )/edge_coverage):]

    # find first occurrence of qualifing line for frame
    return [int( dims[1] / edge_coverage - np.where(ax_hori_upper > thresh)[0][0] ),
                 int( dims[0] / edge_coverage - np.where(ax_vert_left > thresh)[0][0] ),
                 int( np.where(ax_hori_lower > thresh)[0][0] + dims[1] * ( edge_coverage-1 )/edge_coverage ),
                 int( np.where(ax_vert_right > thresh)[0][0] + dims[0] * ( edge_coverage-1 )/edge_coverage )]

def detect_rot_rect(mat, minlineprop, rotrange, edge_coverage = 3):
    '''Detect lines that are horizontal and rectangular and at least
    2/3 of picture length. Finds also slightly rotated lines by image rotation'''
    checkrange = np.insert(
        np.arange(-int(rotrange / 2), int(rotrange / 2), 0.5), 0, 0)
    test = mat.copy()

    for degree in checkrange:
        res = detect_rect(test, minlineprop)
        if check_for_rect(res, thresh=1.98):
            print("Rotated", degree, "degrees.", end = '\n')
            return res, degree
        else:
            print("Rotate:", degree, "degrees.", end = '\r')
            test = ndimage.rotate(mat, degree, reshape = False)
    return 0, 0

def preprocess(img):
    # denoise for more edgy picture
    edg = cv2.fastNlMeansDenoisingColored(img, None, 8, 7, 12)
    # Canny edge detection
    edg = cv2.Canny(edg, 20, 250, apertureSize = 3)
    #edg = cv2.Laplacian(edg, cv2.CV_8U)
    # blur the edges slightly for smoother lines
    edg = ndimage.gaussian_filter(edg, 2.1)
    # see the detected lines:
    return edg

def process(img, rotate):
    edg = preprocess(img)
    # main line-based frame detection
    rectd, degr = detect_rot_rect(edg, .58, rotate, 4)

    if rectd is not 0:
        # rotate the original image to correct angle
        if degr != 0:
            img = ndimage.rotate(img, degr, reshape = False)

        # crop the frame
        frames = get_rect_bounds(rectd, thresh=1.98)
        proc_img = crop(img, frames)

        # else recurse the frame picture until no
        # good frames are to be found.

        if get_size(proc_img) > get_size(img) / 1.6:
            return process(proc_img, 4)
        else:
            return proc_img
    else:
        return img

def save(mat, filename, extra):
    outname = os.path.dirname(
        filename) + "/" + os.path.splitext(os.path.basename(
            filename))[0] + extra
    cv2.imwrite(outname, mat)
    return outname

input_file = 'img/Lembit0008.JPG'
rotation = 10
img = cv2.imread(input_file, 1)
proc_img = process(img, rotation)

edge_dec(img, proc_img)

# outname = save(proc_img, input_file, "_crop.png")
# print("Found cut and printed:", outname)

# NOTE: NOT USED!
# def apply_hough(mat, thresh = 110, maxgap = 3, minline = 50):
#     '''Detect and create mask of lines from the picture.
#     Hough method, probabilistic line detection but not accurate enough
#     for standalone usage. Mayby add it before rectangle detection for more
#     flexible frame detection?'''
#     dims = mat.shape
#     blank = np.zeros(dims)
#     lines = cv2.HoughLinesP(mat,
#                             rho = 1,
#                             theta = np.pi/180,
#                             threshold = thresh,
#                             minLineLength = minline,
#                             maxLineGap = maxgap)
# 
#     for x1,y1,x2,y2 in np.squeeze( lines ):
#         cv2.line(blank,(x1,y1),(x2,y2),(254),1)
#     return blank
