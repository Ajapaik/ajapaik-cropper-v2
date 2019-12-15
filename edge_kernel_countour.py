# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import scipy.stats as stat
import scipy.signal as signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from argh import ArghParser, arg

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

####
#### NOTE: from here on snatched from the git
####

def get_contours(img):
    """Threshold the image and get contours."""
    # First make the image 1-bit and get contours
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the right threshold level
    tl = 100
    ret, thresh = cv2.threshold(imgray, tl, 255, 0)
    while white_percent(thresh) > 0.55:
        tl += 10
        ret, thresh = cv2.threshold(imgray, tl, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # filter contours that are too large or small
    contours = [cc for cc in contours if contourOK(img, cc)]
    return contours

def get_boundaries(img, contours):
    """Find the boundaries of the photo in the image using contours."""
    # margin is the minimum distance from the edges of the image, as a fraction
    ih, iw = img.shape[:2]
    minx = iw
    miny = ih
    maxx = 0
    maxy = 0

    for cc in contours:
        x, y, w, h = cv2.boundingRect(cc)
        if x < minx: minx = x
        if y < miny: miny = y
        if x + w > maxx: maxx = x + w
        if y + h > maxy: maxy = y + h

    return (minx, miny, maxx, maxy)

def get_size(img):
    """Return the size of the image in pixels."""
    ih, iw = img.shape[:2]
    return iw * ih

def white_percent(img):
    """Return the percentage of the thresholded image that's white."""
    return cv2.countNonZero(img) / get_size(img)

def contourOK(img, cc):
    """Check if the contour is a good predictor of photo location."""
    # Dont check edges, they dont matter here
    # if near_edge(img, cc): return False # shouldn't be near edges
    x, y, w, h = cv2.boundingRect(cc)
    if w < 100 or h < 100: return False # too narrow or wide is bad
    area = cv2.contourArea(cc)
    if area > (get_size(img) * 0.3): return False
    if area < 200: return False
    return True

def crop(img, boundaries):
    """Crop the image to the given boundaries."""
    minx, miny, maxx, maxy = boundaries
    return img[miny:maxy, minx:maxx]

####
#### From here on original
####

def check_for_rect(mat):
    '''Checker function for controlling the presence of lines
    with z-scores over 3.5 in quarters of edge detected images'''
    dims = mat.shape

    ax_vert = stat.zscore(np.std(mat, axis=1))
    ax_vert_left = ax_vert[:int(dims[0]/4)]
    ax_vert_right = ax_vert[int(dims[0]*3/4):]

    ax_hori = stat.zscore(np.std(mat, axis=0))
    ax_hori_upper = ax_hori[:int(dims[1]/4)]
    ax_hori_lower = ax_hori[int(dims[1]*3/4):]

    # check for 3.5 (99.5 limit) in the corners and that
    # the area is smaller than third of axis length
    return (np.sum( ax_vert_left[ax_vert_left > 1.98] ) / dims[0] > 0 and
       np.sum( ax_vert_right[ax_vert_right > 1.98] ) / dims[0] > 0 and
       np.sum( ax_hori_upper[ax_hori_upper > 1.98] ) / dims[1] > 0 and
       np.sum( ax_hori_lower[ax_hori_lower > 1.98] ) / dims[1] > 0 and
       np.sum( ax_vert[ax_vert > 1.98] ) / dims[0] < 0.33 and
       np.sum( ax_hori[ax_hori > 1.98] ) / dims[1] < 0.33)

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

def detect_rot_rect(mat, minlineprop, rotrange):
    '''Detect lines that are horizontal and rectangular and 2/3 of picture length.
        Finds also slightly curved lines by image rotation'''
    checkrange = np.insert(
        np.arange(-int(rotrange / 2), int(rotrange / 2)), 0, 0)
    test = mat.copy()

    for degree in checkrange:
        res = detect_rect(test, minlineprop)
        if check_for_rect(res):
            print("Rotated", degree, "degrees.", end = '\n')
            return res, degree
        else:
            print("Rotate:", degree, "degrees.", end = '\r')
            test = ndimage.rotate(mat, degree, reshape = False)
    return 0, 0

def get_rect_bounds(mat):
    dims = mat.shape
    ax_vert = stat.zscore(np.std(mat, axis=1))
    ax_vert_left = ax_vert[:int(dims[0] / 4)][::-1]
    ax_vert_right = ax_vert[int(dims[0] * 3/4):]

    ax_hori = stat.zscore(np.std(mat, axis=0))
    ax_hori_upper = ax_hori[:int(dims[1] / 4)][::-1]
    ax_hori_lower = ax_hori[int(dims[1] * 3/4):]

    return [int( dims[1] / 4 - np.where(ax_hori_upper > 1.98)[0][0] ),
                 int( dims[0] / 4 - np.where(ax_vert_left > 1.98)[0][0] ),
                 int( np.where(ax_hori_lower > 1.98)[0][0] + dims[1] * 3/4 ),
                 int( np.where(ax_vert_right > 1.98)[0][0] + dims[0] * 3/4 )]

def process(img, rotate):
    dims = img.shape
    # denoise for more edgy picture
    edg = cv2.fastNlMeansDenoising(img, None, 8, 7, 12)
    # Canny edge detection
    edg = cv2.Canny(edg, 20, 250, apertureSize = 3)
    # blur the edges slightly for smoother lines
    edg = ndimage.gaussian_filter(edg, 2.1)
    # see the detected lines:
    #edge_dec(img, detect_rect( edg , .58))

    # main line-based frame detection
    rectd, degr = detect_rot_rect(edg, .58, rotate)

    if rectd is not 0:
        # rotate the original image to correct angle
        if degr != 0:
            img = ndimage.rotate(img, degr, reshape = False)

        # crop the frame
        frames = get_rect_bounds(rectd)
        proc_img = crop(img, frames)

        # apply the countour detection
        contours = get_contours(proc_img)
        bounds = get_boundaries(proc_img, contours)
        dproc_img = crop(proc_img, bounds)

        # check if countour picture sized appropriately
        if get_size(dproc_img) > get_size(proc_img) / 4 and get_size(dproc_img) != get_size(proc_img):
            return process( dproc_img, 4)

        # else recurse the frame picture until no
        # good frames are to be found.
        return process(proc_img, 4)
    else:
        return img

# MAIN
def mainer(input_file, rotation=20):
    img = cv2.imread(input_file, 1)
    proc_img = process(img, rotation)
    edge_dec(img, proc_img)

    if get_size(proc_img) < get_size(img) / 4 or get_size(proc_img) == get_size(img):
        print("Did not find a good cut, skipping picture.")
            # build here the multiple picture detector
        return None

    outname = os.path.dirname(
        input_file) + "/" + os.path.splitext(os.path.basename(
            input_file))[0] + "_crop.png"
    cv2.imwrite(outname, proc_img)
    print("Found cut and printed:", outname)
    return None

# RUNNER
if __name__ == '__main__':
    parser = ArghParser()
    parser.set_default_command(mainer)
    parser.dispatch()

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
