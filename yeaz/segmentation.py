# -*- coding: utf-8 -*-
import os
import sys
from .unet import unet, Model
import numpy as np
import skimage
from skimage import io
import skimage.transform as trans

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import dilation
from skimage.filters import gaussian
from skimage.measure import label
import numpy as np


def peak_local_max_as_mask(image, markers, indices):
  coords = peak_local_max(image, markers)
  mask = np.zeros(image.shape, dtype=bool)
  mask[tuple(coords.T)] = True
  return mask


def segment(th, pred, min_distance=10, topology=None): 
    """
    Performs watershed segmentation on thresholded image. Seeds have to
    have minimal distance of min_distance. topology defines the watershed
    topology to be used, default is the negative distance transform. 
    Can either be an array with the same size af th, or a function that will
    be applied to the distance transform.
    
    After watershed, the borders found by watershed will be evaluated in terms
    of their predicted value. If the borders are highly predicted to be cells,
    the two cells are merged. 
    """
    dtr = ndi.morphology.distance_transform_edt(th)
    if topology is None:
        topology = -dtr
    elif callable(topology):
        topology = topology(dtr)

    coords = peak_local_max(-topology, min_distance)
    m = np.zeros(topology.shape, dtype=np.bool)
    m[tuple(coords.T)] = True
    
    # Uncomment to start with cross for every pixel instead of single pixel
    m_lab = label(m) #comment this
    #m_dil = dilation(m)
    #m_lab = label(m_dil)
    wsh = watershed(topology, m_lab, mask=th, connectivity=2)
    merged = cell_merge(wsh, pred)
    return correct_artefacts(merged)
    
    
def correct_artefacts(wsh):
    """
    Sometimes artefacts arise with 3 or less pixels which are surrounded entirely
    by another cell. Those are removed here.
    """
    unique, count = np.unique(wsh, return_counts=True)
    to_remove = unique[count<=3]
    for rem in to_remove:
        rem_im = wsh==rem
        rem_cont = dilation(rem_im) & ~rem_im
        vals, val_counts = np.unique(wsh[rem_cont], return_counts=True)
        replace_val = vals[np.argmax(val_counts)]
        if replace_val != 0:
            wsh[rem_im] = int(replace_val)
    return wsh


def cell_merge(wsh, pred):
    """
    Procedure that merges cells if the border between them is predicted to be
    cell pixels.
    """
    wshshape=wsh.shape
    
    # masks for the original cells
    objs = np.zeros((wsh.max()+1,wshshape[0],wshshape[1]), dtype=bool)	
    
    # masks for dilated cells
    dil_objs = np.zeros((wsh.max()+1,wshshape[0],wshshape[1]), dtype=bool)
    
    # bounding box coordinates	
    obj_coords = np.zeros((wsh.max()+1,4))
    
    # cleaned watershed, output of function	
    wshclean = np.zeros((wshshape[0],wshshape[1]))
    
    # kernel to dilate objects
    kernel = np.ones((3,3), dtype=bool)	
    
    for obj1 in range(wsh.max()):
        # create masks and dilated masks for obj
        objs[obj1,:,:] = wsh==(obj1+1)	
        dil_objs[obj1,:,:] = dilation(objs[obj1,:,:], kernel)	
        
        # bounding box
        obj_coords[obj1,:] = get_bounding_box(dil_objs[obj1,:,:])
    
    objcounter = 0	# counter for new watershed objects
    
    for obj1 in range(wsh.max()):	
        dil1 = dil_objs[obj1,:,:]

        # check if mask has been deleted
        if np.sum(dil1) == 0:
            continue
        
        objcounter = objcounter + 1
        orig1 = objs[obj1,:,:]

        for obj2 in range(obj1+1,wsh.max()):
            dil2 = dil_objs[obj2,:,:]
        
            # only check border if bounding box overlaps, and second mask 
            # is not yet deleted
            if (do_box_overlap(obj_coords[obj1,:], obj_coords[obj2,:])
                and np.sum(dil2) > 0):
                
                border = dil1 * dil2	
                border_pred = pred[border]
                
                # Border is too small to be considered
                if len(border_pred) < 32:
                    continue
                
                # Sum of top 25% of predicted border values
                q75 = np.quantile(border_pred, .75)
                top_border_pred = border_pred[border_pred >= q75]
                top_border_height = top_border_pred.sum()
                top_border_area = len(top_border_pred)
                
                # merge cells
                if top_border_height / top_border_area > .99:
                    orig1 = np.logical_or(orig1, objs[obj2,:,:])
                    dil_objs[obj1,:,:] = np.logical_or(dil1, dil2)
                    dil_objs[obj2,:,:] = np.zeros((wshshape[0], wshshape[1]))
                    obj_coords[obj1,:] = get_bounding_box(dil_objs[obj1,:,:])
                    
        wshclean = wshclean + orig1*objcounter
            
    return wshclean


def do_box_overlap(coord1, coord2):
    """Checks if boxes, determined by their coordinates, overlap. Safety
    margin of 2 pixels"""
    return (
    (coord1[0] - 2 < coord2[0] and coord1[1] + 2 > coord2[0]
        or coord2[0] - 2 < coord1[0] and coord2[1] + 2 > coord1[0]) 
    and (coord1[2] - 2 < coord2[2] and coord1[3] + 2 > coord2[2]
        or coord2[2] - 2 < coord1[2] and coord2[3] + 2 > coord1[2]))

    
def get_bounding_box(im):
    """Returns bounding box of object in boolean image"""
    coords = np.where(im)
    
    return np.array([np.min(coords[0]), np.max(coords[0]), 
                     np.min(coords[1]), np.max(coords[1])])


def threshold(im,th = None):
    """
    Binarize an image with a threshold given by the user, or if the threshold is None, calculate the better threshold with isodata
    Param:
        im: a numpy array image (numpy array)
        th: the value of the threshold (feature to select threshold was asked by the lab)
    Return:
        bi: threshold given by the user (numpy array)
    """
    im2 = im.copy()
    if th == None:
        th = skimage.filters.threshold_isodata(im2)
    bi = im2
    bi[bi > th] = 255
    bi[bi <= th] = 0
    return bi

  
# -*- coding: utf-8 -*-
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import pandas as pd
import numpy as np

def get_model(model_filename, threshold=0.5):
    model = unet(pretrained_weights = model_filename, input_size = (None,None,1))
    model.__call__ = model.predict
    return 

  
def threshold(im, th=threshold):
    """
    Binarize an image with a threshold given by the user, or if the threshold is None, calculate the better threshold with isodata
    Param:
        im: a numpy array image (numpy array)
        th: the value of the threshold (feature to select threshold was asked by the lab)
    Return:
        bi: threshold given by the user (numpy array)
    """
    im = im.copy()
    if th == None:
        th = skimage.filters.threshold_isodata(im)
    im[im > th] = 255
    im[im <= th] = 0
    return im

  
def get_segmentation_mask(image, model_filename):
    predictor = model_filename
    if isinstance(predictor, str):
        predictor = get_model(model_filename, threshold=threshold)
    height, width = image.shape
    row_add = -height % 16
    col_add = -width % 16
    image = np.pad(image, ((0,0), (0, row_add), (0, col_add)))
    return predictor(image)[:height, :width]


def threshold_segmentation_mask(image, threshold=None):
    thresholded_mask = np.zeros(predictor.shape, np.uint8)
    for mask, frame in zip(thresholded_mask, predictor):
      mask[:] = threshold(frame, threshold)
    return thresholded_mask


def segment_instances(probability_maps, thresholded_maps, min_distance):
  segmentation = np.zeros(predictions.shape, np.float64)
  for probability_map, thresholded_map, segmentation_frame in zip(
     probability_maps, thresholded_maps, segmentation):
      segmentation_frame[:] = segment(
        thresholded_map, probability_map, min_distance = min_distance)
   
  masks = np.concatenate([
    frame[None] == np.arange(1, frame.max()+1)[:, None, None]
    for frame_num, frame in enumerate(segmentation.astype(np.int32))                  
  ], axis=0)
  frames = sum(([frame_num] * int(frame.max())
                for frame_num, frame in enumerate(segmentation)), [])
  y, x = zip(*(
    map(np.mean, np.where(mask))
    for mask in masks
  )
  assert len(frames) == len(masks), "If this assertion fails, that is awkward"
  return pd.DataFrame({'frame': frames, 'mask': np.arange(len(masks)), 'x': x, 'y': y}), masks
