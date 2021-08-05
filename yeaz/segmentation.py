# -*- coding: utf-8 -*-
import os
import sys
from .unet import unet, Model
import numpy as np
import skimage
from skimage import io
import skimage.transform as trans


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
      mask[:] = nn.threshold(frame, threshold)
    return thresholded_mask


def segment_instances(probability_maps, thresholded_maps, min_distance):
  segmentation = np.zeros(predictions.shape, np.float64)
  for probability_map, thresholded_map, segmentation_frame in zip(
     probability_maps, thresholded_maps, segmentation):
      segmentation_frame[:] = segment.segment(
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
