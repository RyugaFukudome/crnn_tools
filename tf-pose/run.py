import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import glob

if __name__ == '__main__':

    image_list = glob.glob('img_list/*')
    w, h = model_wh('432x368')
    e = TfPoseEstimator(get_graph_path('cmu'), target_size=(w, h))
    
    for index,img in enumerate(image_list):
        print('img_name: 'img)
        image = common.read_imgfile(img, None, None)
        if image is None:
            sys.exit(-1)

        t = time.time()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        elapsed = time.time() - t

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        try:
            cv2.imwrite('./output_img_list/test'+str(index)+'.jpg',image)
        except Exception as e:
            print('matplitlib error',e)
