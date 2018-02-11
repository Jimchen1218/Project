#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#coding=utf-8
'''
name:test_facedetection.py
date:2/7/2018
author:jimchen1218@sina.com
'''

import numpy as np
import os
import sys
import tensorflow as tf
import cv2

from matplotlib import pyplot as plt
from PIL import Image


import gc

import label_map_util
import visualization_utils as vis_util

print(__doc__)

#if tf.__version__ != '1.4.0':
#  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')
  
#%matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")




FOLDER_MODEL_NAME = 'rfcn_resnet101'
PATH_TO_CKPT = FOLDER_MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'pascalface_label_map.pbtxt')
NUM_CLASSES = 1
  
#open a image 
def img_open(img_file):
		ret_img = cv2.imread(img_file)
		return ret_img
		
def img_save(img_file,filename):
	  cv2.imwrite(filename,img_file)

def img_get_height_width(image):
		ret_h,ret_w = image.shape[:2]
		print("img_get_width_height width:%d,height:%d\n"%(ret_w,ret_h))
		return ret_h,ret_w

def mask_build(image):
		ret_mask = np.zeros(image.shape[:2], np.uint8)
		return ret_mask

def img_bgr2gray(image):
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		return gray

def img_erode_dilate(image):
		erode=cv2.erode(image,None,iterations=1)
		dilate=cv2.dilate(erode,None,iterations=2)
		#print("dilate:%s\n"%(dilate))		
		return dilate

def img_addnoise(image):
		width,height = image.shape[:2]
		for i in range(2000):
				noise_x = np.random.randint(0,width)
				noise_y = np.random.randint(0,height)
				image[noise_x][noise_y] = 255
		return image
		
def img_blur(image):
		image = cv2.blur(image,(3,5))	
		#print("img_blur image:%s\n"%(image))
		return image

def img_gaussianblur(image):
		image=cv2.GaussianBlur(image, (3, 3), 0)
		#print("img_blur image:%s\n"%(image))
		return image		
		
def img_medianblur(image):
		image = np.hstack([cv2.medianBlur(image,3),
		                     cv2.medianBlur(image,5),
		                     cv2.medianBlur(image,7)
		                     ])	
		#image = cv2.medianBlur(image,5)
		#print("img_medianblur image:%s\n"%(image))
		return image		
		
def img_filter2D(image):
		kernel = np.ones((10,10),np.float32)/25
		image = cv2.filter2D(image,-1,kernel)
		return image	
		
def img_resize_interpolation_cubic(image):
		width,height = img_get_width_height(image)
		blur_cubic = cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC)
		return dilate	
		
def img_resize_interpolation_bilinear(image):
		width,height = img_get_width_height(image)
		blur_cubic = cv2.resize(image,(width,height),interpolation=cv2.INTER_LINEAR)
		return dilate

def img_sub_mean(image):
		#print("image:%s\n"%(image))
		img = np.array(img)
		mean = np.mean(img)
		img = img - mean
		img = img * 0.9 + mean * 0.9
		img /= 255
		return img
		
def img_inverse(image):		
		image = np.array(image)
		image = 255 -image
		return image

def img_contour(image):
		gray = img_bgr2gray(image)
		cv2.imshow("gray", gray)
		cv2.waitKey(0)
		inverse = img_inverse(gray)
		cv2.imshow("inverse", inverse)
		cv2.waitKey(0)
		ret, binary = cv2.threshold(inverse,127,255,cv2.THRESH_BINARY)
		#print("img_contour ret:%s\n"%(ret))
		img, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		print("img_contour contours:%s,hierarchy:%s\n"%(contours,hierarchy))
		ret = cv2.drawContours(inverse,contours,-1,(0,0,255),3)
		cnt = len(contours)
		print("cnt:%s\n"%(cnt))
		print("contours[0][0]:%s\n"%(contours[0][0]))

		x, y, w, h = cv2.boundingRect(cnt)
		print("x:%s,y:%s,w:%s,h:%s\n"%(x,y, w, h))
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
		
		cv2.imshow("img", img)
		cv2.waitKey(0)
		#return (x, y, w, h)


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
  
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  print("boxes im_width:%d,im_height:%d"%(im_width,im_height))
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def generate_imgs(image,image_name,image_np,boxes, scores,max_boxes_to_draw = 4,min_score_thresh = 0.95):
    image_w,image_h = 0,0
    cood_x,cood_y,cood_dx,cood_dy = 0,0,0,0
    print("generate_imgs boxes.shape[0]:%s\n"%(boxes.shape[0]))
    #if not max_boxes_to_draw:
     #   max_boxes_to_draw = boxes.shape[0]
    for i in range(max_boxes_to_draw):
        print("generate_imgs scores[%d]:%s\n"%(i,scores[0][i]))
        if scores[0][i] > min_score_thresh:
            print("generate_imgs boxes[%d]:%s"%(i,boxes[0][i]))
            y,x,h,w = boxes[0][i][0],boxes[0][i][1],boxes[0][i][2],boxes[0][i][3]
            #print("generate_imgs x:%s,y:%s,w:%s,h:%s"%(x,y,w,h))
            image_w,image_h = image_np.shape[1],image_np.shape[0]
            print("generate_imgs image_w:%d,image_h:%d"%(image_w,image_h))
            cood_x = int(x*image_w)
            cood_y = int(y*image_h*0.95)
            cood_dx = int(w*image_w)
            cood_dy = int(h*image_h*1.05)
            print("generate_imgs cood_x:%s,cood_y:%s,cood_dx:%s,cood_dy:%s"%(cood_x,cood_y,cood_dx,cood_dy))
            #image_face_cropped = image[cood_x:cood_dx,cood_y:cood_dy]
            #image_gray = cv2.cvtColor(image_face_cropped,cv2.COLOR_RGB2GRAY)
            rect = (cood_x,cood_y,cood_dx,cood_dy)
            rect_img = image.crop(rect)
            face_name = image_name+"_"+str(i+1)+".jpg"
            print("generate_imgs face_name:",face_name)
            rect_img.save(face_name)
            #cv2.imwrite("%d.jpg"%i,image_gray[:,:])
            #save_img(boxes[0])

def main():
    PATH_TO_TEST_IMAGES_DIR = 'facedetection\\'
    cwd_dir = os.getcwd()
    full_dir_path = cwd_dir + "\\"+PATH_TO_TEST_IMAGES_DIR
    fileslist_in_dir = os.listdir(full_dir_path)
    files_count = len(fileslist_in_dir)
    print("main files_count:%s\n"%(files_count))
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(0,files_count) ]
    #IMAGE_SIZE = (8, 8)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            #print("num_detections:%d"%(sess.run(num_detections)))
            for image_path in TEST_IMAGE_PATHS:
                print("image_path:%s"%(image_path))
                image_name = image_path.split("\\")[1].split(".")[0]
                print("image_name:%s"%(image_name))
                image = Image.open(image_path)
                
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})

                print("boxes:%s\n"%(boxes))
                print("boxes[0]:%s\n"%(boxes[0]))
                print("boxes[0][0]:%s\n"%(boxes[0][0])) 
                print("boxes[0][0][0]:%s\n"%(boxes[0][0][0]))
                
                print("scores:%s\n"%(scores))
                print("image_scores[0]:%s\n"%(scores[0]))
                print("image_scores[0][0]:%s\n"%(scores[0][0]))
                
                generate_imgs(image,image_name,image_np,boxes,scores)
                print("run success!!!")


if __name__ == "__main__":
		main()
		gc.collect() 
 
  