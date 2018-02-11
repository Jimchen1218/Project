
import cv2
import numpy as np
import json
import time
import end2end as e2e
import plate_typediff as td
import finemapping as fm
import niblack_thresholding as nt
import finemapping_vertical as fv
import cache



watch_cascade = cv2.CascadeClassifier('model\\cascade.xml')


def computeSafeRegion(shape,bounding_rect):
    top = bounding_rect[1] # y
    bottom  = bounding_rect[1] + bounding_rect[3] # y +  h
    left = bounding_rect[0] # x
    right =   bounding_rect[0] + bounding_rect[2] # x +  w

    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]

    # print "computeSateRegion input shape",shape
    if top < min_top:
        top = min_top
        # print "tap top 0"
    if left < min_left:
        left = min_left
        # print "tap left 0"

    if bottom > max_bottom:
        bottom = max_bottom
        #print "tap max_bottom max"
    if right > max_right:
        right = max_right
        #print "tap max_right max"

    # print "corr",left,top,right,bottom
    return [left,top,right-left,bottom-top]


def cropped_from_image(image,rect):
    x, y, w, h = computeSafeRegion(image.shape,rect)
    cv2.imwrite("cropped_from_image.jpg",image[y:h,x:w])
    return image[y:y+h,x:x+w]


def detectPlateRough(image_gray,resize_h = 720,en_scale =1.08 ,top_bottom_padding_rate = 0.05):
    print(image_gray.shape)

    if top_bottom_padding_rate>0.2:
        print("error:top_bottom_padding_rate > 0.2:",top_bottom_padding_rate)
        exit(1)

    height = image_gray.shape[0]
    print("detectPlateRough:height:",height)
    padding =    int(height*top_bottom_padding_rate)
    scale = image_gray.shape[1]/float(image_gray.shape[0])
    print("detectPlateRough:scale:",scale)
    image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
    image_color_cropped = image[padding:resize_h-padding,0:image_gray.shape[1]]
    image_gray = cv2.cvtColor(image_color_cropped,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("1.jpg",image_gray[:,:])    
    #print("detectPlateRough:image_gray:",image_gray)
    watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),maxSize=(36*10, 9*10))
    print("detectPlateRough:watches:",watches)
    cropped_images = []
    for i,(x, y, w, h) in enumerate(watches):
        cropped_origin = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
        x -= int(w * 0.14)
        w += int(w * 0.28)
        y -= int(h * 0.6)
        h += int(h * 1.2);

        cropped = cropped_from_image(image_color_cropped, (int(x), int(y), int(w), int(h)))
        
        cv2.imwrite(str(i)+".jpg",cropped[:,:])
        #image = cv2.resize(image_gray, (int(scale*resize_h), resize_h))
        #cv2.imwrite(str(i)+".jpg",image_gray[y:h,x:w])
        cropped_images.append([cropped,[x, y+padding, w, h],cropped_origin])
    return cropped_images


def detect_and_boundingbox(imgname):
		image = cv2.imread(imgname)
		print("detect_and_boundingbox height:",image.shape[0])
		print("detect_and_boundingbox width:",image.shape[1])
		if image.shape[0] < 180:
			resize_h = 90
		if image.shape[0] < 360:
			resize_h = 180
		elif image.shape[0] < 720:
			resize_h = 360
		elif image.shape[0] > 1440:
			resize_h = int(image.shape[0]/2)
		else:
			resize_h = 720        

		images = detectPlateRough(image,resize_h)


if __name__ == '__main__':
		print("__main__ start!!")
		detect_and_boundingbox('orig_3.jpg')
