#coding=utf-8
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import math

from scipy.ndimage import filters

from skimage import measure,draw

from PIL import Image
import matplotlib.cm as cm
import scipy.signal as signal


def conv2d():
		# Gauss 
		def func(x,y,sigma=1):
		    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

		# 生成标准差为5的5*5高斯算子
		suanzi1 = np.fromfunction(func,(5,5),sigma=5)

		# Laplace扩展算子
		suanzi2 = np.array([[1, 1, 1],
		                    [1,-8, 1],
		                    [1, 1, 1]])

		# 打开图像并转化成灰度图像
		image = Image.open("111.jpg").convert("L")
		image_array = np.array(image)
		print("conv2d image_array:",image_array)
		
		# 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
		image_blur = signal.convolve2d(image_array, suanzi1, mode="same")

		# 对平滑后的图像进行边缘检测
		image2 = signal.convolve2d(image_blur, suanzi2, mode="same")

		# 结果转化到0-255
		image2 = (image2/float(image2.max()))*255

		# 将大于灰度平均值的灰度值变成255（白色），便于观察边缘
		image2[image2>image2.mean()] = 255

		# 显示图像
		plt.subplot(2,1,1)
		plt.imshow(image_array,cmap=cm.gray)
		plt.axis("off")
		plt.subplot(2,1,2)
		plt.imshow(image2,cmap=cm.gray)
		plt.axis("off")
		plt.show()



def angle(x,y):
    return int(math.atan2(float(y),float(x))*180.0/3.1415)


def h_rot(src, angle, scale=1.0):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    pass


def v_rot(img, angel, shape, max_angel):
    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*np.cos((float(max_angel )/180) * 3.14)),shape[0])
    interval = abs( int( np.sin((float(angel) /180) * 3.14)* shape[0]))
    pts1 = np.float32([[0,0],[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])

    M  = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,size)
    return dst,M


def skew_detection(image_gray):
    h, w = image_gray.shape[:2]
    eigen = cv2.cornerEigenValsAndVecs(image_gray,12, 5)
    angle_sur = np.zeros(180,np.uint)
    eigen = eigen.reshape(h, w, 3, 2)
    flow = eigen[:,:,2]
    vis = image_gray.copy()
    vis[:] = (192 + np.uint32(vis)) / 2
    d = 12
    points =  np.dstack( np.mgrid[d/2:w:d, d/2:h:d] ).reshape(-1, 2)
    for x, y in points:
        vx, vy = np.int32(flow[int(y), int(x)]*d)
        # cv2.line(rgb, (x-vx, y-vy), (x+vx, y+vy), (0, 355, 0), 1, cv2.LINE_AA)
        ang = angle(vx,vy)
        angle_sur[(ang+180)%180] +=1

    # torr_bin = 30
    angle_sur = angle_sur.astype(np.float)
    angle_sur = (angle_sur-angle_sur.min())/(angle_sur.max()-angle_sur.min())
    angle_sur = filters.gaussian_filter1d(angle_sur,5)
    skew_v_val =  angle_sur[20:180-20].max()
    skew_v = angle_sur[30:180-30].argmax() + 30
    skew_h_A = angle_sur[0:30].max()
    skew_h_B = angle_sur[150:180].max()
    skew_h = 0
    if (skew_h_A > skew_v_val*0.3 or skew_h_B > skew_v_val*0.3):
        if skew_h_A>=skew_h_B:
            skew_h = angle_sur[0:20].argmax()
        else:
            skew_h = - angle_sur[160:180].argmax()
    return skew_h,skew_v

def findcontours(image):
		image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		contours = measure.find_contours(image_gray,0.5)
		print("findcontours contours:",contours)
		#cv2.imwrite("contours.png",contours)


def fastDeskew(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    skew_h,skew_v = skew_detection(image_gray)
    print("校正角度 h ",skew_h,"v",skew_v)
    deskew,M = v_rot(image,int((90-skew_v)*1.5),image.shape,60)
    return deskew,M


def detectangle_houghline(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #height = img.shape[0]
    width = img.shape[1]
    edges = cv2.Canny(img,100,200)
    #edges = cv2.Sobel(img,cv2.CV_16S,1,0)
    #edges = cv2.convertScaleAbs(edges)
    cv2.imwrite("edges.png",edges)
    minThres = int(width/2)
    lines = cv2.HoughLines(edges,1,np.pi/180,minThres)
    # p1:img,p2:output lines,p3:rho,p4:theta,p5:threshold,p6:minLineLength,p7:minLineGap		
    #lines = cv2.HoughLinesP(edges,1,np.pi/180,minThres,minLineLength=60,maxLineGap=10)
    print("detecthoughline lines:",lines)
    if lines is None:
        return None
    
    print("detecthoughline lines[0][0][1]:",lines[0][0][1])
    theta = lines[0][0][1]
    lean_angle=int(theta*180/np.pi-90)
    print("detectangle_houghline lean_angle:",lean_angle)
    return lean_angle
		
def drawleanline(img,r,theta):
    width = img.shape[1]
    a= np.cos(theta)
    b=np.sin(theta)
    x0=a*r
    y0=b*r
    x1=int(x0+width*(-b))
    y1=int(y0+width*(a))
    x2=int(x0-width*(-b))
    y2=int(y0-width*(a))
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
    cv2.imwrite("img_addline.png",img)	

def rotateimageangle_saveimg(image,angle):
    if angle != 0:
        cv2.imwrite("img_rotate.jpg",image)
        return True
    height = image.shape[0]
    width = image.shape[1]
    r_x,r_y = width/2,height/2
    print("rotateimageangle_saveimg r_x:",r_x," r_y:",r_y)
    M = cv2.getRotationMatrix2D((r_x,r_y),angle,1)
    img_rotate = cv2.warpAffine(img,M,(width,height))
    cv2.imwrite("img_rotate.jpg",img_rotate)
    return True
		
def rotateimageangle(image,angle):
    if angle != 0:
        return image
    height = image.shape[0]
    width = image.shape[1]
    r_x,r_y = width/2,height/2
    print("rotateimageangle r_x:",r_x," r_y:",r_y)
    M = cv2.getRotationMatrix2D((r_x,r_y),angle,1)
    img_rotate = cv2.warpAffine(image,M,(width,height))
    return img_rotate
	
def deskew_and_rotate():
    imgname = '0.jpg'
    img = cv2.imread(imgname)
    angle = detectangle_houghline(img)
    rotateimageangle(img,angle)

if __name__ == '__main__':
    print("__main__ start!!")
    #skew_h,skew_v = skew_detection(gray)
    #deskew,M = v_rot(img,int((90-skew_v)*1.5),img.shape,60)
    #cv2.imwrite("deskew.png",deskew)
    #findcontours(img)
    # img = h_rot(img,skew_h)
    # if img.shape[0]>img.shape[1]:
    #     img = h_rot(img, -90)
    deskew_and_rotate()
		
		