#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#coding=utf-8
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2;
import numpy as np;
import os;
from math import *
import argparse

index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10, "浙": 11, "皖": 12,
         "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20, "琼": 21, "川": 22, "贵": 23, "云": 24,
         "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30, "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36,
         "6": 37, "7": 38, "8": 39, "9": 40, "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48,
         "J": 49, "K": 50, "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
         "W": 61, "X": 62, "Y": 63, "Z": 64};

chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
             "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
             "Y", "Z"
             ];

def AddSmudginess(img, Smu):
    rows = r(Smu.shape[0] - 50)
    cols = r(Smu.shape[1] - 50)
    adder = Smu[rows:rows + 50, cols:cols + 50];
    adder = cv2.resize(adder, (50, 50));
    #   adder = cv2.bitwise_not(adder)
    img = cv2.resize(img,(50,50))
    img = cv2.bitwise_not(img)
    img = cv2.bitwise_and(adder, img)
    img = cv2.bitwise_not(img)
    return img

def rot(img,angel,shape,max_angel):
    size_o = [shape[1],shape[0]]
    size = (shape[1]+ int(shape[0]*cos((float(max_angel )/180) * 3.14)),shape[0])
    interval = abs( int( sin((float(angel) /180) * 3.14)* shape[0]));
    pts1 = np.float32([[0,0]         ,[0,size_o[1]],[size_o[0],0],[size_o[0],size_o[1]]])
    if(angel>0):
        pts2 = np.float32([[interval,0],[0,size[1]  ],[size[0],0  ],[size[0]-interval,size_o[1]]])
    else:
        pts2 = np.float32([[0,0],[interval,size[1]  ],[size[0]-interval,0  ],[size[0],size_o[1]]])
    M  = cv2.getPerspectiveTransform(pts1,pts2);
    dst = cv2.warpPerspective(img,M,size);
    return dst;

def rotRandrom(img, factor, size):
    """ 使图像轻微的畸变
        img 输入图像
        factor 畸变的参数
        size 为图片的目标尺寸
    """    
    shape = size;
    pts1 = np.float32([[0, 0], [0, shape[0]], [shape[1], 0], [shape[1], shape[0]]])
    pts2 = np.float32([[r(factor), r(factor)], [ r(factor), shape[0] - r(factor)], [shape[1] - r(factor),  r(factor)],
                       [shape[1] - r(factor), shape[0] - r(factor)]])
    M = cv2.getPerspectiveTransform(pts1, pts2);
    dst = cv2.warpPerspective(img, M, size);
    return dst;



def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);

    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    hsv[:,:,2] = hsv[:,:,2]*(0.2+ np.random.random()*0.8);

    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
    return img

def random_envirment(img,data_set):
    index=r(len(data_set))
    env = cv2.imread(data_set[index])
    env = cv2.resize(env,(img.shape[1],img.shape[0]))
    bak = (img==0);
    bak = bak.astype(np.uint8)*255;
    inv = cv2.bitwise_and(bak,env)
    img = cv2.bitwise_or(inv,img)
    return img

def GenCh(f,val):
    img=Image.new("RGB", (45,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    #print("GenCh val:\n",val)
    draw.text([0, 1],str(val),(0,0,0),font=f)
    img =  img.resize((23,70))
    A = np.array(img)
    return A
    
def GenCh1(f,val):
    img=Image.new("RGB", (23,70),(255,255,255))
    draw = ImageDraw.Draw(img)
    #print("GenCh1 val:\n",val)
    print("GenCh1 chr(val):\n",chr(val))
    draw.text((0, 1),chr(val),(0,0,0),font=f)
    A = np.array(img)
    return A
    
def AddGauss(img, level):
    return cv2.blur(img, (level * 2 + 1, level * 2 + 1));

def r(val):
    return int(np.random.random() * val)

def AddNoiseSingleChannel(single):
    diff = 255-single.max();
    noise = np.random.normal(0,1+r(6),single.shape);
    noise = (noise - noise.min())/(noise.max()-noise.min())
    noise= diff*noise;
    noise= noise.astype(np.uint8)
    dst = single + noise
    return dst

def addNoise(img,sdev = 0.5,avg=10):
    img[:,:,0] =  AddNoiseSingleChannel(img[:,:,0]);
    img[:,:,1] =  AddNoiseSingleChannel(img[:,:,1]);
    img[:,:,2] =  AddNoiseSingleChannel(img[:,:,2]);
    return img;

class generateplate:
    def __init__(self,fontCh,fontEng,NoPlates):
        self.fontC =  ImageFont.truetype(fontCh,43,0);
        self.fontE =  ImageFont.truetype(fontEng,60,0);
        self.img=np.array(Image.new("RGB", (226,70),(255,255,255)))
        self.bg  = cv2.resize(cv2.imread("./images/template.bmp"),(226,70));
        self.smu = cv2.imread("./images/smu2.jpg");
        self.noplates_path = [];
        for parent,parent_folder,filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent+"/"+filename;
                self.noplates_path.append(path);

    def draw(self,val):
        offset= 2;
        #print("draw val:\n",val)
        #print("draw val[0]:\n",val[0])
        firstval= val.decode("utf-8")[0]
        #print("draw firstval:\n",firstval)
        #print("draw charfirst:\n",charfirst)
        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,firstval);
        #print("draw self.img:\n",self.img)	
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[3]);
        for i in range(5):
            base = offset+8+23+6+23+17 +i*23 + i*6 ;
            self.img[0:70, base  : base+23]= GenCh1(self.fontE,val[i+4]);
        return self.img
        
    def generate(self,text):
        if len(text) == 7:
            print("generate text:\n",text)
            txt_encode = text.encode("utf-8")
            print("generate txt_encode:\n",txt_encode)
            fg = self.draw(txt_encode)
            fg = cv2.bitwise_not(fg);
            com = cv2.bitwise_or(fg,self.bg);
            com = rot(com,r(60)-30,com.shape,30);
            com = rotRandrom(com,10,(com.shape[1],com.shape[0]));
            #com = AddSmudginess(com,self.smu)
            com = tfactor(com)
            com = random_envirment(com,self.noplates_path);
            com = AddGauss(com, 1+r(4));
            com = addNoise(com);
            return com
        
    def genPlateString(self,pos,val):
        plateStr = "";
        box = [0,0,0,0,0,0,0];
        if(pos!=-1):
            box[pos]=1;  
        for unit,cpos in zip(box,range(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if cpos == 0:
                    bytefirst = chars[r(31)]
                    plateStr += bytefirst
                elif cpos == 1:
                    plateStr += chars[41+r(24)]
                else:
                    plateStr += chars[31 + r(34)] 
        return plateStr;

    def genBatch(self, batchSize,pos,charRange, outputPath,size):
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)
        for i in range(batchSize):
            plateStr = self.genPlateString(-1,-1)
            print("genBatch\n i:",i," plateStr:",plateStr,"\n")
            img =  self.generate(plateStr);
            #print("genBatch img:\n",img)
            img = cv2.resize(img,size);
            filename = os.path.join(outputPath, str(i).zfill(4) + '.' + plateStr+ ".jpg")
            cv2.imwrite(filename, img);


def parse_args():
    args = argparse.ArgumentParser()
    #parser.add_argument('--font_ch', default='./font/platech.ttf')
    #parser.add_argument('--font_en', default='./font/platechar.ttf')
    #parser.add_argument('--bg_dir', default='./NoPlates')
    args.add_argument('--out_dir', default='./plate_train', help='output directory')
    args.add_argument('--make_num', default=100, type=int, help='sample numbers')
    #parser.add_argument('--img_w', default=120, type=int, help='width')
    #parser.add_argument('--img_h', default=32, type=int, help='height')
    return args.parse_args()

def main(args):
    gp = generateplate("./font/platech.ttf", './font/platechar.ttf', "./NoPlates")
    gp.genBatch(args.make_num,2,range(31,65), args.out_dir,(272,72))


def find_edge(image):
    print("find_edge image.shape[0]:",image.shape[0])
    print("find_edge image.shape[1]:",image.shape[1])
    sum_i = image.sum(axis=0)
    print("find_edge sum_i:",sum_i)
    print("find_edge image.shape[0]:",image.shape[0])
    sum_i =  sum_i.astype(np.float)
    sum_i/=image.shape[0]*255
	# print sum_i
		
    start= 0
    end = image.shape[1]-1
    print("find_edge start:",start,"end:",end)
    for i,one in enumerate(sum_i):
        print("find_edge one:",one)
        if one>0.4:
            start = i;
            if start-3<0:
                start = 0
            else:
                start -=3

            break;
    for i,one in enumerate(sum_i[::-1]):
        if one>0.4:
            end = end - i;
            if end+4>image.shape[1]-1:
                end = image.shape[1]-1
            else:
                end+=4
            break
    return start,end


def verticalEdgeDetection(image):
    image_sobel = cv2.Sobel(image.copy(),cv2.CV_8U,1,0)
    # image = auto_canny(image_sobel)

    # img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT
    # canny_image  = auto_canny(image)
    flag,thres = cv2.threshold(image_sobel,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    print(flag)
    flag,thres = cv2.threshold(image_sobel,int(flag*0.7),255,cv2.THRESH_BINARY)
    # thres = simpleThres(image_sobel)
    print(flag)
    kernel = np.ones(shape=(3,15))
    print(kernel)
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    thres = cv2.morphologyEx(thres,cv2.MORPH_CLOSE,kernel)
    print(thres)
    return thres

def horizontalSegmentation(image):
    thres = verticalEdgeDetection(image)
    # thres = thres*image
    head,tail = find_edge(thres)
    # print head,tail
    # cv2.imshow("edge",thres)
    tail = tail+5
    if tail>135:
        tail = 135
    image = image[0:35,head:tail]
    image = cv2.resize(image, (int(136), int(36)))
    cv2.imwrite("image_resize.jpg",image)
    #cv2.imshow("image_resize",image)
    #cv2.waitKey(0)  
    #cv2.destroyAllWindows() 
    return image

if __name__ == '__main__':
		#main(parse_args())
    fn = '14.jpg'
    img = cv2.imread(fn)
    image_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    end = horizontalSegmentation(image_gray)
    print("__main__ generateplate success!!!\n")



