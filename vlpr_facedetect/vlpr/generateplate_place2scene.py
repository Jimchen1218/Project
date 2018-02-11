#coding=utf-8
'''
生成车牌数据，将车牌放到自然图像中
'''
import os
# import sys
import numpy as np
import cv2
import argparse
# import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from math import *

from PlateCommon import *

TEMPLATE_IMAGE = "./images/template.bmp"
# 这里假设车牌最小的检测尺寸是65*21，检测车牌的最小图像为65*21，车牌宽高比变化范围是(1.5, 4.0)
PLATE_SIZE_MIN = (65, 21)

class GenPlatePlacetoScene:
    '''车牌数据生成器，车牌放在自然场景中，位置信息存储在同名的txt文件中
    '''
    def __init__(self, fontCh, fontEng, NoPlates):
        self.fontC = ImageFont.truetype(fontCh, 43, 0)    # 省简称使用字体
        self.fontE = ImageFont.truetype(fontEng, 60, 0)   # 字母数字字体
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread(TEMPLATE_IMAGE), (226, 70))
        self.noplates_path = []
        for parent, _, filenames in os.walk(NoPlates):
            print("__init__ parent:\n",parent)
            for filename in filenames:
                self.noplates_path.append(parent + "/" + filename)

    def gen_plate_string(self):
        '''生成车牌号码字符串'''
        plate_str = ""
        for cpos in range(7):
            if cpos == 0:
                plate_str += chars[r(31)]
            elif cpos == 1:
                plate_str += chars[41+r(24)]
            else:
                plate_str += chars[31 + r(34)]
            print("gen_plate_string plate_str:\n",plate_str)
        return plate_str

    def drawimage(self, val):
        offset= 2
        print("drawimage val:\n",val)
        charfirst= val.decode("utf-8")[0]
        for i in range(len(val)-1):
           print("drawimage chr(val):\n",chr(val[i+1]))
        self.img[0:70, offset+8:offset+8+23] = GenCh(self.fontC, charfirst)
        self.img[0:70, offset+8+23+6:offset+8+23+6+23] = GenCh1(self.fontE, val[3])
        for i in range(5):
            base = offset+8+23+6+23+17+i*23+i*6
            self.img[0:70, base:base+23]= GenCh1(self.fontE, val[i+4])
        return self.img

    def generate(self,text):
        print(text, len(text))
        #fg = self.draw(text.decode(encoding="utf-8"))   # 得到白底黑字
        fg = self.drawimage(text.encode("utf-8"))
        # cv2.imwrite('01.jpg', fg)
        fg = cv2.bitwise_not(fg)    # 得到黑底白字
        # cv2.imwrite('02.jpg', fg)
        com = cv2.bitwise_or(fg, self.bg)   # 字放到（蓝色）车牌背景中
        # cv2.imwrite('03.jpg', com)
        com = rot(com, r(60)-30, com.shape, 30) # 矩形-->平行四边形
        # cv2.imwrite('04.jpg', com)
        com = rotRandrom(com, 10, (com.shape[1], com.shape[0])) # 旋转
        # cv2.imwrite('05.jpg', com)
        com = tfactor(com) # 调灰度
        # cv2.imwrite('06.jpg', com)
        com, loc = random_scene(com, self.noplates_path)    # 放入背景中
        if com is None or loc is None:
            print("generate loc:",loc," com:",com,"\n")
            return None, None
        # cv2.imwrite('07.jpg', com)
        com = AddGauss(com, 1+r(4)) # 加高斯平滑
        # cv2.imwrite('08.jpg', com)
        com = addNoise(com)         # 加噪声
        # cv2.imwrite('09.jpg', com)
        return com, loc

    def gen_batch(self, batchSize, outputPath):
        '''批量生成图片'''
        if (not os.path.exists(outputPath)):
            os.mkdir(outputPath)
        for i in range(batchSize):
            plate_str = self.gen_plate_string()
            img, loc =  self.generate(plate_str)
            if img is None:
                continue
            cv2.imwrite(outputPath + "/" + str(i).zfill(2) + ".jpg", img)
            with open(outputPath + "/" + str(i).zfill(2) + ".txt", 'w') as obj:
                line = ','.join([str(v) for v in loc]) + ',"' + plate_str + '"\n' 
                obj.write(line)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--bg_dir', default='./backgrounds/', help='background_image directory')
    args.add_argument('--out_dir', default='./plate_test/', help='output directory')
    args.add_argument('--train_num', default=100, type=int, help='train number')
    return args.parse_args()

def main(args):
    gpps = GenPlatePlacetoScene("./font/platech.ttf", './font/platechar.ttf', args.bg_dir)
    gpps.gen_batch(args.train_num, args.out_dir)


if __name__ == '__main__':
    main(parse_args())




