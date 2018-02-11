import cv2
import numpy as np
import json
import time
import end2end as e2e
import plate_typediff as td
import finemapping as fm
import niblack_thresholding as nt
import finemapping_vertical as fv
import plate_detect as pd
import cache
import os


def plate_recognizechar(image,resize_h = 720):
    images = pd.detectPlateRough(image,resize_h,top_bottom_padding_rate=0.1)
    jsons = []
    for j,plate in enumerate(images):
        plate,rect,origin_plate =plate
        #res, confidence = e2e.recognizeOne(origin_plate)
        #print("RecognizePlateJson res",res)       

        #cv2.imwrite("./"+str(j)+"_rough.jpg",plate)
        # plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        plate  =cv2.resize(plate,(136,int(36*2.5)))
        t1 = time.time()

        ptype = td.platetype_predict(plate)
        print("plate_recognize ptype",ptype)
        if ptype>0 and ptype<4:
            plate = cv2.bitwise_not(plate)
        # demo = verticalEdgeDetection(plate)

        image_rgb = fm.findContoursAndDrawBoundingBox(plate)
        image_rgb = fv.finemappingVertical(image_rgb)
        cache.verticalMappingToFolder(image_rgb)
        #print("e2e:",e2e.recognizeOne(image_rgb)[0])
        image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2GRAY)

        cv2.imwrite("./"+str(j)+".jpg",image_gray)
        # image_gray = horizontalSegmentation(image_gray)

        t2 = time.time()
        print("plate_recognize transform time:",t2-t1)
        res, confidence = e2e.recognizeOne(image_rgb)
        t3 = time.time()
        print("plate_recognize transform time:",t3-t2)
        print("plate_recognize res",res," confidence:",confidence)
        res_json = {}
        if confidence  > 0.9:
            res_json["Name"] = res
            res_json["Type"] = td.plateType[ptype]
            res_json["Confidence"] = confidence;
            res_json["x"] = int(rect[0])
            res_json["y"] = int(rect[1])
            res_json["w"] = int(rect[2])
            res_json["h"] = int(rect[3])
            jsons.append(res_json)
    print(json.dumps(jsons,ensure_ascii=False))
    return json.dumps(jsons,ensure_ascii=False)
    
def carplate_recognize(imgname):
    image = cv2.imread(imgname)
    print("carplate_recognize height:",image.shape[0])
    #if image.shape[0] < 180:
    #    resize_h = 90
    #if image.shape[0] < 360:
    #    resize_h = image.shape[0]
    #elif image.shape[0] < 720:
    #    resize_h = 360
    #elif image.shape[0] > 1440:
    #    resize_h = int(image.shape[0])
    #else:
    #    resize_h = 720
    resize_h = int(image.shape[0])
    images = plate_recognizechar(image,resize_h)
    print("main images:%s\n"%(images))



if __name__ == '__main__':
    PATH_TO_TEST_IMAGES_DIR = 'images\\'
    print("__main__ start!!")
    cwd_dir = os.getcwd()
    full_dir_path = cwd_dir + "\\"+PATH_TO_TEST_IMAGES_DIR
    fileslist_in_dir = os.listdir(full_dir_path)
    files_count = len(fileslist_in_dir)
    print("main files_count:%s\n"%(files_count))
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, fileslist_in_dir[i]) for i in range(0,files_count) ]  
    for image_path in TEST_IMAGE_PATHS:
        print("main image_path:%s\n"%(image_path))
        carplate_recognize(image_path)
    
    
      