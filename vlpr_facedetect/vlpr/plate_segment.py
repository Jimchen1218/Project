'''
filename:plate_segment.py
create date:1/31/2018
author:jim.chen
'''
import cv2
import os
import gc
import numpy as np

def img2gray(filename):
	img=cv2.imread(filename)
	img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cv2.imwrite("img_gray.png",img_gray)
	return img_gray
	
def img2bin(image):
	img_bin = image
	cv2.threshold(img_bin,127,255,cv2.THRESH_BINARY_INV,image)
	cv2.imwrite("img_bin.png",img_bin)
	return img_bin
	

#whitebg	,trim the horizontal border
def trim_horizontalborder_whitebg(image,threshold):
	white=[]
	white_max=0
	height = image.shape[0]
	width=image.shape[1]
	trim_start = 0
	trim_end = height
	print("trim_horizontalborder_whitebg height:",height,"width:",width)
	img_mid = int(height/2)
	print("trim_horizontalborder_whitebg img_mid:",img_mid)
	#mean = image[:,:].mean()
	#print("trim_horizontalborder_whitebg mean:",mean)
	
	total = 0
	for i in range(height):
		for j in range(width):
			if image[i][j] == 0:	
				total +=1
				
	mean = int(total/height)
	print("trim_horizontalborder_whitebg mean:",mean)
	
	for i in range(img_mid-1,0,-1):
		s = 0
		for j in range(width):
			if image[i][j] == 0:
				s+= 1
		#print("trim_horizontalborder_whitebg i:",i,"total black pixel:",s)
		if s < width*threshold:
			trim_start = i
			break

	for i in range(img_mid,height,1):
		s = 0
		for j in range(width):
			if image[i][j] == 0:
				s+= 1
		#print("trim_horizontalborder_whitebg i:",i,"total black pixel:",s)
		if s < width*threshold:
			trim_end = i
			break
			
	print("trim_horizontalborder_whitebg trim_start:",trim_start,"trim_end:",trim_end)
	img_trimhoriborder = image[trim_start:trim_end,0:width]
	cv2.imwrite("trim_horizontalborder_whitebg.png",img_trimhoriborder)
	return img_trimhoriborder
	
#blackbg	,trim the horizontal border	
def trim_horizontalborder_blackbg(image,threshold = 0.05):
	white=[]
	white_max=0
	height = image.shape[0]
	width=image.shape[1]
	trim_start = 0
	trim_end = height
	print("trim_horizontalborder_blackbg height:",height,"width:",width)
	img_mid = int(height/2)
	print("trim_horizontalborder_blackbg img_mid:",img_mid)
	
	total = 0
	for i in range(height):
		for j in range(width):
			if image[i][j] == 255:	
				total +=1	
	mean = int(total/height)
	print("trim_horizontalborder_blackbg mean:",mean)
		
	for i in range(img_mid-1,0,-1):
		s = 0
		for j in range(width):
			if image[i][j] == 255:
				s+= 1
		print("trim_horizontalborder_blackbg i:",i,"total white pixel:",s)
		if s < height*threshold:
			trim_start = i
			break

	for i in range(img_mid,height,1):
		s = 0
		for j in range(width):
			if image[i][j] == 255:
				s+= 1
		print("trim_horizontalborder_blackbg i:",i,"total white pixel:",s)
		if s < height*threshold:
			trim_end = i
			break
			
	print("trim_horizontalborder_blackbg trim_start:",trim_start,"trim_end:",trim_end)
	img_trimhoriborder = image[trim_start:trim_end,0:width]
	cv2.imwrite("trim_horizontalborder.png",img_trimhoriborder)
	return img_trimhoriborder	
	
def get_isblackbg(image):
	isblackbg = False
	white=[]
	black=[]
	white_max=0
	black_max=0
	height = image.shape[0]
	width=image.shape[1]
	print("get_isblackbg height:",height,"width:",width)
	for i in range(width):
		s = 0
		t = 0
		for j in range(height):
			if image[j][i] == 255:
				s+= 1
			if image[j][i] == 0:
				t+=1
		white_max=max(white_max,s)
		black_max=max(black_max,t)
		white.append(s)
		black.append(t)
		#print("get_isblackbg line:",i,"\ntotal pixel white:",s)
		#print("total pixel black:",t)
		
	print("get_isblackbg black_max:",black_max,"white_max:",white_max)
	if black_max > white_max:
		isblackbg=True

	return isblackbg,white_max,black_max,white,black
	
#blackbg,find each char border
def find_eachcharborder_blackbg(image,threshold):
	char=[]	
	gap=[]
	white=[]
	height = image.shape[0]
	width=image.shape[1]
	print("find_eachcharborder_blackbg height:",height,"width:",width)
	for i in range(width):
		s = 0
		for j in range(height):
			if image[j][i] == 255:
				s+= 1
		white.append(s)
		#print("find_eachcharborder_blackbg white:",white)
		
	s = 0
	b = 0
	if white[0] == 0:
		gap_first = True
	else:
		gap_first = False
	for i in range(width):
		if white[i] != 0:
			s+=1
			if b != 0:
				gap.append(b)
			b = 0
		else:
			b+=1
			if s != 0:
				char.append(s)
			s = 0
	
	print("find_eachcharborder_blackbg gap_first:",gap_first)
	print("find_eachcharborder_blackbg gap:",gap)
	print("find_eachcharborder_blackbg char:",char)	
	
	char_width = max(char)
	half_char_width = int(char_width/2)
	print("find_eachcharborder_blackbg char_width:",char_width)
	
	char_border=[]
	gap_len = len(gap)
	char_len = len(char)
	if char_len - gap_len > 0:
		gap.append(1)
	else:
		char.append(1)
		
	char_end = 0
	char_center = 0
	
	if gap_first:
		for i in range(char_len):
			char_end += gap[i]+char[i]
			print("find_eachcharborder_blackbg char_end:",char_end)
			char_center = char_end - int(char[i]/2)
			print("find_eachcharborder_blackbg char_center:",char_center)
			char_border.append([char_center-half_char_width,char_center+half_char_width])
			print("find_eachcharborder_blackbg char_border:",char_border)
	else:
		char_end += char[0]
		print("find_eachcharborder_blackbg char_end:",char_end)
		char_center = char_end - int(char[0]/2)
		print("find_eachcharborder_blackbg char_center:",char_center)
		char_border.append([char_center-half_char_width,char_center+half_char_width])
		print("find_eachcharborder_blackbg char_border:",char_border)			
		for i in range(char_len-1):
			char_end += gap[i]+char[i+1]
			print("find_eachcharborder_blackbg char_end:",char_end)
			char_center = char_end - int(char[i+1]/2)
			print("find_eachcharborder_blackbg char_center:",char_center)
			char_border.append([char_center-half_char_width,char_center+half_char_width])
			print("find_eachcharborder_blackbg char_border:",char_border)			

	del_first = False
	del_last = False	
	del_dot = True	
	print("find_eachcharborder_blackbg char_len:",char_len)
	if char_len >7:
		if char[0]<int(char[1]/2):
			del_first = True
		if char[char_len-1]<int(char[char_len-2]/2):
			del_last = True
			
	if char_len >7:
		if del_first:
			if gap[2]*2<=gap[3]:
				del_dot = False
		else:
			if gap[1]*2<=gap[2]:
				del_dot = False

	if del_last:
		char_border.pop(char_len-1)
	if del_dot and del_first:
		char_border.pop(3)
	elif del_dot:
		char_border.pop(2)
	if del_first:
		char_border.pop(0)

	print("find_eachcharborder_blackbg char_border:",char_border)
	return char_border

'''
#whitebg,find each char border
def find_eachcharborder_whitebg(image,threshold):
	char=[]	
	gap=[]
	black=[]
	height = image.shape[0]
	width=image.shape[1]
	print("find_eachcharborder_blackbg height:",height,"width:",width)
	for i in range(width):
		s = 0
		for j in range(height):
			if image[j][i] == 0:
				s+= 1
		black.append(s)
		print("find_eachcharborder_blackbg black:",black)
		
	s = 0
	b = 0
	gap_first = False
	for i in range(width):
		if black[0] == 0:
			gap_first = True
			
		if black[i] != 0:
			s+=1
			if b != 0:
				gap.append(b)
			b = 0
		else:
			b+=1
			if s != 0:
				char.append(s)
			s = 0
	
	print("find_eachcharborder gap_first:",gap_first)
	print("find_eachcharborder gap:",gap)
	print("find_eachcharborder char:",char)	
	
	char_width = max(char)
	half_char_width = int(char_width/2)
	print("find_eachcharborder char_width:",char_width)
	
	char_border=[]
	gap_len = len(gap)
	char_len = len(char)
	if char_len - gap_len > 0:
		gap.append(1)		
		
	
	char_end = 0
	char_center = 0
	
	if gap_first:
		for i in range(char_len):
			char_end += gap[i]+char[i]
			print("find_eachcharborder char_end:",char_end)
			char_center = char_end - int(char[i]/2)
			print("find_eachcharborder char_center:",char_center)
			char_border.append([char_center-half_char_width,char_center+half_char_width])
			print("find_eachcharborder char_border:",char_border)

	del_first = False
	del_last = False
	if char_len == 8 or char_len == 9:
		if char[0]<int(char[1]/2):
			del_first = True
		if char[char_len-1]<int(char[char_len-2]/2):
			del_last = True

	if del_last:
		char_border.pop(char_len-1)
	if del_first:
		char_border.pop(0)

	print("find_eachcharborder char_border:",char_border)
	return char_border
'''		
	
	
def char_segment(image):
	char_id = 0
	i=1
	start = 1
	end = start+1
	white=[]
	black=[]		
	char_border =[]
	height = image.shape[0]
	width = image.shape[1]
	print("char_segment height:",height,"width:",width)
	isblackbg,white_max,black_max,white,black = get_isblackbg(image)
	print("char_segment isblackbg:",isblackbg)
	#if isblackbg:
	img = trim_horizontalborder_blackbg(image,0.3)
	#img_trimborder = cv2.blur(img_trimborder,(2,2))
	#kernel = np.ones((3,3),np.uint8)
	#mask = np.zeros((height, width), np.uint8)
	#mask[:] = 1
	#cv2.floodFill(img,seed_pt = None, (random.randint(0,255), random.randint(0,255), random.randint(0,255)), (lo,)*3, (hi,)*3, flags)
	#img = cv2.erode(img,kernel,1)
	#img = cv2.erode(img,kernel,1)
	#img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
	#img = cv2.dilate(img,kernel,1)
	
	#cv2.imwrite("img_trimborder_trans2blur.png",img_trimborder)
	char_border = find_eachcharborder_blackbg(img,0.1)
	#else:
	#	img_trimborder = trim_horizontalborder_whitebg(image,0.25)
	#	char_border = find_eachcharborder_whitebg(img_trimborder,0.05)
	for i in range(len(char_border)):
		img_char = img[0:img.shape[0],char_border[i][0]:char_border[i][1]]
		filename = 'char_%s.png'%(i+1)
		cv2.imwrite(filename,img_char)
				
def whitebg2blackbg(image):
		image = cv2.bitwise_not(image)	
		return image
		
'''		
		height = image.shape[0]
		width = image.shape[1]
		for i in range(width):
			for j in range(height):
				if image[j][i] == 0:
					image[j][i] = 255
				else:
					image[j][i] = 0
'''	
	
	
def plate_segment():
		isblackbg = False
		white=[]
		black=[]
		white_max=0
		black_max=0
		filename_orig  = "img_rotate.jpg"
		print("plate_segment")
		img = img2gray(filename_orig)
		img = img2bin(img)
		isblackbg,white_max,black_max,white,black = get_isblackbg(img)
		print("plate_segment isblackbg:",isblackbg)
		if isblackbg == False:
			img = whitebg2blackbg(img)
			cv2.imwrite("trans2blackbg.png",img)
		kernel = np.ones((3,3),np.uint8)
		img = cv2.erode(img,kernel,1)		
		#op_open = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
		img = cv2.dilate(img,kernel,1)
		img = cv2.erode(img,kernel,1)
		char_segment(img)
		gc.collect()
	
	
def main():
	plate_segment()	
	
if __name__ == "__main__":
	main()	
	