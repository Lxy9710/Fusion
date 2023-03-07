import os
import shutil
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib2 import Path
import math
# # json
# json_folder = "new_data_0307/sequence2/json_label/"
# json_folder = "new_data_0307/sequence3/json_label/"
# json_name = os.listdir(json_folder)
# png_folder= "new_data_0307/sequence3/SegmentationClass/"
# png_folder= "E:/fusion/fusionOUTM/dl3 based fusion1/new_data_0307/sequence3/SegmentationClass/"
# os.makedirs(png_folder,exist_ok=True)
jpg_folder="new_data_0307/sequence5/JPEGImages/"
os.makedirs(jpg_folder,exist_ok=True)
# 通过labelme读json

# os.system("activate labelme")
# for i in range(len(json_name)):
#     if(os.path.splitext(json_name[i])[1] == ".json"):
#         json_path = json_folder + json_name[i]
#         os.system("labelme_json_to_dataset " + json_path)

#整理标注过后的图 这里只有rgb没有转成灰度，需要的话最后有，这里数据集重命名格式是按顺序0000-99999，再大型的数据集加elif即可

i=0

# for name in json_name:
# 	if os.path.isdir(json_folder + name):
# 		if i < 10:  # 0-9
# 			newname = "000" + str(i) + '.png'
# 		elif i>9 and i<100:
# 			newname="00"+str(i)+'.png'
# 		elif i>99 and i<1000:
# 			newname="0"+str(i)+'.png'
# 		elif i>999 and i<10000:
# 			newname=str(i)+'.png'
# 		i=i+1
# 		os.chdir(json_folder + name)
# 		old_name = 'label.png'
# 		os.rename(old_name, newname) 
# 		shutil.move(json_folder + name + "/" + newname, png_folder)
# 		os.system("labelme_json_to_dataset " + json_path)
j=0
 
#整理jpg('1676963590.242275', '.jpg')

# # os.system("activate labelme")
# for j in range(len(json_name)):
# 	if os.path.splitext(json_name[j])[-1] == ".jpg":
# 		if i < 10:  # 0-9
# 				newname = "000" + str(i) +".jpg"
# 		elif i>9 and i<100:
# 				newname="00"+str(i)+".jpg"
# 		elif i>99 and i<1000:
# 				newname="0"+str(i)+".jpg"
# 		elif i>999 and i<10000:
# 				newname=str(i)+".jpg"
# 		os.rename(json_folder+json_name[j],json_folder+newname)
# 		shutil.move(json_folder  + newname, jpg_folder)
# 		i=i+1
# 	j=j+1

# i = 0
# for name in json_name:
# 	if os.path.isdir(json_folder + name):
# 		if i < 10:  # 0-9
# 			newname = "000" + str(i) + '.png'
# 		elif i>9 and i<100:
# 			newname="00"+str(i)+'.png'
# 		elif i>99 and i<1000:
# 			newname="0"+str(i)+'.png'
# 		elif i>999 and i<10000:
# 			newname=str(i)+'.png'
# 		i += 1		
# 		os.chdir(json_folder + name)
# 		old_name = 'label.png'
# 		os.rename(old_name, newname)
# 		shutil.move(json_folder + name + "/" + newname, png_folder)

# for i in range(len(json_name)):
#     if(os.path.splitext(json_name[i])[1] == ".jpg"):
# 	    if i < 10:  # 0-9
# 		    newname = "000" + str(i) + '.png'
#         elif i>9 and i<100:
# 			newname="00"+str(i)+'.png'
# 		elif i>99 and i<1000:
# 			newname="0"+str(i)+'.png'
# 		elif i>999 and i<10000:
# 			newname=str(i)+'.png'
# 		i += 1		
# 		os.chdir(json_folder + name)
# 		old_name = 'label.png'
# 		os.rename(old_name, newname)
# 		shutil.move(json_folder + name + "/" + newname, png_folder)
# 	    shutil.move(json_folder + json_name[i], "E:\localpy\segmentation\JPEGImages")
        # json_path = json_folder + json_name[i]
        # os.system("labelme_json_to_dataset " + json_path)
# 改像素值
# png_name = os.listdir(png_folder)
# for name in png_name:
# 	data_source = cv2.imread(png_folder +'/'+ name)
# 	data = np.array(data_source)
# 	img_path = png_folder + '/' + name
# 	for i in range(data[:, :, 0].shape[0]):
# 		for j in range(data[:, :, 0].shape[1]):
# 			if data[:, :, 2][i][j]  > 0 :
# 				data[:, :, 2][i][j] = 255 #Red
# 				data[:, :, 1][i][j] = 255 #Green
# 				data[:, :, 0][i][j] = 255 #Blue
# 	cv2.imwrite(img_path , data)
# 	png = Image.open(img_path).convert('L').save(img_path)

#原始数据转换  
# inputdir = r'F:\ProcessData\AllData'    #存放初始数据的文件夹
# outputdir = r'E:\localpy\segmentation\JPEGImages'
# c = 1
# for dir in os.listdir(inputdir):
#     # 设置旧文件名（就是路径+文件名）
#     oldname = inputdir + os.sep + dir   # os.sep添加系统分隔符

#     # 设置新文件名
#     #c = outputdir + os.sep + dir.split('_')[1]

#     a = "0" * (6 - len(str(c)))
#     newname =outputdir + os.sep +a + str(c) + '.jpg'
#     shutil.copyfile(oldname, newname)  # 用os模块中的rename方法对文件改名
#     print(oldname, '======>', newname)
# #     c += 1

# 这里划分数据集 按顺序划的
outdir = r'new_data_0307/sequence5/ImageSets/Segmentation/'
os.makedirs(outdir,exist_ok=True)
img_name = os.listdir(jpg_folder)
images = []
train=''
val=''
test=''
for file in os.listdir(jpg_folder):
    p = Path(file) 
    filename = p.stem
    images.append(filename)
# 训练集测试集验证集比例为：4：2：2 随机划分可以直接使用 但时序的话不好直接分 所以我按比例分
#  train, test = train_test_split(images, train_size=0.5, random_state=0)
# val, test = train_test_split(test, train_size=0.5, random_state=0)
train=images[0: math.floor(len(images)/2)]
val=images[math.floor(len(images)/2)+1:math.floor(len(images)*0.75)]
test=images[math.floor(len(images)*0.75):len(images)-1]    

with open(outdir + os.sep +"train.txt", 'w') as f:
    f.write('\n'.join(train))
with open(outdir + os.sep +"val.txt", 'w') as f:
    f.write('\n'.join(val))
with open(outdir + os.sep +"test.txt", 'w') as f:
    f.write('\n'.join(test))
