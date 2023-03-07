import os
import shutil
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib2 import Path
 
#整理jpg('1676963590.242275', '.jpg')
json_folder = 'E:\localpy\segmentation\json_label/'
json_name = os.listdir(json_folder)
# # # os.system("activate labelme")
# for i in range(len(json_name)):
# 	# if os.path.isdir(os.path.splitext(json_name[i])[-1] == '.jpg'):
# 	if i < 10:  # 0-9
# 			newname = "000" + str(i+1) + '.jpg'
# 	elif i>9 and i<100:
# 			newname="00"+str(i+1)+'.jpg'
# 	elif i>99 and i<1000:
# 			newname="0"+str(i+1)+'.jpg'
# 	elif i>999 and i<10000:
# 			newname=str(i+1)+'.jpg'
# 	os.rename(json_folder+json_name[i], json_folder+newname)
# 	i=i+1
	


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
# #         # json_path = json_folder + json_name[i]
#         # os.system("labelme_json_to_dataset " + json_path)
 
# #json到png的批量命名/转移
# png_folder= "E:\localpy\segmentation\png/"
# os.makedirs(png_folder,exist_ok=True)
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
		# shutil.move("segmentation\json_label\1676963588_5975072_json\001.png", "segmentation\png")
 
# #改像素值
# # png_name = os.listdir(png_folder)
# # for name in png_name:
# # 	data_source = cv2.imread(png_folder +'/'+ name)
# # 	data = np.array(data_source)
# # 	img_path = png_folder + '/' + name
# # 	for i in range(data[:, :, 0].shape[0]):
# # 		for j in range(data[:, :, 0].shape[1]):
# # 			if data[:, :, 2][i][j]  > 0 :
# # 				data[:, :, 2][i][j] = 255 #Red
# # 				data[:, :, 1][i][j] = 255 #Green
# # 				data[:, :, 0][i][j] = 255 #Blue
# # 	cv2.imwrite(img_path , data)
# # 	png = Image.open(img_path).convert('L').save(img_path)

# #原始数据转换  
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
#     c += 1

imagedir = 'E:\localpy\segmentation\JPEGImages'
outdir = r'E:\localpy\segmentation\ImageSets'

images = []
for file in os.listdir(imagedir):
    p = Path(file) 
    filename = p.stem
    images.append(filename)
# 训练集测试集验证集比例为：4：2：2
train, test = train_test_split(images, train_size=0.5, random_state=0)
val, test = train_test_split(test, train_size=0.5, random_state=0)

with open(outdir + os.sep +"train.txt", 'w') as f:
    f.write('\n'.join(train))
with open(outdir + os.sep +"val.txt", 'w') as f:
    f.write('\n'.join(val))
with open(outdir + os.sep +"test.txt", 'w') as f:
    f.write('\n'.join(test))
