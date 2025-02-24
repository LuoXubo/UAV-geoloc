import aircv as ac
from PIL import Image 
import cv2 as cv

imbig = cv.imread("sanjiang105_transparent_mosaic_group1.tif",2)#获取大图片
imlit =cv.imread("./images_1009/01/s00.jpg",2)#获取大图片
w , h = imbig.shape
w2, h2 = imlit.shape
piaxdata = imbig
piaxdata2 = imlit
list_frist=[]#用于存放获取到的和图像左上角第一个点一样像素的点的位置
for i in range(w):
    for j in range(h):#遍历全图
        if piaxdata[i,j] == piaxdata2[0,0]:#若相等则将点的位置放入list
            list_frist.append([i,j])
i=1
while not len(list_frist)==1:#list中有多个点的时候，需要靠相邻的点来判别，
    for x,y in list_frist:
        if x+i>=w:#防止超界报错
            list_frist.remove([x, y])
        if not piaxdata[x+i,y] ==  piaxdata2[0+i, 0]:
            list_frist.remove([x, y])#不同则移除
    i+=1
x, y = list_frist[0]
print(x,y)
#从大图中框出来
for i in range(w2):
    piaxdata[x+i,y] = (255,0,0)
for i in range(w2):
    piaxdata[x + i, y+h2] = (255, 0, 0)
for i in range(h2):
    piaxdata[x,y+i] = (255,0,0)
for i in range(h2):
    piaxdata[x + w2, y+i] = (255, 0, 0)
#imbig.show()
cv.imwrite("result.png",piaxdata)