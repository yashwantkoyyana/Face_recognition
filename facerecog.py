import numpy as np
from numpy import linalg as LA
import cv2
from PIL import Image
import os
import glob
from scipy.spatial import distance


path, dirs, files = next(os.walk("/home/iiitk/Desktop/yash/images"))
file_count = len(files)
#print(file_count)
#cv_img = []

orig_array=np.zeros(shape=(file_count,10000))

k=0
for img in glob.glob("/home/iiitk/Desktop/yash/images/*.jpeg"):
    l = 0
    image = cv2.imread(img)
    image1 = cv2.resize(image, (100 ,100))
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("image", gray)

    for i in range(100):
        for j in range(100):
            orig_array[k][l]=gray[i,j]
            l=l+1

    k=k+1


orig_array1=orig_array.transpose()
#print(orig_array1)
size1=orig_array1.shape
#print(size1)

m=np.mean(orig_array1, axis=1)
#print(m)
for i in range(size1[1]):
    orig_array1[:,i]=orig_array1[:,i]-m

#print(orig_array1.shape)
x=np.matmul(orig_array1.transpose(),orig_array1)


w, v = LA.eig(x)
#print(w)
#print(v)

k=[]
rows, cols = (len(w),len(w))

#z = np.zeros(shape=(rows,cols))

for i in range(len(w)):
    if w[i]>=1:
        k.append(i)

#print(len(k))


l=0

b= np.zeros(shape=(rows,len(k)))
for i in range(len(k)):

    j=k[i]
    y=0
    for t in range(len(w)):
            b[y][l]=v[t][j]
            y=y+1
    l=l+1
#print(b)
#print(b.shape)

c=np.matmul(orig_array1,b)
g=c.shape
#print(g)

temp3=np.zeros(shape=(len(k),1))


for i in range(g[1]+1):
    temp=orig_array1[:,i][np.newaxis]

    temp1 = np.transpose(temp)

    temp2 = np.matmul(c.transpose(), temp1)
    #print(temp2.shape)
    temp3 = np.concatenate((temp3, temp2), axis=1)


#print(temp3)


testimg = cv2.imread('/home/iiitk/Desktop/yash/test.jpeg')
testimg1 = cv2.resize(testimg, (100, 100))
testgray = cv2.cvtColor(testimg, cv2.COLOR_BGR2GRAY)
testimage=np.zeros(shape=(1,10000))
loop=0
for i in range(100):
    for j in range(100):
            testimage[0][loop]=testgray[i,j]
            loop=loop+1



testimage1=testimage-m
#print(testimage)
testimage2=testimage1.transpose()
projtestimage=np.matmul(c.transpose(),testimage2)
#print(projtestimage)


temp4=[]
temp4=temp3[:,5][np.newaxis]
temp4=temp4.transpose()
#dist=projtestimage-temp4

dist = np.linalg.norm(projtestimage-temp4)
print(dist)


'''

for i in range(g[1]):
    temp=orig_array[:,i][np.newaxis]
    #temp = a[:, i]
    temp1=np.transpose(temp)
    #print(temp1.shape)
    #temp2= d*temp.reshape(temp.size,1)

    temp2=np.matmul(orig_array.transpose(),temp1)

    #print(temp2.shape)
    temp3=np.concatenate((temp3, temp2), axis=1)
    

'''
