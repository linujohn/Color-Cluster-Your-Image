#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 13:19:26 2018

@author: linu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:50:03 2018

@author: linu
"""

from sklearn.cluster import KMeans
import numpy as np
import cv2
import scipy
import PIL

image = cv2.imread("/home/user/Desktop/kmeans/picture.jpg")
reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
kmeans = KMeans(n_clusters=3, n_init=40, max_iter=500).fit(reshaped)
#print(kmeans.labels_)
#print(cluster_centers_)
#print(kmeans.cluster_centers_)

clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),(image.shape[0], image.shape[1]))
#print(clustering)

    
sortedLabels = sorted([n for n in range(3)],
    key=lambda x: -np.sum(clustering == x))
    
#print(sortedLabels)
x=200
y=200
label=clustering[x][y]
kmeansImage = np.zeros(image.shape[:3], dtype=np.uint8)

for i in enumerate(sortedLabels):
    kmeansImage[clustering == label]=int(255)
    


def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))

    return out

def region_growing(img, seed):
    list = []
    outimg = np.zeros_like(img)
    list.append((seed[0], seed[1]))
    processed = []
    while(len(list) > 0):
        pix = list[0]
        outimg[pix[0], pix[1]] = 255
        for coord in get8n(pix[0], pix[1], img.shape):
            if img[coord[0],coord[1]] != 0:
                outimg[coord[0],coord[1]] = 255
                if not coord in processed:
                    list.append(coord)
                processed.append(coord)
        list.pop(0)
        cv2.imshow("progress",outimg)
        cv2.waitKey(1)
    return outimg



gray_image = cv2.cvtColor(kmeansImage, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)



seed =(x,y)
out = region_growing(img,seed)
backtorgb = cv2.cvtColor(out,cv2.COLOR_GRAY2RGB)

res=cv2.bitwise_and(image,backtorgb)
cv2.imshow('original', image)
cv2.imshow('Region Growing', res)
cv2.imwrite("region.jpg",res)
cv2.waitKey(0)
cv2.destroyAllWindows()

