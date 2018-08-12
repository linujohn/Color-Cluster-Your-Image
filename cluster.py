#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:50:03 2018

@author: linu
"""

from sklearn.cluster import KMeans
import numpy as np
import cv2

image = cv2.imread("blue.jpg")
reshaped = image.reshape(image.shape[0] * image.shape[1], image.shape[2])
kmeans = KMeans(n_clusters=6, n_init=40, max_iter=500).fit(reshaped)

clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),(image.shape[0], image.shape[1]))
   
sortedLabels = sorted([n for n in range(3)],
    key=lambda x: -np.sum(clustering == x))
    

x=400
y=400
label=clustering[x][y]
kmeansImage = np.zeros(image.shape[:3], dtype=np.uint8)

for i in enumerate(sortedLabels):
    kmeansImage[clustering == label]=int(255)
    

res=cv2.bitwise_and(image,kmeansImage)

cv2.imshow("clustered",res)
cv2.waitKey(0)
cv2.destroyAllWindows()

