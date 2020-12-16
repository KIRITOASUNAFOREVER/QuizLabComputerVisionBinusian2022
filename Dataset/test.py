#Nama = Aldo Jonathan Handaka
#NIM = 2201736971

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

#Note:
# Saya menggunakan jupyter notebook maka jika ada error maka coba di jupyter notebook
# FIK = Flann Index Kdtree
# IP = Index Parameter
# SP = Search Parameter

def GrayIMG(img):
    Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return Gray

def BFilter(img):
    Bfilter = cv2.bilateralFilter(img, 15, 75, 75)
    return Bfilter

objectTarget=cv2.imread("object-1.jpg")

Images=[]
for imagePath in os.listdir("Data"):
    if imagePath.split('.')[1]=="jpg":
        Images.append(cv2.imread("Data/"+imagePath))

objectTargetProcessedGray=GrayIMG(objectTarget)
objectTargetProcessedBFilter=BFilter(objectTargetProcessedGray)
Marray = []
for j in range(len(Images)):
    IMG=[Images[j]]
    Images[j]=GrayIMG(Images[j])
    Images[j]=BFilter(Images[j])
    IMGO = cv2.imread("object-1.jpg")
    Surf = cv2.xfeatures2d.SURF_create()
    Surf.setHessianThreshold(8000)
    Kobject, Dobject = Surf.detectAndCompute(IMGO,None)
    Kscene, Dscene = Surf.detectAndCompute(Images[j],None)
    FIK = 0
    IP = dict(algorithm=FIK)
    SP = dict(check=50)
    FLA = cv2.FlannBasedMatcher(IP, SP)
    Match = FLA.knnMatch(Dobject,Dscene,k=2)
    MatchM = []
    TotalM = 0
    
    for i in range(len(Match)):
        MatchM.append([0, 0])
    for i, (m,n) in enumerate(Match):
        if m.distance < 0.7 * n.distance:
            MatchM[i] = [1, 0]
            TotalM += 1
    Marray.append(TotalM)
    
MAX = 0
MAXIndex = 0
for l in range(len(Marray)):
    if(Marray[l] > MAX):
        MAX = Marray[l]
        MAXIndex = l

Surf = cv2.xfeatures2d.SURF_create()
Surf.setHessianThreshold(1000)
IMGO = cv2.imread("object-1.jpg",0)
Kobject, Dobject = Surf.detectAndCompute(IMGO,None)
Kscene, Dscene = Surf.detectAndCompute(Images[MAXIndex],None)

FIK = 0
IP = dict(algorithm = FIK)
SP = dict(check=50)
FLA = cv2.FlannBasedMatcher(IP, SP)

Match = FLA.knnMatch(Dobject,Dscene,k=2)
MatchM = []

for i in range(len(Match)):
    MatchM.append([0,0])
    
for i, (z,q) in enumerate(Match):
    if z.distance < 0.7 * q.distance:
        MatchM[i] = [1,0]
        
IMGR = cv2.drawMatchesKnn(
    IMGO, Kobject,
    Images[MAXIndex], Kscene,
    Match, None,
    matchColor=[0, 255, 0],
    singlePointColor=[0, 0, 255],
    matchesMask= MatchM
)

plt.imshow(IMGR)
plt.xticks([]), plt.yticks([])
plt.show()