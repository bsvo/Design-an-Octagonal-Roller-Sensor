
from math import log10, sqrt 
import cv2 
import numpy as np 
import os
from PIL import Image


def list_png_files(root_directory):
    png_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".png"):
                png_files.append(os.path.join(root, file))
    return png_files
  



def MSEPSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return mse,psnr 

def evalFly(directory):
    allPngs = list_png_files(directory)
    recon = [] 
    ev = []
    for i in allPngs:
        fname = i.split('/')[-1]
        if fname[0] == 'e':
            ev.append(i)
        else:
            recon.append(i)

    recon = sorted(recon, key = lambda num: int(num[-5]))

    ev = sorted(ev, key = lambda num: int(num[-5]))

    toReturn = []
    for r,e in zip(recon,ev):
        re = cv2.imread(r) 
        ev = cv2.imread(e) 
        toReturn.append(MSEPSNR(re, ev))
    return toReturn

def euclideanDist(p1,p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))



def main(): 
    # print("MSE","PSNA")
    # print(evalFly("./octagonal/eval/og")) 

    # print(calculate_centroids("./octagonal/eval/ball")) 
    flyStats = [[0.12716, 57.087, 0.118, 57.413],
    [0.136, 56.79, 0.09375,	58.411],
    [0.20346, 55.046, 0.13787, 56.736],
    [0.1287, 57.033, 0.0914, 58.5215],
    [0.193,	55.28, 0.1225, 57.25],
    [0.2069, 54.974, 0.116, 57.483]]

    flyErr = np.array(flyStats)

    # Average OG MSE, OG PCNA, BG RM MSE, BG RM PCNA
    for i in range(len(flyErr[0])):
        print(np.mean(flyErr[:,i]))

    centroidStats = np.array([[(255.59,18.913),(253.86,31.73)],
    [(385.61, 142.72),(378,149)],
    [(86.675,202.845),(98.89, 201.89)],
    [(337.139, 242.596),(332.27,237.165)]])

    ballStats = np.array(centroidStats)

    for c1,c2 in centroidStats:
        print(0.12071 * euclideanDist(c1,c2))


if __name__ == "__main__": 
    main() 