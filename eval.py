
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
  

# blue channel centroid
def calculate_centroids(directory):
    allPngs = list_png_files(directory)
    centroids = []
    for i in allPngs:
        img = Image.open(i)
        print(img)
        img = img.convert('RGB')
        image_array = np.array(img)
        indices = np.indices((image_array.shape[0], image_array.shape[1]))
        channel = image_array[:, :, 2]
        total_intensity = channel.sum()
        x_centroid = (indices[1] * channel).sum() / total_intensity
        y_centroid = (indices[0] * channel).sum() / total_intensity
        centroids.append((x_centroid,y_centroid))
    return centroids

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


def main(): 
    # print("MSE","PSNA")
    # print(evalFly("./octagonal/eval/og")) 

    print(calculate_centroids("./octagonal/eval/ball")) 


if __name__ == "__main__": 
    main() 