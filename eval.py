
from math import log10, sqrt 
import cv2 
import numpy as np 
import os

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
  
def main(): 
    allPngs = list_png_files("./octagonal/eval/og")
    recon = [] 
    ev = []
    for i in allPngs:
        fname = i.split('/')[-1]
        if fname[0] == 'e':
            ev.append(i)
        else:
            recon.append(i)

    recon = sorted(recon, key = lambda num: int(num[-5]))

    print(recon)
    ev = sorted(ev, key = lambda num: int(num[-5]))
    print("bruh")
    for r,e in zip(recon,ev):
        print(r)
        print(e)
        re = cv2.imread(r) 
        ev = cv2.imread(e) 
        print(MSEPSNR(re, ev))
    #  original = cv2.imread("original_image.png") 
    #  compressed = cv2.imread("compressed_image.png", 1) 
    #  value = PSNR(original, compressed) 
    #  print(f"PSNR value is {value} dB") 
       
if __name__ == "__main__": 
    main() 