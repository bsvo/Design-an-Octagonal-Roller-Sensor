import os
from os import path as osp
import numpy as np
import cv2
import matplotlib.pyplot as plt

from reconstruction import CalibData
from utils import processInitialFrame, match_grad, fast_poisson, detect_contact_area    
import params as pr

from PIL import Image


calib_folder = "./octagonal/for_recon"

# load calib_data
calib_data = CalibData(osp.join(calib_folder, "calib.npz"))
# load background image
bg_img = cv2.imread(osp.join(calib_folder, "frame_0.ppm"))
bg_proc = processInitialFrame(bg_img)

import open3d as o3d
#vis = o3d.visualization.Visualizer()
#vis.create_window()
# load input image
# find ppm files except for frame_0.ppm
imgnames = sorted([fn for fn in os.listdir(calib_folder) if fn.endswith(".ppm") and fn != "frame_0.ppm"])
count = 0
for imgname in imgnames:
    input_fn = osp.join(calib_folder, imgname)
    input_img = cv2.imread(input_fn)
    frame = input_img[pr.border:-pr.border, pr.border:-pr.border, :]
    dI = frame.astype("float") - bg_proc

    #input_img = cv2.imread(osp.join(calib_folder, "frame_1.ppm"))
    #frame = input_img[pr.border:-pr.border, pr.border:-pr.border, :]
    #dI = frame.astype("float") - bg_proc

    dI_single_ch = -np.max(dI, axis=2)
    marker_mask = (dI_single_ch > pr.markerAreaThresh).astype('uint8')
    # overestimate
    sz = 3
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*sz + 1, 2*sz+1), (sz,sz))
    marker_mask = cv2.dilate(marker_mask, element).astype('bool')
    # match grads
    im_grad_x, im_grad_y, im_grad_mag, im_grad_dir =\
        match_grad(dI, marker_mask, calib_data, bg_proc)

    # use contact area detection copying the gradients instead of using all the gradients
    bg_grads_x = np.zeros(dI.shape[:2])
    bg_grads_y = np.zeros(dI.shape[:2])

    contact_area = detect_contact_area(dI, bg_proc, 0.1)
    im_grad_x[~contact_area] = bg_grads_x[~contact_area] 
    im_grad_y[~contact_area] = bg_grads_y[~contact_area] 

    im_grad_y[im_grad_y < 0] = 0
    # coloredVals = np.where(im_grad_y>0)
    # print(coloredVals[0])
    # sumX = 0
    # sumY = 0
    # totalSum = 0
    # for y,x in zip(coloredVals[0],coloredVals[1]):
    #     sumX += im_grad_y[y,x] * x 
    #     sumY += im_grad_y[y,x] * y
    #     totalSum += im_grad_y[y,x]
    # print((sumX/totalSum,sumY/totalSum))
    # print(im_grad_y[im_grad_y > 0])
    # for i in im_grad_y:
    #     print(i)
    # print(im_grad_y.shape)


    colormap = plt.cm.PuBu  
    normed_data = (im_grad_y - np.min(im_grad_y)) / (np.max(im_grad_y) - np.min(im_grad_y))
    colored_data = colormap(normed_data)

    # Convert to image and save
    colored_image = Image.fromarray((colored_data * 255).astype('uint8'), 'RGBA')

    imgName = "recon" + str(count) +".png"
    colored_image.save(imgName)

    # gradClip = np.clip(im_grad_y, 0, None)
    # image = Image.fromarray(gradClip.astype('uint8'))  # Convert to uint8
    # image.save('recon.jpeg')

    fig, axs = plt.subplots(1,2)

    axs[0].imshow(im_grad_x)
    axs[0].set_title("im_grad_x")
    axs[1].imshow(im_grad_y)
    axs[1].set_title("im_grad_y")
    plt.show()
    
    # ----
    # reconstructed height map in mm
    # height_map = fast_poisson(im_grad_x, im_grad_y)*calib_data.pixmm

    # # visualize in matplotlib
    # # plt.imshow(height_map, extent=[0, 1, 0, 1])
    # # plt.show()

    # np.save(input_fn.replace(".jpg", "_hf.npy"), height_map)

    # # visualize 3d
    # h,w = height_map.shape
    # x = np.arange(w) * calib_data.pixmm
    # y = np.arange(h) * calib_data.pixmm
    # mesh_x, mesh_y = np.meshgrid(x, y)

    # xyz = np.dstack([mesh_x, mesh_y, 50*height_map]).reshape(-1, 3)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.visualization.draw_geometries([pcd], window_name="Reconstructed mesh")
    count +=1

