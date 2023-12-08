import cv2
import numpy as np
from math import sin, cos, radians, degrees
from scipy.ndimage import gaussian_filter
from PIL import ImageFilter
from scipy.spatial.distance import cdist
from scipy.fftpack import dst, idst
from skimage.morphology import binary_opening
from sklearn.metrics import pairwise_distances_chunked

#from gsight_utilities_pack.calibration import params as pr
#from gsight_utilities_pack.calibration import per_sensor_params as psp
import params as pr
import per_sensor_params as psp


def detect_contact_area(dI, bg_proc, thresh, verbose=False):
    kscale = 5
    convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)
    # gaussian smoothing bg
    bg_proc_d = bg_proc.astype('float')
    for i in range(3):
        bg_proc_d[:,:,i] = convEachDim(bg_proc_d[:,:,i])

    if verbose:
        import matplotlib.pyplot as plt
        plt.imshow(dI/bg_proc_d)
        plt.title("signal")
        plt.show()

    signal = dI/bg_proc_d

    contact_area = (signal[...,0] > thresh) | (signal[...,1] > thresh) | (signal[...,2] > thresh)

    # mean filtering
    contact_area = binary_opening(contact_area, footprint=np.ones((9, 9)))

    if verbose:
        plt.imshow(contact_area)
        plt.title("Contact area")
        plt.show()

    return contact_area

def overlay_circle(orig_img, circle):
    center = circle.center
    radius = circle.radius
    color_circle = circle.color_circle
    opacity = circle.opacity

    overlay = orig_img.copy()
    center_tuple = (int(center[0]), int(center[1]))
    cv2.circle(overlay, center_tuple, radius, color_circle, -1)
    # line_end = (int(center[0]+radius*cos(radians(theta))), int(center[1]+radius*sin(radians(theta))))
    # cv2.line(overlay, center_tuple, line_end, color_line, 1)
    cv2.addWeighted(overlay, opacity, orig_img, 1 - opacity, 0, overlay)
    return overlay


def processInitialFrame(img):
    # gaussian filtering with square kernel with
    # filterSize : kscale*2+1
    # sigma      : kscale
    kscale = pr.kscale
    border_sz = pr.border

    img_d = img.astype('float')
    convEachDim = lambda in_img: gaussian_filter(in_img, kscale)

    f0 = img.copy()
    for ch in range(img_d.shape[2]):
        f0[:, :, ch] = convEachDim(img_d[:, :, ch])

    # Removing border pixels
    # check size
    removeBorder = lambda in_img: in_img[border_sz:-border_sz, border_sz:-border_sz, :]

    f0 = removeBorder(f0)
    frame_ = removeBorder(img_d)

    # Checking the difference between original and filtered image
    diff_threshold = pr.diffThreshold
    dI = np.mean(f0 - frame_, axis=2)
    idx = np.nonzero(dI < diff_threshold)

    # Mixing image based on the difference between original and filtered image
    frame_mixing_per = pr.frameMixingPercentage
    h, w, ch = f0.shape
    pixcount = h * w

    for ch in range(f0.shape[2]):
        f0[:, :, ch][idx] = frame_mixing_per * f0[:, :, ch][idx] + (1 - frame_mixing_per) * frame_[:, :, ch][idx]

    return f0


def find_ball_params(I, frame, circle):
    h, w = I.shape

    # find overestimated marker masks
    valid_map = (I > pr.markerAreaThresh).astype('uint8')
    sz = 7
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * sz + 1, 2 * sz + 1), (sz, sz))

    valid_map = np.bitwise_not(cv2.dilate(valid_map, element))

    I = I * valid_map

    xcoord, ycoord = np.meshgrid(range(w), range(h))
    center = circle.center
    xcoord = xcoord - center[0]
    ycoord = ycoord - center[1]
    rsqcoord = xcoord * xcoord + ycoord * ycoord

    rad = circle.radius
    rad_sq = rad * rad
    contact_mask = rsqcoord < (rad_sq)

    return contact_mask, valid_map


def lookuptable_from_ball(dI, bg_img, circle, valid_mask, grads):
    bins = pr.numBins
    ball_radius_pix = psp.ball_radius / psp.pixmm

    zeropoint = psp.zeropoint
    lookscale = psp.lookscale

    center = circle.center
    radius = circle.radius

    f01 = bg_img.copy()
    f01[f01 == 0] = 1
    t = np.mean(f01)
    # f01 = 1 + ((t / bg_img) - 1) * 2
    f01 = 1 + ((t / f01) - 1) * 2

    sizey, sizex = dI.shape[:2]
    [xq, yq] = np.meshgrid(range(sizex), range(sizey))
    xq = xq - center[0]
    yq = yq - center[1]

    validId = np.nonzero(valid_mask)
    xvalid = xq[validId];
    yvalid = yq[validId]
    rvalid = np.sqrt(xvalid * xvalid + yvalid * yvalid)

    if (np.max(rvalid - ball_radius_pix) > 0):
        print("Contact Radius(%f) is too large(%f). Ignoring the exceeding area" % (np.max(rvalid), ball_radius_pix))
        rvalid[rvalid > ball_radius_pix] = ball_radius_pix - 0.001

    gradxseq = np.arcsin(rvalid / ball_radius_pix);
    gradyseq = np.arctan2(-yvalid, -xvalid)

    binm = bins - 1

    r1 = dI[:, :, 0][validId] * f01[:, :, 0][validId]
    g1 = dI[:, :, 1][validId] * f01[:, :, 1][validId]
    b1 = dI[:, :, 2][validId] * f01[:, :, 2][validId]

    rgb1 = np.stack((r1, g1, b1), axis=1)

    rgb2 = (rgb1 - zeropoint) / lookscale
    rgb2[rgb2 < 0] = 0;
    rgb2[rgb2 > 1] = 1

    rgb3 = np.floor(rgb2 * binm).astype('int')

    r3 = rgb3[:, 0]
    g3 = rgb3[:, 1]
    b3 = rgb3[:, 2]

    # initialize for the first time
    if grads.grad_mag is None:
        grads.grad_mag = np.zeros((bins, bins, bins))
        grads.grad_dir = np.zeros((bins, bins, bins))
        grads.countmap = np.zeros((bins, bins, bins), dtype='uint8')

    tmp = grads.countmap[r3, g3, b3]
    grads.countmap[r3, g3, b3] = grads.countmap[r3, g3, b3] + 1

    # when the gradient is added the first time
    idx = np.where(tmp == 0)
    grads.grad_mag[r3[idx], g3[idx], b3[idx]] = gradxseq[idx]
    grads.grad_dir[r3[idx], g3[idx], b3[idx]] = gradyseq[idx]

    # updating the gradients
    idx = np.where(tmp > 0)
    grads.grad_mag[r3[idx], g3[idx], b3[idx]] = (grads.grad_mag[r3[idx], g3[idx], b3[idx]] * tmp[idx] + gradxseq[
        idx]) / (tmp[idx] + 1)

    # wrap around checks
    a1 = grads.grad_dir[r3[idx], g3[idx], b3[idx]]
    a2 = gradyseq[idx]

    diff_angle = a2 - a1
    a2[diff_angle > np.pi] -= 2 * np.pi
    a2[-diff_angle > np.pi] += 2 * np.pi

    grads.grad_dir[r3[idx], g3[idx], b3[idx]] = (a1 * tmp[idx] + a2) / (tmp[idx] + 1)


def lookuptable_smooth(grads, verbose=False):
  bins = pr.numBins
  countmap = grads.countmap
  grad_mag =  grads.grad_mag
  grad_dir = grads.grad_dir

  if not countmap[0,0,0] or\
      countmap[0,0,0] == 1:
    
    grad_mag[0,0,0] == 0
    grad_dir[0,0,0] == 0

  validid = np.nonzero(countmap)

  # no interpolation needed
  if(validid[0].size == (bins**3)):
    return grad_mag, grad_dir

  if verbose: print(f"Create meshgrid for {bins}")
  # Nearest neighbor interpolation
  xout, yout, zout = np.meshgrid(range(bins), range(bins), range(bins))

  invalid_id = np.nonzero((countmap == 0))

  xvalid = xout[validid]; yvalid = yout[validid]; zvalid = zout[validid]
  xinvalid = xout[invalid_id]; yinvalid = yout[invalid_id]; zinvalid = zout[invalid_id]

  xyzvalid = np.stack((xvalid, yvalid, zvalid), axis=1)
  xyzinvalid = np.stack((xinvalid, yinvalid, zinvalid), axis=1)

  # perform following operation in batches as big matrix allocation crashes the system
  # if verbose: print(f"Compute cdist {xyzinvalid.shape[0]} x {xyzvalid.shape[0]}")
  # dist = cdist(xyzinvalid, xyzvalid)
  
  dist_gen = pairwise_distances_chunked(xyzinvalid, xyzvalid)
  num_processed_invalid_pts = 0
  for dist in dist_gen:
    print(f"processing {num_processed_invalid_pts+dist.shape[0]}/{invalid_id[0].size} invalid pts")
    closest_id = np.argmin(dist, axis=1)
    closest_valid_idx = (validid[0][closest_id], validid[1][closest_id], validid[2][closest_id])

    
    id_into_invalid_id_arr = num_processed_invalid_pts + np.arange(dist.shape[0])
    curr_invalid_id = (invalid_id[0][id_into_invalid_id_arr], invalid_id[1][id_into_invalid_id_arr], invalid_id[2][id_into_invalid_id_arr])
    grad_mag[curr_invalid_id] = grad_mag[closest_valid_idx]
    grad_dir[curr_invalid_id] = grad_dir[closest_valid_idx]

    num_processed_invalid_pts += dist.shape[0]

  return grad_mag, grad_dir


# recon
def match_grad(dI, marker_mask, calib_data, bg_img):
    f01 = bg_img.copy()
    # f01[f01==0] = 1
    f01[f01==0] = 1
    t = np.mean(f01)
    f01 = 1 + ((t / f01) - 1) * 2

    sizex, sizey = dI.shape[:2]

    im_grad_mag = np.zeros((sizex, sizey))
    im_grad_dir = np.zeros((sizex, sizey))

    ndI = dI * f01

    binm = calib_data.bins - 1

    valid_mask = np.bitwise_not(marker_mask)
    validId = np.nonzero(valid_mask)

    rgb1 = ndI[validId]
    rgb2 = (rgb1 - calib_data.zeropoint) / calib_data.scale
    rgb2[rgb2 > 1] = 1
    rgb2[rgb2 < 0] = 0

    rgb3 = np.floor(rgb2 * binm).astype('uint')
    r3 = rgb3[:, 0]
    g3 = rgb3[:, 1]
    b3 = rgb3[:, 2]

    im_grad_mag[validId] = calib_data.grad_mag[r3, g3, b3]
    im_grad_dir[validId] = calib_data.grad_dir[r3, g3, b3]

    tmp = np.tan(im_grad_mag)
    im_grad_x = tmp * np.cos(im_grad_dir)
    im_grad_y = tmp * np.sin(im_grad_dir)

    return im_grad_x, im_grad_y, im_grad_mag, im_grad_dir


def fast_poisson(gx, gy):
    [h, w] = gx.shape
    gxx = np.zeros((h, w))
    gyy = np.zeros((h, w))

    j = np.arange(h - 1)
    k = np.arange(w - 1)

    [tmpx, tmpy] = np.meshgrid(j, k)
    gyy[tmpx + 1, tmpy] = gy[tmpx + 1, tmpy] - gy[tmpx, tmpy];
    gxx[tmpx, tmpy + 1] = gx[tmpx, tmpy + 1] - gx[tmpx, tmpy];

    f = gxx + gyy

    f2 = f[1:-1, 1:-1]

    tt = dst(f2, type=1, axis=0) / 2

    f2sin = (dst(tt.T, type=1, axis=0) / 2).T

    [x, y] = np.meshgrid(np.arange(1, w - 1), np.arange(1, h - 1))

    denom = (2 * np.cos(np.pi * x / (w - 1)) - 2) + (2 * np.cos(np.pi * y / (h - 1)) - 2)

    f3 = f2sin / denom

    [xdim, ydim] = tt.shape
    tt = idst(f3, type=1, axis=0) / (xdim)
    img_tt = (idst(tt.T, type=1, axis=0) / ydim).T

    img_direct = np.zeros((h, w))
    img_direct[1:-1, 1:-1] = img_tt

    return img_direct
