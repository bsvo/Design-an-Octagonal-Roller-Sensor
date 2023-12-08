import gc
from glob import glob
from os import path as osp
import numpy as np
import cv2
import matplotlib.cm as cm

# import nanogui as ng
# from nanogui import Texture
# from nanogui import glfw


#from gsight_utilities_pack.calibration.utils import processInitialFrame, match_grad, fast_poisson
#from gsight_utilities_pack.calibration import params as pr
from utils import processInitialFrame, match_grad, fast_poisson
import params as pr

w, h = 640, 480

class CalibData:
  """docstring for CalibData"""
  def __init__(self, fn):
    self.fn = fn
    data = np.load(fn)

    # automate this
    self.bins = data['bins']
    self.grad_mag = data['grad_mag']
    self.grad_dir = data['grad_dir']
    self.zeropoint = data['zeropoint']
    self.scale = data['zeropoint']
    self.scale = data['scale']
    self.frame_sz = data['frame_sz']
    self.pixmm = data['pixmm']


# class ReconApp(ng.Screen):
#   orig_img = None
#   change = False
#   height_map = None
#   def __init__(self):
#     super(ReconApp, self).__init__((1624, 768), "Gelsight Reconstruction App")

#     window = ng.Window(self, "IO Window")
#     window.set_position((15, 15))
#     window.set_layout(ng.GroupLayout())

#     # calib data folder 
#     ng.Label(window, "Folder dialog", "sans-bold")
#     tools = ng.Widget(window)
#     tools.set_layout(ng.BoxLayout(ng.Orientation.Horizontal,
#                               ng.Alignment.Middle, 0, 6))
#     b = ng.Button(tools, "Open")

#     def cb():
#         self.img_data_dir = ng.directory_dialog("")
#         print("Selected directory = %s" % self.img_data_dir)

#         # check for background Frame
#         # obtains fnames(currently accepts jpg/ppm/png)
#         self.fnames = glob(osp.join(self.img_data_dir, "*.jpg")) +\
#               glob(osp.join(self.img_data_dir, "*.ppm")) +\
#               glob(osp.join(self.img_data_dir, "*.png"))

#         self.next_img_num = 0
#         # print(self.fnames)

#         self.background_check(self.fnames)

#         # check for calib file
#         fnlist = glob(osp.join(self.img_data_dir, "*.npz"))
#         if(len(fnlist)==1):
#           self.calib_fn = fnlist[0]
#           self.calib_data = CalibData(self.calib_fn)
#         elif(len(fnlist)>1):
#           print("More than one calib files(.npz) found! The calibration folder should contain only 1 npz calib file")
#         else:
#           print("No calibration file(.npz) found! Please place the background image and calibration file in the calibration folder.")

#     b.set_callback(cb)

#     ng.Label(window, "File dialog", "sans-bold")
#     tools = ng.Widget(window)
#     tools.set_layout(ng.BoxLayout(ng.Orientation.Horizontal,
#                               ng.Alignment.Middle, 0, 6))
#     b = ng.Button(tools, "Open")
#     valid = [("png", "Portable Network Graphics"),\
#                  ("jpg", "JPEG"),\
#                  ("ppm", "Portable Pixmap Image File")]

#     def recon_cb():
#         result = ng.file_dialog(valid, False)
#         print("File dialog result = %s" % result)
#         if result != "":
#           self.change = True
#           self.orig_img = cv2.imread(result)

#           # calculate gradients
#           assert pr.border>0
#           frame = self.orig_img[pr.border:-pr.border, pr.border:-pr.border, :]
#           dI = frame.astype("float") - self.bg_proc

#           # find markers
#           dI_single_ch = -np.max(dI, axis=2)
#           marker_mask = (dI_single_ch > pr.markerAreaThresh).astype('uint8')
#           # overestimate
#           sz = 3
#           element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*sz + 1, 2*sz+1), (sz,sz))
#           marker_mask = cv2.dilate(marker_mask, element).astype('bool')
#           # match grads
#           im_grad_x, im_grad_y, im_grad_mag, im_grad_dir =\
#                       match_grad(dI, marker_mask, self.calib_data, self.bg_proc)

#           # reconstructed height map in mm
#           self.height_map = fast_poisson(im_grad_x, im_grad_y)*self.calib_data.pixmm



#     b.set_callback(recon_cb)

#     self.img_window = ng.Window(self, "Current image")
#     self.img_window.set_position((150, 15))
#     # print(self.img_window.size())
#     self.img_window.set_layout(ng.GroupLayout())

#     self.img_view = ng.ImageView(self.img_window)

#     self.img_tex = ng.Texture(
#                       pixel_format=Texture.PixelFormat.RGB, 
#                       component_format=Texture.ComponentFormat.UInt8,
#                       size=[w, h],
#                       min_interpolation_mode=Texture.InterpolationMode.Trilinear,
#                       mag_interpolation_mode=Texture.InterpolationMode.Nearest,
#                       flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
#                   )

#     self.depth_window = ng.Window(self, "Depth image")
#     self.depth_window.set_position((600, 15))
#     # print(self.depth_window.size())
#     self.depth_window.set_layout(ng.GroupLayout())

#     self.depth_view = ng.ImageView(self.depth_window)

#     self.depth_tex = ng.Texture(
#                       pixel_format=Texture.PixelFormat.RGB, 
#                       component_format=Texture.ComponentFormat.Float32,
#                       size=[w-2*pr.border, h-2*pr.border],
#                       min_interpolation_mode=Texture.InterpolationMode.Trilinear,
#                       mag_interpolation_mode=Texture.InterpolationMode.Nearest,
#                       flags= Texture.TextureFlags.RenderTarget
#                   )

#     self.perform_layout()

#   def background_check(self, fnames):
#     found = False
#     for fnId, fn in enumerate(fnames):
#       baseFn = osp.basename(fn)
#       if(baseFn == "frame_0.ppm" or \
#           baseFn == "frame0.jpg" or \
#           baseFn == "frame0.png"):
#         self.bg_img_fn = fn
#         self.bg_id = fnId

#         self.bg_img = cv2.imread(self.bg_img_fn)
#         self.bg_proc = processInitialFrame(self.bg_img)
#         found = True
#         break

#     if not found:
#       print("No background Image Found! Looking for frame_0.ppm/frame0.jpg/frame0.png")
#       self.set_visible(False)

#   def draw(self, ctx):
#     self.img_window.set_size((400,400))
#     self.img_view.set_size((w, h))

#     self.depth_window.set_size((400,400))
#     self.depth_view.set_size((w, h))

#     if self.change:
#       # Add to img view
#       img_data = self.orig_img.copy()
#       if(self.img_tex.channels() > 3):
#         height, width = img_data.shape[:2]
#         alpha = 255*np.ones((height, width,1), dtype=img_data.dtype)
#         img_data = np.concatenate((img_data, alpha), axis=2)
      
#       self.img_tex.upload(img_data)
#       self.img_view.set_image(self.img_tex)
#       # self.img_view.center()

#       if(self.height_map is not None):
#         # print("uploading height map")
#         height_map_f = self.height_map

#         # Crashes in macOS
#         # print("Preview of depth crashes in macOS. Therfore file is dumped to /tmp/depth.tiff")
#         # scale b/w [0,1]
#         height_map_f = (height_map_f - height_map_f.min())/(height_map_f.max() - height_map_f.min())
#         height_map_c = cm.viridis(height_map_f).astype("float32")
#         self.depth_tex.upload(height_map_c)
        
#         self.depth_view.set_image(self.depth_tex)
#         # self.depth_view.center()
#         # print("Done uploading height")

#         self.change = False

#     super(ReconApp, self).draw(ctx)


#   def keyboard_event(self, key, scancode, action, modifiers):
#     if super(ReconApp, self).keyboard_event(key, scancode,
#                                           action, modifiers):
#         return True
#     if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
#         self.set_visible(False)
#         return True


# if __name__ == "__main__":
#   ng.init()
#   app = ReconApp()
#   app.draw_all()
#   app.set_visible(True)
#   ng.mainloop(refresh=1 / 60.0 * 1000)
#   del app
#   gc.collect()
#   ng.shutdown()