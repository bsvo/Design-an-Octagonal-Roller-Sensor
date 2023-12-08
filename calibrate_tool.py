import gc
from glob import glob
from os.path import join, splitext, isdir, exists, basename
import numpy as np
import cv2

import nanogui as ng
from nanogui import Texture
from nanogui import glfw

from utils import overlay_circle, processInitialFrame, find_ball_params, lookuptable_from_ball, lookuptable_smooth
import params as pr
import per_sensor_params as psp

# w, h = 640, 480


# w,h = 3280//4, 2464//4

class Circle:
    """docstring for Circle"""
    color_circle = (0, 128, 0)
    radius = 20
    increments = 4
    opacity = 0.5

    def __init__(self, w, h):
        self.center = [w / 2, h / 2]


class Grads:
    """docstring for Grads"""
    grad_mag = None
    grad_dir = None
    countmap = None


class CalibrateApp(ng.Screen):
    fnames = list()
    read_all = False
    load_img = True
    change = False
    bg_img_fn = None
    bg_id = None

    def __init__(self):
        self.grads = Grads()

        super(CalibrateApp, self).__init__((1024, 768), "Gelsight Calibration App")

        window = ng.Window(self, "IO Window")
        window.set_position((15, 15))
        window.set_layout(ng.GroupLayout())

        ng.Label(window, "Folder dialog", "sans-bold")
        tools = ng.Widget(window)
        tools.set_layout(ng.BoxLayout(ng.Orientation.Horizontal,
                                      ng.Alignment.Middle, 0, 6))

        text_box = ng.TextBox(tools, "image directory")
        text_box.set_editable(True)
        text_box.set_fixed_size((500, 25))
        text_box.set_value(join("..", "data", "NETID", "SENSOR_NAME"))
        text_box.set_font_size(20)
        text_box.set_alignment(ng.TextBox.Alignment.Right)

        b = ng.Button(tools, "Open")

        def cb():
            self.img_data_dir = text_box.value()
            # check if it is a valid dir
            if not isdir(self.img_data_dir):
                print(f"Not a valid dir({self.img_data_dir})\nPlease enter valid dir")
            else:
                print("Selected directory = %s" % self.img_data_dir)
                # check for background Frame
                # obtains fnames(currently accepts jpg/ppm/png)
                self.fnames = glob(join(self.img_data_dir, "*.jpg")) + \
                              glob(join(self.img_data_dir, "*.ppm")) + \
                              glob(join(self.img_data_dir, "*.jpeg")) + \
                              glob(join(self.img_data_dir, "*.png"))
                self.fnames = sorted(self.fnames)

                self.next_img_num = 0
                print(self.fnames)

                self.background_check(self.fnames)

        # def cb():
        #     self.img_data_dir = ng.directory_dialog(osp.join("..", ".." , "calib_folder", "sensor006"))
        #     print("Selected directory = %s" % self.img_data_dir)

        #     # check for background Frame
        #     # obtains fnames(currently accepts jpg/ppm/png)
        #     self.fnames = glob(osp.join(self.img_data_dir, "*.jpg")) +\
        #           glob(osp.join(self.img_data_dir, "*.ppm")) +\
        #           glob(osp.join(self.img_data_dir, "*.jpeg")) +\
        #           glob(osp.join(self.img_data_dir, "*.png"))

        #     self.next_img_num = 0
        #     print(self.fnames)

        #     self.background_check(self.fnames)

        b.set_callback(cb)

        # help message
        help_win = ng.Window(self, "Help box")
        help_win.set_position((15, 200))
        help_win.set_layout(ng.GroupLayout())

        helpWidget = ng.Widget(help_win)
        helpWidget.set_layout(ng.BoxLayout(ng.Orientation.Vertical,
                                  ng.Alignment.Middle, 0, 6))
        
        def addRow(wid, keys, desc):
            row = ng.Widget(wid)
            row.set_layout(ng.BoxLayout(ng.Orientation.Horizontal, ng.Alignment.Middle, 0, 10))

            descWidget = ng.Label(row, desc)
            descWidget.set_fixed_size(200)
            ng.Label(row, keys)

        addRow(helpWidget, "c/f", "Increase/Decrease circle increments")
        addRow(helpWidget, "m/p", "Decrease/Increase circle radius")
        addRow(helpWidget, "Arrow keys", "Move circle center")
        addRow(helpWidget, "Calibrate", "Use the current image and current circle params for calibration")
        addRow(helpWidget, "Save Params", "Save params to calib.npz")

        # image view
        self.img_window = ng.Window(self, "Current image")
        self.img_window.set_position((200, 150))
        print(self.img_window.size())
        self.img_window.set_layout(ng.GroupLayout())

        b = ng.Button(self.img_window, "Skip")
        def skip_cb():
            self.load_img = True
            self.update_img_idx()

        b.set_callback(skip_cb)

        b = ng.Button(self.img_window, "Calibrate")

        # Calibrate button
        def calibrate_cb():
            print("Use for calibration")
            assert pr.border > 0
            frame = self.orig_img[pr.border:-pr.border, pr.border:-pr.border, :]

            dI = frame.astype("float") - self.bg_proc
            dI_single_ch = -np.max(dI, axis=2)

            touch_center = self.circle.center
            radius = self.circle.radius
            
            # cache the data and load it next time for center location
            img_fn = self.fnames[self.next_img_num]
            _, ext = splitext(img_fn)
            with open(img_fn.replace(ext, "_circle.txt"), "w") as f:
                print(f"center: {touch_center}, radius: {radius}")
                f.write("%d %d %d"%(touch_center[0], touch_center[1], radius))
            # --

            contact_mask, valid_mask = find_ball_params(dI_single_ch, frame, self.circle)

            valid_mask = valid_mask & contact_mask

            nomarker_mask = (np.min(-dI, axis=2) < pr.noMarkerMaskThreshold).astype('uint8')
            # underestimate
            sz = 3
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sz + 1, 2 * sz + 1), (sz, sz))
            nomarker_mask = cv2.erode(nomarker_mask, element)

            valid_mask = valid_mask & nomarker_mask

            # updates self.grads
            lookuptable_from_ball(dI, self.bg_proc, \
                                  self.circle, valid_mask, self.grads)
            # Update img index
            self.load_img = True
            self.update_img_idx()

        b.set_callback(calibrate_cb)

        b = ng.Button(self.img_window, "Save Params")

        def cb():
            out_fn_path = join(self.img_data_dir, "calib.npz")
            print("Saving params to %s" % out_fn_path)
            grad_mag, grad_dir = lookuptable_smooth(self.grads)
            
            # store non-smooth LUT 
            grad_mag = self.grads.grad_mag
            grad_dir = self.grads.grad_dir

            np.savez(out_fn_path.replace(".npz", "_nonsmooth.npz"),\
                bins=pr.numBins, 
                grad_mag = grad_mag,
                grad_dir = grad_dir, 
                gradx = -np.cos(grad_dir)*grad_mag,
                grady = -np.sin(grad_dir)*grad_mag,
                zeropoint = psp.zeropoint, 
                scale = psp.lookscale, 
                pixmm = psp.pixmm,
                frame_sz = self.bg_img.shape)
            # ---
            grad_mag, grad_dir = lookuptable_smooth(self.grads, verbose=True)
              
            np.savez(out_fn_path,\
                bins=pr.numBins, 
                grad_mag = grad_mag,
                grad_dir = grad_dir, 
                gradx = -np.cos(grad_dir)*grad_mag,
                grady = -np.sin(grad_dir)*grad_mag,
                zeropoint = psp.zeropoint, 
                scale = psp.lookscale, 
                pixmm = psp.pixmm,
                frame_sz = self.bg_img.shape)
            print("Saved!")

        b.set_callback(cb)

        self.img_view = ng.ImageView(self.img_window)

        self.img_tex = ng.Texture(
            pixel_format=Texture.PixelFormat.RGB,
            component_format=Texture.ComponentFormat.UInt8,
            size=[psp.width, psp.height],
            min_interpolation_mode=Texture.InterpolationMode.Trilinear,
            mag_interpolation_mode=Texture.InterpolationMode.Nearest,
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )

        self.perform_layout()

    def background_check(self, fnames):
        found = False
        for fnId, fn in enumerate(fnames):
            baseFn = basename(fn)
            if (baseFn == "frame_0.ppm" or \
                    baseFn == "frame0.jpg" or \
                    baseFn == "frame0.jpeg" or \
                    baseFn == "frame0.png"):
                self.bg_img_fn = fn
                self.bg_id = fnId

                self.bg_img = cv2.imread(self.bg_img_fn)
                self.bg_proc = processInitialFrame(self.bg_img)
                found = True
                break

        if not found:
            print("No background Image Found! Looking for frame_0.ppm/frame0.jpeg//frame0.jpg/frame0.png")
            self.set_visible(False)

    def update_img_idx(self):
        self.next_img_num += 1
        if (self.next_img_num == len(self.fnames)): self.read_all = True

    def draw(self, ctx):
        self.img_window.set_size((700, 700))
        self.img_view.set_size((psp.width, psp.height))

        # load a new image
        if (self.load_img and len(self.fnames) > 0 and not self.read_all):
            if (self.next_img_num == self.bg_id):
                self.update_img_idx()
            if not self.read_all:
                print("Loading %s" % self.fnames[self.next_img_num])

                # Load img
                self.orig_img = cv2.imread(self.fnames[self.next_img_num])
                size = self.orig_img.shape
                # check for cached circle info
                img_fn = self.fnames[self.next_img_num]
                _, ext = splitext(img_fn)
                circle_fn = img_fn.replace(ext, "_circle.txt")
                if exists(circle_fn):
                    print(f"Loading circle info from {circle_fn}")
                    with open(circle_fn, "r") as f:
                        line = f.readline()
                        touch_center = [int(x) for x in line.split()]
                        self.circle = Circle(size[0], size[1])
                        self.circle.center = (touch_center[0], touch_center[1])
                        self.circle.radius = touch_center[2]
                        print(f"touch center: {touch_center[:2]}, radius: {touch_center[2]}")
                else:
                    print("Default circle center and radius")
                    self.circle = Circle(size[0], size[1])

        # Add circle and add img to viewer
        if ((self.load_img and len(self.fnames) > 0) or self.change):
            # print("update")
            self.load_img = False
            self.change = False
            # Add circle
            img = overlay_circle(self.orig_img, self.circle)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if (self.img_tex.channels() > 3):
                height, width = img.shape[:2]
                alpha = 255 * np.ones((height, width, 1), dtype=img.dtype)
                img = np.concatenate((img, alpha), axis=2)

            # Add to img view
            self.img_tex.upload(img)
            self.img_view.set_image(self.img_tex)

        super(CalibrateApp, self).draw(ctx)

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(CalibrateApp, self).keyboard_event(key, scancode,
                                                    action, modifiers):
            return True
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True
        elif key == glfw.KEY_C:
            self.circle.increments *= 2
        elif key == glfw.KEY_F:
            self.circle.increments /= 2
        else:
            self.change = True
            if (key == glfw.KEY_LEFT):
                self.circle.center[0] -= self.circle.increments
            elif (key == glfw.KEY_RIGHT):
                self.circle.center[0] += self.circle.increments
            elif (key == glfw.KEY_UP):
                self.circle.center[1] -= self.circle.increments
            elif (key == glfw.KEY_DOWN):
                self.circle.center[1] += self.circle.increments
            elif key == glfw.KEY_M:
                self.circle.radius -= 2
            elif key == glfw.KEY_P:
                self.circle.radius += 2

        return False


if __name__ == "__main__":
    ng.init()
    app = CalibrateApp()
    app.draw_all()
    app.set_visible(True)
    ng.mainloop(refresh=1 / 60.0 * 1000)
    del app
    gc.collect()
    ng.shutdown()
