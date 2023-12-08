import numpy as np
from time import sleep
import gc

import nanogui as ng
from nanogui import Texture
from nanogui import glfw

import cv2
import os
import os.path as osp
from gelsight import gsdevice

EXPERIMENT_ROOT = "/home/okemo/Desktop/tactile_lab_session/data"
class CaptureApp(ng.Screen):
    img_save_num = 0

    def __init__(self, sensor_name, netid):
        # create camera
        # the device ID can change after unplugging and changing the usb ports.
        # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        self.sensor_name = sensor_name
        self.dev = gsdevice.Camera(sensor_name)
        self.dev.connect()
        f0 = self.dev.get_image()
        print('image size = ', f0.shape[1], f0.shape[0])
        sleep(1)
        self.width = f0.shape[1] * 2
        self.height = f0.shape[0] * 2

        # prepare save dir
        self.save_dir = osp.join(EXPERIMENT_ROOT, netid, sensor_name)
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)

        super(CaptureApp, self).__init__((1024, 768), "Gelsight Capture App")

        window = ng.Window(self, "Button demo")
        window.set_position((15, 15))
        window.set_layout(ng.GroupLayout())

        ng.Label(window, "Capture commands", "sans-bold")
        b = ng.Button(window, "Capture Frame")
        self.last_frame = None
        # capture button
        def cb():
            fname = "frame_%d.ppm" % self.img_save_num
            while osp.exists(osp.join(self.save_dir, fname)):
                self.img_save_num += 1
                fname = "frame_%d.ppm" % self.img_save_num
            # captured = self.cap_obj.captureTrigger(fname)
            captured = self.last_frame
            if captured is not None:
                cv2.imwrite(osp.join(self.save_dir, fname), captured)
                self.img_save_num += 1
            print("Captured!")

        b.set_callback(cb)

        # image view
        img_window = ng.Window(self, "Current image")
        img_window.set_position((200, 15))
        img_window.set_layout(ng.GroupLayout())

        self.img_view = ng.ImageView(img_window)
        self.img_view.set_size((self.width, self.height))
        
        self.img_tex = ng.Texture(
            pixel_format=Texture.PixelFormat.RGB,
            component_format=Texture.ComponentFormat.UInt8,
            size=[self.width, self.height],
            min_interpolation_mode=Texture.InterpolationMode.Trilinear,
            mag_interpolation_mode=Texture.InterpolationMode.Nearest,
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )
        print("Texture size:", self.width, self.height)

        self.perform_layout()


    def draw(self, ctx):
        # should be in the callback

        if self.dev.while_condition:
            f1 = self.dev.get_image()
            bigframe = cv2.resize(f1, (f1.shape[1] * 2, f1.shape[0] * 2))
            self.last_frame = bigframe
            self.img_tex.upload(cv2.cvtColor(bigframe, cv2.COLOR_BGR2RGB))
        
        self.img_view.set_image(self.img_tex)
        self.img_view.set_size((self.width * 2 + 50, self.height * 2 + 50))
        # self.img_view.center()
        super(CaptureApp, self).draw(ctx)

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(CaptureApp, self).keyboard_event(key, scancode,
                                                  action, modifiers):
            return True
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True


if __name__ == "__main__":

    #import argparse
    #parser = argparse.ArgumentParser(description='Gelsight capture tool')
    #parser.add_argument('--sensor_name', type=str, help='Name of the sensor to use', choices=['GelSight Mini',  'GelSight R15', 'DIGIT'])
    #parser.add_argument('--netid', type=str, help='A folder named by your netID will be created under ~/Desktop/tactile_lab_session to save the images')
    #args = parser.parse_args()
    netid = input("NetID: ")
    sensor_name = input("Sensor name (GelSight Mini, GelSight R15, DIGIT): ")
    while sensor_name not in ["GelSight Mini", "GelSight R15", "DIGIT"]:
        sensor_name = input("Sensor name (GelSight Mini, GelSight R15, DIGIT): ")

    ng.init()
    app = CaptureApp(sensor_name, netid)
    app.draw_all()
    app.set_visible(True)
    ng.mainloop(refresh=1 / 60.0 * 1000)
    del app
    gc.collect()
    ng.shutdown()
