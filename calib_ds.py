import numpy as np

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