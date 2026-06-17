import importlib
import numpy as np
from src.scripts.binding_bins import binding
import matplotlib.pyplot as plt
from src.scripts.misc import set_font, axis_ticks

class td_deconv: 
    def __init__(self,
                 project: str = 'Example', 
                 runnr: int = None, 
                 savename : int = None, 
                 bg_bins : int = 10
                 ):
        
        self.bind_inst = binding(project, runnr, savename)
        self.project = project
        self.PP = importlib.import_module(f'src.projects.{self.project}.paths').ProjectPaths_Binding()
        self.deconv_params = importlib.import_module(f'src.projects.{self.project}.deconv_params').deconv_params
        self.setup = importlib.import_module(f'src.projects.{self.project}.setup').setup

        
        self.bg_bins = bg_bins
        self.bg_const = np.mean(self.bind_inst.spec_meas_all_dets.eval()[:, -self.bg_bins:], axis = 1)

        
        self.inc = (np.linalg.inv(self.bind_inst.rema_dip[:, :, :, ::-1, ::-1].eval()) @ 
               (self.bind_inst.spec_meas_all_dets[:, None, None, :, None].eval() - self.bg_const[:, None, None, None, None]) )[:, :, :, :, 0]
        # inc in dimensions (detectors, parities, states, bins) 


    def plot_td_deconv(self,
                       detector: str, 
                       parity: int,
                       state_idx: int, 
                       cal: list = [0, 1], 
                       color_meas: str = 'black',
                       color_inc: str = 'blue', 
                       plot_shape: tuple = (15, 10),
                       )-> None:
        
        idx_det = self.setup['detectors'].index(detector)

        set_font(25)
        fig, ax = plt.subplots(figsize=plot_shape)
        axis_ticks(ax)

        if len(cal) == 2:
            new_steps = cal[0] + cal[1] * self.bind_inst.steps

        elif len(cal) == 3:
            new_steps = cal[0] + cal[1] * self.bind_inst.steps + cal[2] * self.bind_inst.steps**2
        else:
            print('Choose linear or parabolic calibration.')
            return
        
        plt.step(new_steps, 
                 self.inc[idx_det][parity][state_idx] * np.diag(self.bind_inst.rema_dip[idx_det][parity][state_idx].eval()), 
                 color = color_inc, where = 'mid', label = 'Measured spectrum')
        
        plt.step(new_steps, 
                 self.bind_inst.spec_meas_all_dets[idx_det].eval() - self.bg_const[idx_det], 
                 color = color_meas, where = 'mid', label = 'Top-down deconvolution')
        
        
        
        plt.ylabel(f'Counts per {self.bind_inst.bin_width} keV')
        plt.xlabel(f'Energy in keV')

        plt.legend()
