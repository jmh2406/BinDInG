# config.py
from pathlib import Path

class ProjectPaths_Binding:
    def __init__(self):
        self.ROOT = Path(__file__).resolve().parent.parent.parent
        self.SRC_DIR = self.ROOT 

        '''
        Here you can give the absolute path to directories that 
        contain your measured spectra and response matrices. 
        You can adjust the complete structure to your liking by 
        changing the functions get_spec_path and get_rema_path. 

        In the example, the spectrum base directory self.SPEC_BASE_DIR contains 
        a sub-directory for each run. Each of the sub-directories then contains 
        the .txt files with the spectra for each detector. 

        In the example structure for the response matrices, is similar with 
        the different sub-directories dedicated to the corresponding simulated
        cascades, e.g. simulations for a 0^+ --> 1^+ --> 0^+ cascade
        (denoted as 1p_0p). The files containg the response matrices 
        are in the .h5 format with branch like structre. Each of the 
        top branches there correspond to a detector. 
        '''

        self.SPEC_BASE_DIR = "/d/d04-1/ag_pi/HIGS_KRF_2024/ana_70Zn/gen_JH/specs_txt_cal"
        self.REMA_BASE_DIR = "/d/d04-1/ag_pi/HIGS_KRF_2024/ana_70Zn/deconv/det_response"



    def get_spec_path(self, runnr: int, typus: str, detector: str ) -> str:    

        '''
        Here you can define the structure of your measured spectra files. 
        Adjust the function to you needs such that it returns the complete path of 
        your spectra.
        Spectra can be calibrated or uncalibrated and with an arbitrary binning.
        '''

        path_name = f'{self.SPEC_BASE_DIR}/run{runnr:04d}/run_{runnr:04d}_{typus}_{detector}.txt'
        return path_name

    def get_rema_path(self, cascade: str) -> str: 
        '''
        Here you can define the structure of your response matrix files. 
        Adjust the function to you needs such that it returns the complete path of 
        your response matrices.
        Response matrices need to be given in .h5 format for now.
        '''
        if cascade == '0deg':
            path_name = f'{self.REMA_BASE_DIR}/{cascade}/sim_1M_0deg_2_5cmoffset/resp_matrix_{cascade}_1M_0deg_2_5cmoffset.h5'
        else: 
            path_name = f'{self.REMA_BASE_DIR}/{cascade}/sim_100M/resp_matrix_{cascade}_100M.h5'

        return path_name

