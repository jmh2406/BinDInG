import h5py
from pathlib import Path
import numpy as np 
import importlib



def get_matr(project: str, cascade: str, detector: str) -> np.ndarray: 
    '''
    Function to get a response matrix as a numpy array for a given cascade 
    and a given detector from the .h5 detector response matrix files. 
    Paths and structre of the response matrix files are given in the paths.py file 
    and can be changed there.
    '''
    
    PP = importlib.import_module(f'src.projects.{project}.paths').ProjectPaths_Binding()
    rema_path = PP.get_rema_path(cascade)
    setup = importlib.import_module(f'src.projects.{project}.setup').setup

    if cascade != '0deg':
        idx_det = setup['detectors'].index(detector)
        typus = setup['types'][idx_det]

    else:
        typus = 'HPGe'
        detector = '0deg'


    p = Path(rema_path)
    with h5py.File(p, "r") as f:
        if typus == 'Clover':
            idx_h5 = setup['h5_indices'][idx_det]
            matrices = [f[f'det{i}'][() ] for i in idx_h5]
            data = np.sum(matrices, axis=0)
        elif detector == '0deg':
            idx_h5_0deg = setup['h5_indices_0deg']
            data = f[f'det{idx_h5_0deg}'][()]
        else:
            idx_h5 = setup['h5_indices'][idx_det]
            data = f[f'det{idx_h5}'][()]

        return data


def recalibrate_and_rebin_to_20000(spec, a, b, c = 0):
    '''
    Recalibrate a given spectrum spec with parameters a, b (and c) 
    and shape it to 1 keV bins with a maximum energy of 20000 keV

    '''
    n = 20000
    E_old = np.arange(n)
    E_new = a * E_old + b + c * E_old ** 2
    
    left = np.floor(E_new).astype(int)
    right = left + 1
    frac = E_new - left
    
    rebinned = np.zeros(n)
    mask_l = (left >= 0) & (left < n)
    mask_r = (right >= 0) & (right < n)
    
    idx_l = left[mask_l].astype(np.int64)
    val_l = (spec[mask_l] * (1.0 - frac[mask_l])).astype(np.float64)
    
    idx_r = right[mask_r].astype(np.int64)
    val_r = (spec[mask_r] * frac[mask_r]).astype(np.float64)
    
    np.add.at(rebinned, idx_l, val_l)
    np.add.at(rebinned, idx_r, val_r)
    return rebinned




def bin_matrix(M: np.ndarray, bin_x: int, bin_y: int)-> np.ndarray:
    '''
    Rebin a matrix to new bin widths bin_x and bin_y. Note that the
    new bin width is given relative to the old one. I.e. if you 
    want to rebin a matrix with a bin width of (1 keV, 1 keV) to a new 
    bin width (10 keV, 10 keV) you need to give bin widths bin_x = 10 and
    bin_y = 10. The old binning needs 
    to be divisible by the new binnnig.
    '''
    n_rows, n_cols = M.shape

    n_rows_new = (n_rows // bin_x) * bin_x
    n_cols_new = (n_cols // bin_y) * bin_y
    M_crop = M[:n_rows_new, :n_cols_new]

    M_reshaped = M_crop.reshape(n_rows_new // bin_x, bin_x,
                                n_cols_new // bin_y, bin_y)

    M_binned = M_reshaped.sum(axis=(1, 3))

    return M_binned



def rebin_array(arr, new_bins):
    '''
    Rebin a given array (or spectrum) by an integer amount. 
    Note that the new binning given relative to the old one, 
    similar to the bin_matrix function. The old binning needs 
    to be divisible by the new binning. The argument new_bins
    is the number of bins in the new spectrum.
    '''
    n = len(arr)
    if isinstance(new_bins, int):
        factor = n // new_bins
        reshaped = arr.reshape(new_bins, factor)
        return reshaped.sum(axis=1)

    else:
        raise ValueError('Choose integer for the new_bins')
    



def get_respmatrix(project: str, cascade: str, detector: str, bin_width : int) -> tuple:
    '''
    Quick function to get a response matrix with a certain bin width as an np.ndarray. For 
    more information take a look at the get_matr and bin_matrix functions.
    '''
    matr = get_matr(project, cascade, detector)
    bin_matr = bin_matrix(matr, bin_width, bin_width)
    return bin_matr





def get_meas_spec(project: str, 
                  run: int, 
                  detector: str,
                  cal: list, 
                  bin_width : int, 
                  max_bin: int = 15000):
    '''
    Function to get the measured spectra as an np.ndarray. Paths and structure to the measured spectra 
    can be changed in the paths.py file. 
    '''

    PP = importlib.import_module(f'src.projects.{project}.paths').ProjectPaths_Binding()
    setup = importlib.import_module(f'src.projects.{project}.setup').setup

    bin_nr = max_bin // bin_width
    run04d = f'{run:04d}' if type(run) == int else run


    if detector != 'ZeroDegree':
        idx_det = setup['detectors'].index(detector)
        typus = setup['types'][idx_det]


        # run04d = f'{run:04d}' if type(run) == int else run

        if setup['types'][idx_det] == 'Clover':
            spec_raw = np.sum([np.loadtxt(PP.get_spec_path(run, typus, detector + f'E{x}')) for x in range(1, 5)], axis = 0)
        else:
            spec_raw = np.loadtxt(PP.get_spec_path(run, typus, detector))

    else:
        typus = 'HPGe'
        spec_raw = np.loadtxt(PP.get_spec_path(run, typus, detector))


    if len(cal) == 2:
        spec = recalibrate_and_rebin_to_20000(spec_raw,  cal[1], cal[0])
    else: 
        spec = recalibrate_and_rebin_to_20000(spec_raw,  cal[1], cal[0], cal[2])

    spec_reb = rebin_array(spec[0:max_bin], bin_nr)

    return spec_reb