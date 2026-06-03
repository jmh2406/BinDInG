import h5py
from pathlib import Path
import sys
import math
import numpy as np 
from deconv.bayes_inf.deconv_params import deconv_params


ana_dir = "/d/d04-1/ag_pi/HIGS_KRF_2024/ana_70Zn"


sys.path.insert(1, f'{ana_dir}/common')
from geant_dets import geant_dets


deconv_dir = f"{ana_dir}/deconv"


def plot_matr(cascade: str, simulname: str, detector: str) -> None: 

    p = Path(f"{deconv_dir}/det_response/{cascade}/sim_{simulname}/resp_matrix_{cascade}_{simulname}.h5")
    with h5py.File(p, "r") as f:
        if str(detector) == '5':
            matrices = [f[f'det{i}'][() ] for i in range(0, 4)]
            data = np.mean(matrices, axis=0)
        elif str(detector) == '7':
            matrices = [f[f'det{i}'][() ] for i in range(4, 8)]
            data = np.mean(matrices, axis=0)
        elif str(detector) == 'B1':
            matrices = [f[f'det{i}'][() ] for i in range(8, 12)]
            data = np.mean(matrices, axis=0)
        elif str(detector) == 'B2':
            matrices = [f[f'det{i}'][() ] for i in range(12, 16)]
            data = np.mean(matrices, axis=0)
        elif str(detector) == 'B3':
            matrices = [f[f'det{i}'][() ] for i in range(16, 20)]
            data = np.mean(matrices, axis=0)
        elif str(detector) == 'B4':
            matrices = [f[f'det{i}'][() ] for i in range(20, 24)]
            data = np.mean(matrices, axis=0) 
        elif str(detector) == 'B5':
            matrices = [f[f'det{i}'][() ] for i in range(24, 28)]
            data = np.mean(matrices, axis=0)      
 
        else:
            det_idx = [idx for idx, i in enumerate(geant_dets) if i == detector][0]
            data = f[f'det{det_idx}'][()]
    return data

# def recalibrate_and_rebin_to_20000(spec, a, b):
#     """
#     Wendet E' = a*E + b an und verteilt die Counts per linearer Interpolation
#     auf die beiden nächsten integer-keV-Bins.
#     Ergebnis: Spektrum bleibt exakt 20000 Bins breit.
#     """

#     n = 20000
#     E_old = np.arange(n)
#     E_new = a * E_old + b

#     rebinned = np.zeros(n)

#     for i in range(n):
#         e = E_new[i]

#         # linker und rechter integer-Bin
#         left = int(np.floor(e))
#         right = left + 1

#         # Abstand im Intervall [0,1)
#         frac = e - left

#         # Verteilung auf Nachbarbins
#         if 0 <= left < n:
#             rebinned[left] += spec[i] * (1 - frac)
#         if 0 <= right < n:
#             rebinned[right] += spec[i] * frac

#     return rebinned
def recalibrate_and_rebin_to_20000(spec, a, b, c = 0):
    n = 20000
    E_old = np.arange(n)
    E_new = a * E_old + b + c * E_old ** 2
    
    # Vektorisierte Berechnung von left, right und frac
    left = np.floor(E_new).astype(int)
    right = left + 1
    frac = E_new - left
    
    rebinned = np.zeros(n)
    # Masken für den gültigen Bereich
    mask_l = (left >= 0) & (left < n)
    mask_r = (right >= 0) & (right < n)
    
    # Indizes und Werte VORHER extrahieren und bereinigen
    idx_l = left[mask_l].astype(np.int64)
    val_l = (spec[mask_l] * (1.0 - frac[mask_l])).astype(np.float64)
    
    idx_r = right[mask_r].astype(np.int64)
    val_r = (spec[mask_r] * frac[mask_r]).astype(np.float64)
    
    # Jetzt sind idx_l und val_l garantiert einfache 1D-Vektoren
    np.add.at(rebinned, idx_l, val_l)
    np.add.at(rebinned, idx_r, val_r)
    return rebinned

def bin_matrix(M, bin_x, bin_y, method="sum"):
    n_rows, n_cols = M.shape

    # Größe anpassen (abschneiden, wenn nicht teilbar)
    n_rows_new = (n_rows // bin_x) * bin_x
    n_cols_new = (n_cols // bin_y) * bin_y
    M_crop = M[:n_rows_new, :n_cols_new]

    # In Blöcke umformen
    M_reshaped = M_crop.reshape(n_rows_new // bin_x, bin_x,
                                n_cols_new // bin_y, bin_y)

    if method == "sum":
        M_binned = M_reshaped.sum(axis=(1, 3))
    elif method == "mean":
        M_binned = M_reshaped.mean(axis=(1, 3))
    else:
        raise ValueError("method must be 'sum' or 'mean'")

    return M_binned

def rebin_array(arr, new_bins, method="sum"):
    n = len(arr)

    # Case 1: Equal-size rebinning (integer factor)
    if isinstance(new_bins, int):
        if n % new_bins != 0:
            raise ValueError(f"Array length {n} not divisible by new_bins {new_bins}")
        factor = n // new_bins
        reshaped = arr.reshape(new_bins, factor)
        if method == "sum":
            return reshaped.sum(axis=1)
        elif method == "mean":
            return reshaped.mean(axis=1)
        else:
            raise ValueError("method must be 'sum' or 'mean'")

    # Case 2: Custom bin edges (list/array of indices)
    else:
        new_bins = np.asarray(new_bins)
        if new_bins[0] != 0 or new_bins[-1] != n:
            raise ValueError("Custom bins must start at 0 and end at len(arr)")
        rebinned = []
        for i in range(len(new_bins) - 1):
            start, stop = new_bins[i], new_bins[i + 1]
            segment = arr[start:stop]
            if method == "sum":
                rebinned.append(segment.sum())
            elif method == "mean":
                rebinned.append(segment.mean())
            else:
                raise ValueError("method must be 'sum' or 'mean'")
        return np.array(rebinned)

def get_respmatrix(matrix: str, cascade: str, detector: str, bin_width : int) -> tuple:
    matr = plot_matr(cascade, matrix, detector)
    bin_matr = bin_matrix(matr, bin_width, bin_width)

    return bin_matr

def get_meas_spec(run: int, detector: str, bg_scale: float, cal: list, bin_width : int):
    max_bin = 10000
    bin_nr = max_bin // bin_width

    run04d = f'{run:04d}' if type(run) == int else run

    if detector[0] == 'L' :
        typus = 'LaBr'
        spec_raw = np.loadtxt(f'{ana_dir}/gen_JH/specs_txt_cal/run{run04d}/run_{run04d}_{typus}_{detector}.txt') - np.loadtxt(f'{deconv_dir}/bayes_inf/bg_ones.txt') * bg_scale
    elif detector == 'ZeroDegree':
        typus = 'HPGe'
        spec_raw = np.loadtxt(f'{ana_dir}/gen_JH/specs_txt_cal/run{run04d}/run_{run04d}_{typus}_{detector}.txt') - np.loadtxt(f'{deconv_dir}/bayes_inf/bg_ones.txt') * bg_scale
    else:
        typus = 'Clover'
        spec_raw = np.sum([np.loadtxt(f'{ana_dir}/gen_JH/specs_txt_cal/run{run04d}/run_{run04d}_{typus}_{detector}E{x}.txt') - np.loadtxt(f'{deconv_dir}/bayes_inf/bg_ones.txt') * bg_scale for x in range(1, 5)], axis = 0)


    if len(cal) == 2:
        spec = recalibrate_and_rebin_to_20000(spec_raw,  cal[1], cal[0])# -20000 2.076# 1.13, -800, -3500 0.753
    else: 
        spec = recalibrate_and_rebin_to_20000(spec_raw,  cal[1], cal[0], cal[2])# -20000 2.076# 1.13, -800, -3500 0.753

    spec_reb = rebin_array(spec[0:10000], bin_nr)

    return spec_reb