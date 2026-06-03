#!/usr/bin/env python

import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
import h5py
from pathlib import Path
import sys
import numpy as np 
from src.scripts.get_matrix_and_specs import get_respmatrix, get_meas_spec
import matplotlib.pyplot as plt
from src.projects.deconv_params import deconv_params


ana_dir = '/d/d04-1/ag_pi/HIGS_KRF_2024/ana_70Zn'

from common.misc import UArrayWrapper, axis_ticks

def get_cached_data(name, low, high, bin_width, compute_func):
    cache_dir = Path(f"{ana_dir}/deconv/bayes_inf/cache_files")
    cache_dir.mkdir(exist_ok=True)
    
    file_path = cache_dir / f"{name}_{low}_{high}_bw{bin_width}.npy"
    
    if file_path.exists():
        return np.load(file_path)
    
    print(f"Calculate {name} initial (Cache: {low}-{high} keV, BW: {bin_width})...")
    data = compute_func()
    np.save(file_path, data)
    return data


def get_rema_100M(casc: str, det: str, mask: np.ndarray, bin_width: int)-> np.ndarray: 
    return (get_respmatrix('100M', casc, det, bin_width)[np.ix_(mask, mask)])[::-1,::-1]

def pytensor_interp(x_new, x_old, y_old):
    return pt.extra_ops.interp(x_new, x_old, y_old)


class binding:
    def __init__(self, runnr: int,  savename: str, bin_width: int = 10, branchings = True, 
                 br12p2 = False,
                 O_cont = False,
                 ng_cont = False):
        
        self.O_cont_bool = O_cont
        self.ng_cont_bool = ng_cont
        self.branch_bool = branchings
        self.br1_2p2_bool = br12p2
        try: 
            self.runnr = int(runnr)
        except:
            self.runnr = str(runnr)
        self.savename = savename
        self.ebeam = deconv_params[runnr]['beam']
        self.bin_width = bin_width
        self.bin_centers = np.arange(bin_width, 10000 + bin_width, bin_width)
        self.low = deconv_params[runnr]['low']
        self.high = deconv_params[runnr]['high']
        self.mask_energy = (self.bin_centers >= self.low) & (self.bin_centers <= self.high) 
        self.td_cut_low = deconv_params[runnr]['cut_dets_low']# [L1, L3, 5, 7]
        self.td_cut_high = deconv_params[runnr]['cut_dets_high']

        # Rema for 0 deg detector for determining beam shape


        self.rema_beam = pytensor.shared(get_cached_data("rema_beam", self.low, self.high, bin_width, lambda: get_respmatrix('1M_0deg_2_5cmoffset', '0deg', '0deg', bin_width)[np.ix_(self.mask_energy, self.mask_energy)]))
        
        
        def compute_iso_rema():
            rema_iso_L1 = (get_respmatrix('100M', '1.5m_0.5m', 'L1', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1] # TODO change to correct beam response matrix
            rema_iso_L3 = (get_respmatrix('100M', '1.5m_0.5m', 'L3', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1]
            rema_iso_5 = (get_respmatrix('100M', '1.5m_0.5m', '5', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1]
            rema_iso_7 = (get_respmatrix('100M', '1.5m_0.5m', '7', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1]

            rema_iso = np.array([rema_iso_L1, rema_iso_L3, rema_iso_5, rema_iso_7])
            return rema_iso
        

        def compute_O_rema():
            rema_O_L1 = (get_respmatrix('100M', '2p_0p', 'L1', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1] # TODO change to correct beam response matrix
            rema_O_L3 = (get_respmatrix('100M', '2p_0p', 'L3', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1]
            rema_O_5 = (get_respmatrix('100M', '2p_0p', '5', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1]
            rema_O_7 = (get_respmatrix('100M', '2p_0p', '7', bin_width)[np.ix_(self.mask_energy, self.mask_energy)])[::-1, ::-1]

            rema_O = np.array([rema_O_L1, rema_O_L3, rema_O_5, rema_O_7])
            return rema_O
        


        def compute_all_rema():
            # rema for non-isotropic radiation 
            rema_L1_1p0p, rema_L1_1p2p = get_rema_100M('1p_0p', 'L1', self.mask_energy, bin_width), get_rema_100M('1p_2p', 'L1', self.mask_energy, bin_width) 
            rema_L1_1m0p, rema_L1_1m2p = get_rema_100M('1m_0p', 'L1', self.mask_energy, bin_width), get_rema_100M('1m_2p', 'L1', self.mask_energy, bin_width) 
            rema_L3_1p0p, rema_L3_1p2p = get_rema_100M('1p_0p', 'L3', self.mask_energy, bin_width), get_rema_100M('1p_2p', 'L3', self.mask_energy, bin_width) 
            rema_L3_1m0p, rema_L3_1m2p = get_rema_100M('1m_0p', 'L3', self.mask_energy, bin_width), get_rema_100M('1m_2p', 'L3', self.mask_energy, bin_width) 
            rema_5_1p0p, rema_5_1p2p = get_rema_100M('1p_0p', '5', self.mask_energy, bin_width), get_rema_100M('1p_2p', '5', self.mask_energy, bin_width) 
            rema_5_1m0p, rema_5_1m2p = get_rema_100M('1m_0p', '5', self.mask_energy, bin_width), get_rema_100M('1m_2p', '5', self.mask_energy, bin_width) 
            rema_7_1p0p, rema_7_1p2p = get_rema_100M('1p_0p', '7', self.mask_energy, bin_width), get_rema_100M('1p_2p', '7', self.mask_energy, bin_width) 
            rema_7_1m0p, rema_7_1m2p = get_rema_100M('1m_0p', '7', self.mask_energy, bin_width), get_rema_100M('1m_2p', '7', self.mask_energy, bin_width) 


            rema_all_dets = np.array([[[rema_L1_1p0p, 
                                                rema_L1_1p2p,
                                                rema_L1_1p0p,
                                                ], 
                                                [rema_L1_1m0p,
                                                rema_L1_1m2p,
                                                rema_L1_1m0p,
                                                ]
                                                ], # L1
                                                [[rema_L3_1p0p, 
                                                rema_L3_1p2p,
                                                rema_L3_1p0p,
                                                ], 
                                                [rema_L3_1m0p,
                                                rema_L3_1m2p,
                                                rema_L3_1m0p,
                                                ] # L3
                                                ],
                                                [[rema_5_1p0p, 
                                                rema_5_1p2p,
                                                rema_5_1p0p,
                                                ], 
                                                [rema_5_1m0p,
                                                rema_5_1m2p,
                                                rema_5_1m0p,
                                                ] # 5
                                                ],
                                                [[rema_7_1p0p, 
                                                rema_7_1p2p,
                                                rema_7_1p0p, 
                                                ], 
                                                [rema_7_1m0p,
                                                rema_7_1m2p,
                                                rema_7_1m0p, 
                                                ] # 7
                                                ]])
            
            return rema_all_dets
        
        self.rema_iso = pytensor.shared(get_cached_data("rema_iso", self.low, self.high, bin_width, lambda: compute_iso_rema()))
        self.rema_all_dets = pytensor.shared(get_cached_data("rema_all_dets", self.low, self.high, bin_width, lambda: compute_all_rema()))
        self.rema_O = pytensor.shared(get_cached_data("rema_iso", self.low, self.high, bin_width, lambda: compute_O_rema()))


        self.spec_meas_all_dets = pytensor.shared(np.array([
            get_meas_spec(self.runnr, 'L1', 0, deconv_params[runnr]['cal_L1'], bin_width)[self.mask_energy],
            get_meas_spec(self.runnr, 'L3', 0, deconv_params[runnr]['cal_L3'], bin_width)[self.mask_energy], 
            get_meas_spec(self.runnr, '5', 0, [0, 1], bin_width)[self.mask_energy],
            get_meas_spec(self.runnr, '7', 0, [0, 1], bin_width)[self.mask_energy],
        ]))


        self.spec_meas_beam = get_meas_spec(deconv_params[self.runnr]['0deg_nr'], 'ZeroDegree', 0, deconv_params[self.runnr]['cal_0deg'], bin_width)[self.mask_energy]
        
        num_bins = int((self.high - self.low) / self.bin_width) + 1
        self.steps = np.linspace(self.low, self.high, num_bins)

        self.model = self.set_model_bins()
    def set_model_bins(self):

        with pm.Model() as model:
            # Priors for beam profile 
            E_beam = pm.Normal('E_beam', self.ebeam, sigma = 20, shape = ())
            Width_beam = pm.TruncatedNormal('Width_beam', mu = self.ebeam * (0.03 / 2.355), sigma = self.ebeam * (0.02 / 2.355), lower = self.ebeam * (0.02 / 2.355), upper = self.ebeam * (0.05 / 2.355) ,shape = ())
            Tail_beam = pm.Uniform("Tail_beam", 0.1, 10, shape=())
            Vol_beam = pm.Exponential("Vol_beam", 1 / 1e8)
            bg_beam = pm.TruncatedNormal(
                'bg_beam', 
                mu=250,          
                sigma=50,      
                lower=200,       
                upper=500,      
                shape=()
            )
            Beam_inc = pm.Deterministic('Beam', theuerkauf(self.bin_centers[self.mask_energy], E_beam, Vol_beam, Width_beam, Tail_beam))
            Beam_fold = pm.Deterministic('Beam_fold', (Beam_inc @ self.rema_beam)+ bg_beam)
            

            # PRIORS 
            # The ideas behind this deconvolution approach is to describe each bin height as a sum of different cascades (i.e. 1p0p, 1m0p, 1m2p, 1p2p, 1p0pe, 1m0pe). 
            # The priors for the each of the distributions are determined using a top-down deconvolutions. 
            # First, the priors for 1p0p and 1m0p are determined as prior_1m0p = rema_1m0p x spec_L3 and prior_1p0p = rema_1p0p x spec_meas_L1

            # Constant background for detector array 
            # bg_const = pm.TruncatedNormal(
            #     'bg_const', 
            #     mu = np.mean(self.spec_meas_all_dets[:, -5:].eval(), axis=1),
            #     lower = np.mean(self.spec_meas_all_dets[:, -5:].eval(), axis=1) - np.mean(self.spec_meas_all_dets[:, -5:].eval(), axis=1),
            #     upper = np.mean(self.spec_meas_all_dets[:, -5:].eval(), axis=1) + np.std(self.spec_meas_all_dets[:, -5:].eval(), axis=1),
            #     shape=(4,)
            # )
            bg_const = pm.Deterministic(
                'bg_const', 
                pt.mean(self.spec_meas_all_dets[:, -10:], axis=1),
            )

            if self.ng_cont_bool:
                # Intoduce ng lines as contaminations in the spectrum. Assume isotropic radiation and a gaussian with a width corresponding to the detector resolution (different for Clover and LaBr)
                E_ng = pm.TruncatedNormal('E_ng', [7631, 7645, 7721], lower = [7630, 7644, 7720], upper = [ 7632, 7646, 7722])
                I_ng = pm.Exponential('I_ng', 1 / 50e3, shape=(E_ng.shape[0],)) 
                ng_width_labr = pm.Uniform('ng_width', 5, 50, shape = (2, ))
                ng_width_clover = pm.Uniform('ng_width_clover', 5, 15, shape = (2, ))
                sqrt_2pi = np.sqrt(2 * np.pi)
                # LaBr dets
                ng_inc_labr = pm.Deterministic('ng_inc', 
                    pt.sum(
                        (I_ng[None, :, None] / (sqrt_2pi * ng_width_labr[:, None, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None, None, :] - E_ng[None, :, None]) / ng_width_labr[:, None, None]) ** 2),
                        axis=1
                    )
                )

                # Clover dets
                ng_inc_clover = pm.Deterministic('ng_inc_clover', 
                    pt.sum(
                        (I_ng[None, :, None] / (sqrt_2pi * ng_width_clover[:, None, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None, None, :] - E_ng[None, :, None]) / ng_width_clover[:, None, None]) ** 2),
                        axis=1
                    )
                )# dets, bins
                ng_fold_labr = pm.Deterministic('ng_fold_labr', self.rema_iso[:2] @ ng_inc_labr[:, :, None])[:, :, 0]  # shape (det, bins) x (det, bins, bins) -> (det, ng, bins)
                ng_fold_clover = pm.Deterministic('ng_fold_clover', self.rema_iso[2:] @ ng_inc_clover[:,:, None])[:, :, 0] # shape (det, binss) x (det, bins, bins) -> (det, ng, bins)


            if self.O_cont_bool:
                # Intoduce 16O as contaminations in the spectrum. Assume isotropic radiation and a gaussian with a width corresponding to the detector resolution (different for Clover and LaBr)
                E_O = pm.TruncatedNormal('E_O', [6910], lower = [6900], upper = [6950])
                E_O_clover = pm.TruncatedNormal('E_O_clover', [6910], lower = [6900], upper = [6930])

                I_O = pm.Exponential('I_O', 1 / 100e3, shape=(E_O.shape[0],)) 
                O_width_labr = pm.Uniform('O_width', 5, 50, shape = (2, ))
                O_width_clover = pm.Uniform('O_width_clover', 5, 15, shape = (2, ))
                sqrt_2pi = np.sqrt(2 * np.pi)
                # LaBr dets
                O_inc_labr = pm.Deterministic('O_inc', 
                    pt.sum(
                        (I_O[None, :, None] / (sqrt_2pi * O_width_labr[:, None, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None, None, :] - E_O[None, :, None]) / O_width_labr[:, None, None]) ** 2),
                        axis=1
                    )
                )

                # Clover dets
                O_inc_clover = pm.Deterministic('O_inc_clover', 
                    pt.sum(
                        (I_O[None, :, None] / (sqrt_2pi * O_width_clover[:, None, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None, None, :] - E_O_clover[None, :, None]) / O_width_clover[:, None, None]) ** 2),
                        axis=1
                    )
                )# dets, bins
                O_fold_labr = pm.Deterministic('O_fold_labr', self.rema_O[:2] @ O_inc_labr[:, :, None])[:, :, 0]  # shape (det, bins) x (det, bins, bins) -> (det, ng, bins)
                O_fold_clover = pm.Deterministic('O_fold_clover', self.rema_O[2:] @ O_inc_clover[:,:, None])[:, :, 0] # shape (det, binss) x (det, bins, bins) -> (det, ng, bins)




            # Set a fit parameter that describes the E1 to M1 ratio. Its called alpha and describes the total spectrum S_tot like S_tot = (1- alpha) * S_1m0p + alpha S_1p0p
            alpha = pm.Uniform('alpha', 0, 1 , shape = ())

            # 1p0p 
            prior_1p0p_raw = pt.linalg.inv(self.rema_all_dets[0][0][0]) @ (self.spec_meas_all_dets[0] - bg_const[0])# bg_const[0]) # dimensions are (detector, parity, final_state, bins, bins) and (detector, bins), respectively 
            if self.ng_cont_bool: # subtract contributions of ng lines as contaminations in the spectrum
                if self.O_cont_bool:
                    prior_1p0p_nong = prior_1p0p_raw - ng_inc_labr[0] - O_inc_labr[0]
                else:
                    prior_1p0p_nong = prior_1p0p_raw - ng_inc_labr[0] 
            else:
                if self.O_cont_bool:
                    prior_1p0p_nong = prior_1p0p_raw  - O_inc_labr[0]
                else:
                    prior_1p0p_nong = prior_1p0p_raw 
            prior_1p0p = pt.where((self.ebeam * self.td_cut_high[0] >= self.bin_centers[self.mask_energy]) 
                                  & (self.bin_centers[self.mask_energy] >= self.ebeam * self.td_cut_low[0])
                                  , prior_1p0p_nong, 0.0) * (alpha) 
            


            # 1m0p 
            prior_1m0p_raw = pt.linalg.inv(self.rema_all_dets[1][1][0]) @ (self.spec_meas_all_dets[1] - bg_const[1])# bg_const[1]) # dimensions are (detector, parity, final_state, bins, bins) and (detector, bins), respectively 
            if self.ng_cont_bool: # subtract contributions of ng lines as contaminations in the spectrum
                if self.O_cont_bool:
                    prior_1m0p_nong = prior_1m0p_raw - ng_inc_labr[1] - O_inc_labr[1]
                else:
                    prior_1m0p_nong = prior_1m0p_raw - ng_inc_labr[1] 

            else:
                if self.O_cont_bool:
                    prior_1m0p_nong = prior_1m0p_raw  - O_inc_labr[1]
                else:
                    prior_1m0p_nong = prior_1m0p_raw 
            prior_1m0p = pt.where((self.ebeam * self.td_cut_high[1] >= self.bin_centers[self.mask_energy])
                                  & (self.bin_centers[self.mask_energy] >= self.ebeam * self.td_cut_low[1]) 
                                  , prior_1m0p_nong, 0.0) * (1 - alpha)

        


            # 1p0p Clover 
            prior_1p0p_raw_clover = pt.linalg.inv(self.rema_all_dets[2][0][0]) @ (self.spec_meas_all_dets[2] - bg_const[2]) # dimensions are (detector, parity, final_state, bins, bins) and (detector, bins), respectively 
            if self.ng_cont_bool: # subtract contributions of ng lines as contaminations in the spectrum
                if self.O_cont_bool:
                    prior_1p0p_nong_clover = prior_1p0p_raw_clover - ng_inc_clover[0] - O_inc_clover[0]
                else:
                    prior_1p0p_nong_clover = prior_1p0p_raw_clover - ng_inc_clover[0] 

            else:
                if self.O_cont_bool:
                    prior_1p0p_nong_clover = prior_1p0p_raw_clover  - O_inc_clover[0]
                else:
                    prior_1p0p_nong_clover = prior_1p0p_raw_clover 
            prior_1p0p_clover = pt.where((self.bin_centers[self.mask_energy] >= self.ebeam * self.td_cut_low[2]) 
                                        & (self.bin_centers[self.mask_energy] <= self.ebeam * self.td_cut_high[2])
                                        , prior_1p0p_nong_clover, 0.0) * (alpha)

            # 1m0p Clover 
            prior_1m0p_raw_clover = pt.linalg.inv(self.rema_all_dets[3][1][0]) @ (self.spec_meas_all_dets[3] - bg_const[3]) # dimensions are (detector, parity, final_state, bins, bins) and (detector, bins), respectively 
            if self.ng_cont_bool: # subtract contributions of ng lines as contaminations in the spectrum
                if self.O_cont_bool:
                    prior_1m0p_nong_clover = prior_1m0p_raw_clover - ng_inc_clover[1] - O_inc_clover[1]
                else:
                    prior_1m0p_nong_clover = prior_1m0p_raw_clover - ng_inc_clover[1] 
            else:
                if self.O_cont_bool:
                    prior_1m0p_nong_clover = prior_1m0p_raw_clover  - O_inc_clover[1]
                else:
                    prior_1m0p_nong_clover = prior_1m0p_raw_clover 

            prior_1m0p_clover = pt.where((self.bin_centers[self.mask_energy] >= self.ebeam * self.td_cut_low[3]) 
                                        & (self.bin_centers[self.mask_energy] <= self.ebeam * self.td_cut_high[3])
                                         , prior_1m0p_nong_clover, 0.0) * (1 - alpha)
      


        
            # For the inelastic channels, two possibilities emerge. Either use the part of the measured spectrum for a guess, or shift the gs-decay curve by the final state energy
            # Shift gs decay shape to lower energies as guess with branching ratio as scaling factors
            if self.branch_bool:
                shift_bins_2p, shift_bins_0pe = int(round(884/10)), int(round(1070/10)) 

                nr_states = 4 if self.br1_2p2_bool else 3
                prior_empty = pt.zeros_like(prior_1p0p)

                # 1p2p
                br_1p2p = pm.Exponential('br_1p2p', 1 / 0.5)
                shifted_1p2p = prior_1p0p[shift_bins_2p:] * br_1p2p
                shifted_1p2p_clover = prior_1p0p_clover[shift_bins_2p:] * br_1p2p
                prior_1p2p = pt.set_subtensor(prior_empty[:shifted_1p2p.shape[0]], shifted_1p2p)
                prior_1p2p_clover = pt.set_subtensor(prior_empty[:shifted_1p2p_clover.shape[0]], shifted_1p2p_clover)


                # 1m2p
                br_1m2p = pm.Exponential('br_1m2p', 1 / 0.5)
                shifted_1m2p = prior_1m0p[shift_bins_2p:] * br_1m2p
                shifted_1m2p_clover = prior_1m0p_clover[shift_bins_2p:] * br_1m2p
                prior_1m2p = pt.set_subtensor(prior_empty[:shifted_1m2p.shape[0]], shifted_1m2p)
                prior_1m2p_clover = pt.set_subtensor(prior_empty[:shifted_1m2p_clover.shape[0]], shifted_1m2p_clover)



                # 1p0pe 
                br_1p0pe = pm.Exponential('br_1p0pe', 1 / 0.5)
                shifted_1p0pe = prior_1p0p[shift_bins_0pe:] * br_1p0pe
                shifted_1p0pe_clover = prior_1p0p_clover[shift_bins_0pe:] * br_1p0pe
                prior_1p0pe = pt.set_subtensor(prior_empty[:shifted_1p0pe.shape[0]], shifted_1p0pe)
                prior_1p0pe_clover = pt.set_subtensor(prior_empty[:shifted_1p0pe_clover.shape[0]], shifted_1p0pe_clover)

                # 1m0pe 
                br_1m0pe = pm.Exponential('br_1m0pe', 1 / 0.5)
                shifted_1m0pe = prior_1m0p[shift_bins_0pe:] * br_1m0pe
                shifted_1m0pe_clover = prior_1m0p_clover[shift_bins_0pe:] * br_1m0pe
                prior_1m0pe = pt.set_subtensor(prior_empty[:shifted_1m0pe.shape[0]], shifted_1m0pe)
                prior_1m0pe_clover = pt.set_subtensor(prior_empty[:shifted_1m0pe_clover.shape[0]], shifted_1m0pe_clover)

                p1_states = pt.stack([prior_1p0p, prior_1p2p, prior_1p0pe])
                m1_states = pt.stack([prior_1m0p, prior_1m2p, prior_1m0pe])
                # also for clovers
                p1_states_clover = pt.stack([prior_1p0p_clover, prior_1p2p_clover, prior_1p0pe_clover])
                m1_states_clover = pt.stack([prior_1m0p_clover, prior_1m2p_clover, prior_1m0pe_clover])



            else: 
                nr_states = 1
                p1_states = pt.stack([prior_1p0p])
                m1_states = pt.stack([prior_1m0p])
                # also for clovers
                p1_states_clover = pt.stack([prior_1p0p_clover])
                m1_states_clover = pt.stack([prior_1m0p_clover])


            priors_nrf = pm.Deterministic('nrf_inc',
                pt.stack([p1_states, m1_states]))
            priors_nrf_clover = pm.Deterministic('nrf_inc_clover',
                pt.stack([p1_states_clover, m1_states_clover]))



            
            nrf_inc_free = pm.Exponential(
                'nrf_inc_free', 
                lam = 1.0 / pt.clip(priors_nrf, 1e-6, np.inf), 
                shape = (2, nr_states, len(self.bin_centers[self.mask_energy]))
            )

            nrf_inc_free_clover = pm.Exponential(
                'nrf_inc_free_clover', 
                lam = 1.0 / pt.clip(priors_nrf_clover, 1e-6, np.inf), 
                shape = (2, nr_states, len(self.bin_centers[self.mask_energy]))
            )

            # Fold all NRF components. 
            nrf_fold = pm.Deterministic('nrf_fold', self.rema_all_dets[:2, :, :nr_states] @ nrf_inc_free[None, :, :, :, None])[:, :, :, :, 0] # shapes: (dets, parites, states, bins, bins) x (parities, states, bins, dummy)
            nrf_fold_sum = pm.Deterministic('nrf_fold_sum', pt.sum(nrf_fold, axis=(1,2))) # shapes (dets, bins)

            
            # Fold all NRF components for clovers. 
            nrf_fold_clover = pm.Deterministic('nrf_fold_clover', self.rema_all_dets[2:, :, :nr_states] @ nrf_inc_free_clover[None, :, :, :, None])[:, :, :, :, 0] # shapes: (dets, parites, states, bins, bins) x (parities, states, bins, dummy)
            nrf_fold_sum_clover = pm.Deterministic('nrf_fold_sum_clover', pt.sum(nrf_fold_clover, axis=(1,2))) # shapes (dets, bins)




            # Prior distribution for atomic background, modeled as an exponential function with slope and scale parameters. 
            atomic_slope_labr = pm.Uniform(
                "Atomic_slope_labr", 0.3e-3, 10e-3, shape=(2, )
            )   
            atomic_scale_labr = pm.Uniform("Atomic_scale_labr", 4, 32, shape=(2, ))
            atomic_backg_labr_inc = pm.Deterministic(
                "Atomic_backg_labr_inc", pt.clip(
                    pt.exp(atomic_scale_labr[:, None] - atomic_slope_labr[:, None] * self.bin_centers[self.mask_energy])
                    - pt.exp(atomic_scale_labr[:, None] - atomic_slope_labr[:, None] * E_beam), 0, np.inf
                )
            )
            atomic_backg_labr_fold = pm.Deterministic("Atomic_backg_labr_fold", self.rema_iso[2:] @ atomic_backg_labr_inc[:, :, None])[:, :, 0]   


            # atomic slope for clover detectors 
            atomic_slope_clover = pm.Uniform(
                "Atomic_slope_clover", 0.3e-3, 10e-3, shape=(2, )
            )   
            atomic_scale_clover = pm.Uniform("Atomic_scale_clover", 4, 32, shape=(2, ))
            atomic_backg_clover_inc = pm.Deterministic(
                "Atomic_backg_clover_inc", pt.clip(
                    pt.exp(atomic_scale_clover[:, None] - atomic_slope_clover[:, None] * self.bin_centers[self.mask_energy])
                    - pt.exp(atomic_scale_clover[:, None] - atomic_slope_clover[:, None] * E_beam), 0, np.inf
                )
            )
            atomic_backg_clover_fold = pm.Deterministic("Atomic_backg_clover_fold", self.rema_iso[:2] @ atomic_backg_clover_inc[:, :, None])[:, :, 0]   



            scaling = pm.Deterministic('scale_tot', pt.ones((2, )))# pm.Exponential('scale_tot', 1 / 1, shape = (2,))
            scaling_clover = pm.Deterministic('scale_clover', pt.ones((2,)))# pm.Exponential('scale_clover', 1 / 1, shape = (2,))

            
            if self.ng_cont_bool: # subtract contributions of ng lines as contaminations in the spectrum
                if self.O_cont_bool:
                    conts_clover = ng_fold_clover + O_fold_clover
                    conts_labr = ng_fold_labr + O_fold_labr
                else:
                    conts_clover = ng_fold_clover 
                    conts_labr = ng_fold_labr 
            else:
                if self.O_cont_bool:
                    conts_clover = O_fold_clover
                    conts_labr = O_fold_labr
                else:
                    conts_clover = 0 
                    conts_labr = 0

            # Add up all contributions to prior spectral distribution up
            spec_prior_fold = pm.Deterministic(
                'spec_dets_unscaled',
                nrf_fold_sum + 
                atomic_backg_labr_fold + 
                conts_labr +
                np.ones(len(self.bin_centers[self.mask_energy])) * bg_const[:2, None] / scaling[:, None]
                )

            spec_prior_fold_clover = pm.Deterministic(
                'spec_dets_unscaled_clover', 
                nrf_fold_sum_clover + 
                atomic_backg_clover_fold +
                conts_clover +
                np.ones(len(self.bin_centers[self.mask_energy])) * bg_const[2:, None] / scaling_clover[:, None]
            )


            # Add experimental spectra of LaBr detectors as observables
            expected_counts = pm.Deterministic('spec_dets', pt.clip(spec_prior_fold * scaling[:, None], 1e-6, 1e6))

            # Add experimental spectra of Clover detectors as observables
            expected_counts_clover = pm.Deterministic('spec_dets_clover', pt.clip(spec_prior_fold_clover * scaling_clover[:, None], 1e-6, 1e6))

            detector_obs = pm.Poisson(
                "spectrum_dets_obs", expected_counts, observed=self.spec_meas_all_dets[:2]
            )

            detector_obs_clover = pm.Poisson(
                "spectrum_dets_obs_clover", expected_counts_clover, observed = self.spec_meas_all_dets[2:]
            )

            # Add beam profile measurement as observable 
            beam_obs = pm.Poisson("spectrum_beam_obs", Beam_fold, observed = self.spec_meas_beam
                                  )

            return model
        
    
    def sample_model(self, ndraws=1000, tune=1000, burn=1000, thin=5):


        with self.model:
            # return_inferencedata=True ist der Standard in PyMC v4+
            self.trace = pm.sample(ndraws, tune=tune, chains=2, nuts_sampler='numpyro')    

        # Jetzt funktioniert der Aufruf
        self.trace.to_netcdf(f"Results/binding_results_run{self.runnr}_{self.savename}.nc")
        print("Done! Results were saved to .nc file")


    def plot_nrf_spectrum(self, param_values, fonts_size = 15, plot_size = (15, 10)):
        import matplotlib.pyplot as plt
        import pytensor


        with self.model:
            model_vars = self.model.free_RVs + self.model.deterministics

            fn = pytensor.function(
                inputs=self.model.free_RVs,
                outputs=[self.model.spec_dets, self.model.spec_dets_clover, self.model.nrf_fold],
                on_unused_input='ignore'
            )


        input_data = {rv.name: param_values[rv.name] for rv in self.model.free_RVs if rv.name in param_values}

        spec_labr, spec_clover, nrf_fold_sum = fn(**input_data)


        energies = self.bin_centers[self.mask_energy]
        scale_tot = param_values.get('scale_tot')
        print(scale_tot)

        fig, ax = plt.subplots(4, 1, figsize=plot_size, sharex=True)

        for i, name in enumerate(['L1', 'L3']):
            ax[i].step(energies, self.spec_meas_all_dets.get_value()[i], label=f"Meas {name}", color="gray", alpha=0.5)
            ax[i].step(energies, spec_labr[i], lw=2)
            ax[i].step(energies, nrf_fold_sum[i, 0, 0, :, 0], linestyle="--", color = 'red')
            ax[i].step(energies, nrf_fold_sum[i, 1, 0, :, 0], linestyle="--", color = 'blue')
            ax[i].set_ylabel("Counts per 10 keV", fontsize = fonts_size)


        for i, name in enumerate(['5', '7']):
            ax[i + 2].step(energies, self.spec_meas_all_dets.get_value()[i+2], label=f"Meas {name}", color="gray", alpha=0.5)
            ax[i + 2].step(energies, spec_clover[i], label=f"Model {name} (Total)", lw=2)
            ax[i + 2].set_ylabel("Counts per 10 keV", fontsize = fonts_size)

        axis_ticks(ax, fonts_size)

        plt.tight_layout()


    def optimize_calibration(self, det_name='L3', cal = [-9158, 1.2808], height = 300, energy = 8125, width = 0.03, bg = 250):

        mode = 'linear' if len(cal) == 2 else 'quadratic'
        det_idx = {'L1': 0, 'L3': 1, '5': 2, '7': 3}[det_name]
        
        rem = self.rema_all_dets.eval()[1, 1, 0] if det_name == 'L3' else self.rema_all_dets.eval()[0, 0, 0]
        rema = pt.linalg.inv(rem)
        energies = self.bin_centers[self.mask_energy]

        
        with pm.Model() as calib_model: 

            offset = pm.Normal("offset", mu=cal[0], sigma=0.01)
            gain = pm.Normal("gain", mu=cal[1], sigma=0.00001)
            
            if mode == 'quadratic':
                quad = pm.Normal("quad", mu=cal[2], sigma=1e-8)
                calibrated_spec = get_meas_spec(self.runnr, det_name, 0, [offset.eval(), gain.eval(), quad.eval()], self.bin_width)[self.mask_energy] - bg * np.ones(len(energies))

            else:
                calibrated_spec = get_meas_spec(self.runnr, det_name, 0, [offset.eval(), gain.eval()], self.bin_width)[self.mask_energy] - bg * np.ones(len(energies))




            plt.step(energies, calibrated_spec)

            expected_counts = pt.dot(rema, calibrated_spec)

            plt.step(energies, expected_counts.eval() / pt.diag(rema).eval())


