#!/usr/bin/env python
import importlib
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pathlib import Path
import numpy as np 
from src.scripts.get_matrix_and_specs import get_respmatrix, get_meas_spec

def pytensor_interp(x_new, x_old, y_old):
    return pt.extra_ops.interp(x_new, x_old, y_old)


class binding:
    def __init__(self, 
                 project: str,
                 runnr: int,  
                 savename: str, 
                 bin_width: int = 10, 
                 max_energy = 10000,
                 branchings = True, 
                 O_cont = False,
                 ng_cont = False, 
                 bg_bins = 10,
                 bg_bins_0deg = 5, 
                 model_bool : bool = True):
        
        self.project = project
        self.PP = importlib.import_module(f'src.projects.{self.project}.paths').ProjectPaths_Binding()
        self.deconv_params = importlib.import_module(f'src.projects.{self.project}.deconv_params').deconv_params
        self.setup = importlib.import_module(f'src.projects.{self.project}.setup').setup



        
        self.O_cont_bool = O_cont
        self.ng_cont_bool = ng_cont
        self.branch_bool = branchings
        self.bg_bins = bg_bins
        self.bg_bins_0deg = bg_bins_0deg


        try: 
            self.runnr = int(runnr)
        except:
            self.runnr = str(runnr)
        self.savename = savename
        self.ebeam = self.deconv_params[runnr]['beam']
        self.bin_width = bin_width
        self.bin_centers = np.arange(bin_width, max_energy + bin_width, bin_width)
        self.low = self.deconv_params[runnr]['low']
        self.high = self.deconv_params[runnr]['high']
        self.mask_energy = (self.bin_centers >= self.low) & (self.bin_centers <= self.high) 
        self.td_cut_low = np.array(self.deconv_params[runnr]['cut_dets_low'])
        self.td_cut_high = np.array(self.deconv_params[runnr]['cut_dets_high'])
        self.dets = np.array(self.setup['detectors'])
        self.mask_germanium = np.where((np.array(self.setup['types']) == 'Clover') | 
                                       (np.array(self.setup['types']) == 'HPGe'))[0]
        self.mask_scinti = np.where((np.array(self.setup['types']) == 'LaBr') |
                                    (np.array(self.setup['types']) == 'CeBr'))[0]

        
        self.mask_hor_scinti = np.where(np.isin(np.array(self.setup['detectors'])[self.mask_scinti], 
                                        ['L1', 'C1', 'GL1', 'GC1', 'L5', 'C5', 'GL5', 'GC5']))[0][0]
        self.mask_ver_scinti = np.where(np.isin(np.array(self.setup['detectors'])[self.mask_scinti], 
                                ['L3', 'C3', 'GL3', 'GC3', 'L7', 'C7', 'GL7', 'GC7']))[0][0]

        self.mask_hor_germanium = np.where(np.isin(np.array(self.setup['detectors'])[self.mask_germanium],
                                        ['G1', '1', 'G5', '5']))[0][0]
        self.mask_ver_germanium = np.where(np.isin(np.array(self.setup['detectors'])[self.mask_germanium], 
                                ['G3', '3', 'G7', '7' ]))[0][0]




        def get_cached_data(name: str, low: int, high: int, bin_width: int, compute_func: callable):
            '''
            Check if a cache file exists for given response matrices. If not, a cache file 
            is created, such that initializing binding for a given bin width and lower and upper 
            energy bound is faster.
            '''
            cache_dir = Path(f"{self.PP.SRC_DIR}/../gen/{self.project}/cache_files")
            cache_dir.mkdir(parents= True, exist_ok=True)
            
            file_path = cache_dir / f"{name}_{low}_{high}_bw{bin_width}.npy"
            
            if file_path.exists():
                return np.load(file_path)
            
            print(f"Calculate {name} initial (Cache: {low}-{high} keV, BW: {bin_width})...")
            data = compute_func()
            np.save(file_path, data)
            return data

        def compute_iso_rema() -> np.ndarray:
            '''
            Load isotropic response matrix. The shape of the output is (n_detectors, n_bins, n_bins). 
            You can change the amount of detectors in the setup.py file.  
            '''
            rema_iso = []
            for det in self.setup['detectors']:
                print(f'Get isotropic response matrix of detector {det}...')
                rema_iso.append(
                    get_respmatrix(self.project, '1.5m_0.5m', det, self.bin_width)[np.ix_(self.mask_energy, self.mask_energy)][:, ::-1, ::-1] 
                )

            rema_iso = np.array(rema_iso)
            return rema_iso

        
        def compute_16O_rema() -> np.ndarray:
            '''
            Load isotropic response matrix. The shape of the output is (n_detectors, n_bins, n_bins). 
            You can change the amount of detectors in the setup.py file.  
            '''
            rema_16O = []
            for det in self.setup['detectors']:
                print(f'Get response matrix of a 0^+ --> 2^+ --> 0^+ cascade for detector {det}...')
                rema_16O.append(
                    get_respmatrix(self.project, '2p_0p', det, self.bin_width)[np.ix_(self.mask_energy, self.mask_energy)][:, ::-1, ::-1] 
                )

            rema_16O = np.array(rema_16O)
            return rema_16O

        def compute_dip_rema() -> np.ndarray:
            '''
            Load response matrices for the decays of dipole excited states to 2^+ or 0^+ states. 
            The shape of the output is (n_detectors, n_dipole_parity, n_states, n_bins, n_bins). 
            n_dipole_parity denotes the two parity options of the spin-1 state, + and -. 
            n_states denotes the number of states that the decay occurs into. The minimum number 
            is n_states = 1 for decays to the ground state. Allowing for decays to excited states
            in the setup.py file increases this number. 
            You can change the amount of detectors in the setup.py file.  
            '''

            rema_dip = [] 
            for det in self.setup['detectors']:
                rema_dip_parities = []
                for parity in ['p', 'm']:
                    rema_dip_spins = []
                    for spin in self.setup['j_states']:
                        sign = '+' if parity == 'p' else '-'
                        print(f'Get response matrix of a 0^+ --> 1^{sign} --> {spin}^+ cascade for detector {det}...')
                        rema_dip_spins.append(
                            get_respmatrix(self.project, f'1{parity}_{spin}p', det, self.bin_width)[np.ix_(self.mask_energy, self.mask_energy)][:, ::-1, ::-1] 
                        )
                    rema_dip_parities.append(rema_dip_spins)
                        
                rema_dip.append(rema_dip_parities)
            
            rema_dip = np.array(rema_dip)
            return rema_dip
        

        self.spec_meas_all_dets = pytensor.shared(np.array([
            get_meas_spec('Example', 
                          self.runnr, 
                          det, 
                          self.deconv_params[runnr]['cal'][idx_det], 
                          bin_width, 
                          max_bin = self.setup['max_energy'])[self.mask_energy]
            for idx_det, det in enumerate(self.setup['detectors'])
        ]))

        self.spec_meas_beam = get_meas_spec(
            'Example',
            self.deconv_params[self.runnr]['0deg_nr'], 
            'ZeroDegree', 
            self.deconv_params[self.runnr]['cal_0deg'], 
            bin_width, 
            max_bin = self.setup['max_energy'])[self.mask_energy]
            

        


        self.rema_iso = pytensor.shared(get_cached_data("rema_iso", self.low, self.high, bin_width, lambda: compute_iso_rema()))
        self.rema_16O = pytensor.shared(get_cached_data("rema_16O", self.low, self.high, bin_width, lambda: compute_16O_rema()))
        self.rema_dip = pytensor.shared(get_cached_data("rema_dip", self.low, self.high, bin_width, lambda: compute_dip_rema()))
        self.rema_beam = pytensor.shared(get_cached_data("rema_beam", self.low, self.high, bin_width, 
                        lambda: get_respmatrix(self.project, '0deg', '0deg', bin_width)[np.ix_(self.mask_energy, self.mask_energy)][:, ::-1, ::-1] ))


        self.steps = self.bin_centers[self.mask_energy]
        if model_bool:
            self.model = self.set_model() 

    def set_model(self):
            
        with pm.Model() as model:


            # PRIORS 
            # The ideas behind this deconvolution approach is to describe each bin height as a sum of different cascades (i.e. 1p0p, 1m0p, 1m2p, 1p2p, 1p0pe, 1m0pe). 
            # The priors for the each of the distributions are determined using a top-down deconvolutions. 
            # First, the priors for 1p0p and 1m0p are determined as prior_1m0p = rema_1m0p x spec_L3 and prior_1p0p = rema_1p0p x spec_meas_L1

            # Constant background for detector array 

            bg_const = pm.Deterministic(
                'bg_const', 
                pt.mean(self.spec_meas_all_dets[:, -self.bg_bins:], axis=1),
            )



            if self.ng_cont_bool: # TODO export mean values of n,g distributions to extra file
                # Intoduce ng lines as contaminations in the spectrum. Assume isotropic radiation and a gaussian with a width corresponding to the detector resolution (different for Germanium and Scintillator)
                E_ng = pm.TruncatedNormal('E_ng', [7631, 7645, 7721], lower = [7630, 7644, 7720], upper = [7632, 7646, 7722])
                I_ng = pm.Exponential('I_ng', 1 / 50e3, shape=(E_ng.shape[0],)) 

                ng_width_scinti = pm.Uniform('ng_width_scinti', 5, 50, shape = (len(self.dets[self.mask_scinti]), ))
                ng_width_germanium = pm.Uniform('ng_width_germanium', 5, 15, shape = (len(self.dets[self.mask_germanium]), ))


                sqrt_2pi = np.sqrt(2 * np.pi)
                # Scintillator detectors
                ng_inc_scinti = pm.Deterministic('ng_inc_scinti', 
                    pt.clip(
                        pt.sum(
                            (I_ng[None, :, None] / (sqrt_2pi * ng_width_scinti[:, None, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None, None, :] - E_ng[None, :, None]) / ng_width_scinti[:, None, None]) ** 2),
                            axis=1
                        ), 1e-6, pt.inf
                    )
                )# dimensions (dets, bins)

                # Germanium detectors
                ng_inc_germanium = pm.Deterministic('ng_inc_germanium', 
                    pt.clip(
                        pt.sum(
                            (I_ng[None, :, None] / (sqrt_2pi * ng_width_germanium[:, None, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None, None, :] - E_ng[None, :, None]) / ng_width_germanium[:, None, None]) ** 2),
                            axis=1
                        ), 1e-6, pt.inf
                    )
                )# dimensions (dets, bins)

                # Fold both incident spectra with corresponding detector response matrices
                ng_fold_scinti = pm.Deterministic('ng_fold_scinti', self.rema_iso[self.mask_scinti] @ ng_inc_scinti[:, :, None])[:, :, 0]  # shape (det, bins) x (det, bins, bins) -> (det, ng, bins)
                ng_fold_germanium = pm.Deterministic('ng_fold_germanium', self.rema_iso[self.mask_germanium] @ ng_inc_germanium[:,:, None])[:, :, 0] # shape (det, binss) x (det, bins, bins) -> (det, ng, bins)


            
            if self.O_cont_bool:
                # Intoduce 16O as contaminations in the spectrum. Assume isotropic radiation and a gaussian with a width corresponding to the detector resolution (different for Germanium and Scintillator)
                E_O_scinti = pm.TruncatedNormal('E_O_scinti', 6910, lower = 6890, upper = 6950 )
                E_O_germanium = pm.TruncatedNormal('E_O_germanium', 6910, lower = 6900, upper = 6930)

                I_O = pm.Exponential('I_O', 1 / 100e3, shape=(1,)) 
                O_width_scinti = pm.Uniform('O_width_scinti', 5, 50, shape = (len(self.dets[self.mask_scinti]), ))
                O_width_germanium = pm.Uniform('O_width_germanium', 5, 15, shape = (len(self.dets[self.mask_scinti]), ))
                sqrt_2pi = np.sqrt(2 * np.pi)

                # Scintillator dets
                O_inc_scinti = pm.Deterministic('O_inc_scinti', 
                    pt.clip(
                            (I_O / (sqrt_2pi * O_width_scinti[:, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None, :] - E_O_scinti) / O_width_scinti[:, None]) ** 2),
                         1e-6, pt.inf
                    )
                )# dets, bin

                # Germanium dets
                O_inc_germanium = pm.Deterministic('O_inc_germanium', 
                    pt.clip(
                        (I_O / (sqrt_2pi * O_width_germanium[:, None])) * pt.exp(-0.5 * ((self.bin_centers[self.mask_energy][None :] - E_O_germanium) / O_width_germanium[:, None]) ** 2),
                     1e-6, pt.inf
                    )
                )# dets, bins


                # Fold both incident spectra with corresponding detector response matrices
                O_fold_scinti = pm.Deterministic('O_fold_scinti', self.rema_16O[self.mask_scinti] @ O_inc_scinti[:, :, None])[:, :, 0]  # shape (det, bins) x (det, bins, bins) -> (det, ng, bins)
                O_fold_germanium = pm.Deterministic('O_fold_germanium', self.rema_16O[self.mask_germanium] @ O_inc_germanium[:, :, None])[:, :, 0] # shape (det, binss) x (det, bins, bins) -> (det, ng, bins)




     

            # Set a fit parameter that describes the E1 to M1 ratio. Its called alpha and describes the total spectrum S_tot like S_tot = (1- alpha) * S_1m0p + alpha S_1p0p
            # It helps setting a good prior 
            # alpha = pm.Uniform('alpha', 0, 1 , shape = ())

            prior_scinti, prior_germanium = [], []
            for type in ['scinti', 'germanium']:
                for idx_parity, parity in enumerate(['p', 'm']):
                    mask_type = self.mask_scinti if type == 'scinti' else self.mask_germanium
                    if type == 'scinti':
                        mask_parity = self.mask_hor_scinti if type == 'p' else self.mask_ver_scinti
                    else:
                        mask_parity = self.mask_hor_germanium if type == 'p' else self.mask_ver_germanium

                    prior_raw = ((self.spec_meas_all_dets[mask_type][mask_parity] - bg_const[mask_type][mask_parity])
                                    @ pt.linalg.inv(self.rema_dip[mask_type][mask_parity][idx_parity][0]) 
                                    ) # dimensions are (detector, parity, final_state, bins, bins) and (detector, bins), respectively 
                
                    contributions = [] # subtract contributions from (n,gamma) or contaminating NRF lines (i.e. 16O at ~6915 keV)
                    if self.ng_cont_bool:
                        contributions.append(ng_inc_scinti[mask_parity])
                    if self.O_cont_bool:
                        contributions.append(O_inc_scinti[mask_parity])
                    total_correction = pt.sum(pt.stack(contributions), axis=0) if contributions else 0
                    

                    prior_nong = prior_raw - total_correction if total_correction != 0 else prior_raw
                    prior = pt.where((self.ebeam * self.td_cut_high[mask_type][mask_parity] >= self.bin_centers[self.mask_energy]) 
                                        & (self.bin_centers[self.mask_energy] >= self.ebeam * self.td_cut_low[mask_type][mask_parity])
                                        , prior_nong, 0.0) #* (alpha) 
                    prior_scinti.append(prior) if type == 'scinti' else prior_germanium.append(prior)


            prior_1p0p_scinti, prior_1m0p_scinti = prior_scinti[0], prior_scinti[1]
            prior_1p0p_germanium, prior_1m0p_germanium = prior_germanium[0], prior_germanium[1]

            

            # For the inelastic channels, two possibilities emerge. Either use the part of the measured spectrum for a guess, or shift the gs-decay curve by the final state energy
            # Shift gs decay shape to lower energies as guess with branching ratio as scaling factors
            if self.branch_bool:

                p1_states_scinti, m1_states_scinti = [prior_1p0p_scinti], [prior_1m0p_scinti] 
                p1_states_germanium, m1_states_germanium = [prior_1p0p_germanium], [prior_1m0p_germanium] 
                for idx_state, e_state in enumerate(self.setup['e_states'][1:]): # exclude ground state by index 1:
                    energy_shift = round(e_state / 10)
                    spin = self.setup['j_states'][1:][idx_state]

                    prior_empty = pt.zeros_like(prior_1p0p_scinti)
                    for parity in ['p', 'm']:
                        br_str = f'br_1{parity}{spin}p'
                        br = pm.Exponential(br_str, 1 / 0.5)

                        prior_gs_scinti = prior_1m0p_scinti if parity == 'm' else prior_1p0p_scinti
                        prior_gs_germanium = prior_1m0p_germanium if parity == 'm' else prior_1p0p_germanium

                        shifted_br_scinti = prior_gs_scinti[energy_shift:] * br
                        shifted_br_germanium = prior_gs_germanium[energy_shift:] * br

                        prior_br_scinti = pt.set_subtensor(prior_empty[:shifted_br_scinti.shape[0]], shifted_br_scinti)
                        prior_br_germanium = pt.set_subtensor(prior_empty[:shifted_br_germanium.shape[0]], shifted_br_germanium)


                        p1_states_scinti.append(prior_br_scinti) if parity == 'p' else m1_states_scinti.append(prior_br_scinti) 
                        p1_states_germanium.append(prior_br_germanium) if parity == 'p' else m1_states_germanium.append(prior_br_germanium) 


                p1_states_scinti, m1_states_scinti = pt.stack(p1_states_scinti), pt.stack(m1_states_scinti)
                p1_states_germanium, m1_states_germanium = pt.stack(p1_states_germanium), pt.stack(m1_states_germanium)

            else: 
                p1_states_scinti = pt.stack([prior_1p0p_scinti])
                m1_states_scinti = pt.stack([prior_1m0p_scinti])
                p1_states_germanium = pt.stack([prior_1p0p_germanium])
                m1_states_germanium = pt.stack([prior_1m0p_germanium])



            priors_nrf_scinti = pm.Deterministic('nrf_inc_scinti',
                pt.stack([p1_states_scinti, m1_states_scinti]))
            priors_nrf_germanium = pm.Deterministic('nrf_inc_germanium',
                pt.stack([p1_states_germanium, m1_states_germanium]))


            nrf_inc_free_scinti = pm.Exponential(
                'nrf_inc_free_scinti', 
                lam = 1.0 / pt.clip(priors_nrf_scinti, 1e-6, np.inf), 
                shape = (len(self.dets[self.mask_scinti]), len(self.setup['e_states']), len(self.bin_centers[self.mask_energy]))
            )

            nrf_inc_free_germanium = pm.Exponential(
                'nrf_inc_free_germanium', 
                lam = 1.0 / pt.clip(priors_nrf_germanium, 1e-6, np.inf), 
                shape = (len(self.dets[self.mask_germanium]), len(self.setup['e_states']), len(self.bin_centers[self.mask_energy]))
            )


            # Fold all NRF components. 
            nrf_fold_scinti = pm.Deterministic('nrf_fold_scinti', self.rema_dip[self.mask_scinti][:, :, :len(self.setup['e_states']), ::-1, ::-1] @ nrf_inc_free_scinti[None, :, :, :, None])[:, :, :, :, 0] # shapes: (dets, parites, states, bins, bins) x (parities, states, bins, dummy)
            nrf_fold_sum_scinti = pm.Deterministic('nrf_fold_sum_scinti', pt.sum(nrf_fold_scinti, axis=(1,2))) # shapes (dets, bins)
        
            # Fold all NRF components for germaniums. 
            nrf_fold_germanium = pm.Deterministic('nrf_fold_germanium', self.rema_dip[self.mask_germanium][:, :, :len(self.setup['e_states']), ::-1, ::-1] @ nrf_inc_free_germanium[None, :, :, :, None])[:, :, :, :, 0] # shapes: (dets, parites, states, bins, bins) x (parities, states, bins, dummy)
            nrf_fold_sum_germanium = pm.Deterministic('nrf_fold_sum_germanium', pt.sum(nrf_fold_germanium, axis=(1,2))) # shapes (dets, bins)






            # Prior distribution for atomic background, modeled as an exponential function with slope and scale parameters. 
            atomic_slope_scinti = pm.Uniform(
                "Atomic_slope_scinti", 0.3e-3, 10e-3, shape=(len(self.dets[self.mask_scinti]), )
            )   
            atomic_scale_scinti = pm.Uniform("Atomic_scale_scinti", 4, 32, shape=(len(self.dets[self.mask_scinti]), ))
            atomic_backg_scinti_inc = pm.Deterministic(
                "Atomic_backg_scinti_inc", pt.clip(
                    pt.exp(atomic_scale_scinti[:, None] - atomic_slope_scinti[:, None] * self.bin_centers[self.mask_energy])
                    - pt.exp(atomic_scale_scinti[:, None] - atomic_slope_scinti[:, None] * self.ebeam), 0, np.inf
                )
            )
            atomic_backg_scinti_fold = pm.Deterministic("Atomic_backg_scinti_fold", self.rema_iso[self.mask_scinti] @ atomic_backg_scinti_inc[:, :, None])[:, :, 0]   



            # Same for germanium detectors 
            atomic_slope_germanium = pm.Uniform(
                "Atomic_slope_germanium", 0.3e-3, 10e-3, shape=(len(self.dets[self.mask_germanium]), )
            )   
            atomic_scale_germanium = pm.Uniform("Atomic_scale_germanium", 4, 32, shape=(len(self.dets[self.mask_germanium]), ))
            atomic_backg_germanium_inc = pm.Deterministic(
                "Atomic_backg_germanium_inc", pt.clip(
                    pt.exp(atomic_scale_germanium[:, None] - atomic_slope_germanium[:, None] * self.bin_centers[self.mask_energy])
                    - pt.exp(atomic_scale_germanium[:, None] - atomic_slope_germanium[:, None] * self.ebeam), 0, np.inf
                )
            )
            atomic_backg_germanium_fold = pm.Deterministic("Atomic_backg_germanium_fold", self.rema_iso[self.mask_germanium] @ atomic_backg_germanium_inc[:, :, None])[:, :, 0]   



            cont_germanium = []
            cont_scinti = []

            if self.ng_cont_bool:
                cont_scinti.append(ng_fold_germanium)
                cont_germanium.append(ng_fold_scinti)

            if self.O_cont_bool:
                cont_germanium.append(O_fold_germanium)
                cont_scinti.append(O_fold_scinti)

            conts_germanium = pt.sum(pt.stack(cont_germanium), axis=0) if cont_germanium else 0
            conts_scinti = pt.sum(pt.stack(cont_scinti), axis=0) if cont_scinti else 0


            # Add up all contributions to prior spectral distribution up
            spec_prior_fold_scinti = pm.Deterministic(
                'spec_dets_scinti',
                pt.clip(
                nrf_fold_sum_scinti + 
                atomic_backg_scinti_fold + 
                conts_scinti +
                np.ones(len(self.bin_centers[self.mask_energy])) * bg_const[self.mask_scinti][:, None],
                1e-6, pt.inf) * np.array(self.setup['eff_ratio'])[self.mask_scinti][:, None]
            )

            spec_prior_fold_germanium = pm.Deterministic(
                'spec_dets_germanium', 
                pt.clip(
                nrf_fold_sum_germanium + 
                atomic_backg_germanium_fold +
                conts_germanium +
                np.ones(len(self.bin_centers[self.mask_energy])) * bg_const[self.mask_germanium][:, None],
                1e-6, pt.inf) * np.array(self.setup['eff_ratio'])[self.mask_germanium][:, None]
            )


            # completion of detector array model by comparing model with measured data for scintillators and germaniums
            detector_obs = pm.Poisson(
                "spectrum_dets_obs_scinti", spec_prior_fold_scinti, observed = self.spec_meas_all_dets[self.mask_scinti]
            )

            # completion of detector array model by comparing model with measured data for scintillators and germaniums
            detector_obs_clover = pm.Poisson(
                "spectrum_dets_obs_clover_scinti", spec_prior_fold_germanium, observed = self.spec_meas_all_dets[self.mask_germanium]
            )



            # Second part of the model revolves around the deconvolution of the beam profile. 
            # The principle is very similar to the binwise deconvolution principle introduced in the model above. 
            # The difference is that now there is only one detector and one contribution to the total spectrum 
            # (and a constant background)




            steps = pm.Deterministic('steps', pytensor.shared(self.steps))
            bg_const_beam = pm.Deterministic('bg_const_beam', pytensor.shared(np.mean(self.spec_meas_beam[-self.bg_bins_0deg:])))


            pho_func_bound = pt.where((self.ebeam * self.deconv_params[self.runnr]['cut_dets_0deg'][1] >= self.bin_centers[self.mask_energy]) 
                                        & (self.bin_centers[self.mask_energy] >= self.ebeam * self.deconv_params[self.runnr]['cut_dets_0deg'][0]), 
                                        pt.clip( (self.spec_meas_beam - bg_const_beam) @ pt.linalg.inv(self.rema_beam), 
                                        1e-6, pt.inf), 
                                        1e-6)
            
            pho_func = pm.Exponential('pho_inc', lam = 1 / pho_func_bound ,
                                        shape = len(self.steps))


            pho_func_fold = pm.Deterministic('pho_fold', (pho_func @ self.rema_beam)) 

            scala = pm.Exponential('scala', lam = 1 / 1)


            obs = pm.Deterministic('obs', pytensor.shared(self.spec_meas_beam))

            beam_obs = pm.Poisson("spectrum_beam_obs", pho_func_fold * scala + bg_const_beam, observed = pytensor.shared(self.spec_meas_beam))


            
            return model

        
    
    def sample_model(self, ndraws=1000, tune=1000, burn=1000, thin=5):
        '''
        Compile the model by simply executing this function. You could adjust the parameters 
        of the pm.sample function if you have problems with convergence, but this shouldn't be necessary usually.
        '''

        with self.model:
            self.trace = pm.sample(ndraws, tune=tune, chains=2, nuts_sampler='numpyro')    

        save_dir = (Path(self.PP.SRC_DIR).resolve().parent / "gen" / f"{self.project}" / "Results")
        save_dir.mkdir(parents=True, exist_ok=True)
        self.trace.to_netcdf(f"{save_dir}/binding_results_{self.project}_run{self.runnr}_{self.savename}.nc")
        print("Done! Results were saved to .nc file")


