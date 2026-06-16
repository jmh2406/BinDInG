#!/usr/bin/env python

import importlib
import arviz as az
import uncertainties.unumpy as unp
import numpy as np 
from get_matrix_and_specs import get_respmatrix, get_meas_spec
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from src.scripts.binding_bins import binding
from src.scripts.misc import axis_ticks, set_font


class Plot_binding:
    def __init__(self, project : str = 'Example',
                run : int = None,
                savename : str = None,
                binwise : bool = True,
                branch_bool : bool = True, ):
        
        self.project = project
        self.runnr = run
        self.binwise = binwise
        self.savename = savename
        self.PP = importlib.import_module(f'src.projects.{self.project}.paths').ProjectPaths_Binding()
        self.deconv_params = importlib.import_module(f'src.projects.{self.project}.deconv_params').deconv_params
        self.setup = importlib.import_module(f'src.projects.{self.project}.setup').setup


        self.bind_inst = binding(self.project, self.runnr, self.savename)

        

        self.branch_bool = branch_bool

        idata = az.from_netcdf(f"{self.PP.SRC_DIR}/../gen/{project}/Results/binding_results_{self.project}_run{self.runnr}_{self.savename}.nc")

        self.bind_inst = binding(project, run, savename)


        self.post_total = unp.uarray(
            np.concatenate([idata.posterior["spec_dets_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["spec_dets_germanium"].mean(dim=["chain", "draw"]).values], axis = 0),
            np.concatenate([idata.posterior["spec_dets_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["spec_dets_germanium"].std(dim=["chain", "draw"]).values], axis = 0)
        )

        if self.binwise == True:
            self.post_nrf = unp.uarray(
                np.concatenate([idata.posterior["nrf_inc_free_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["nrf_inc_free_germanium"].mean(dim=["chain", "draw"]).values], axis=0),
                np.concatenate([idata.posterior["nrf_inc_free_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["nrf_inc_free_germanium"].std(dim=["chain", "draw"]).values], axis=0)
            )
        else:
            self.post_nrf =  unp.uarray(
                np.concatenate([idata.posterior["nrf_inc_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["nrf_inc_germanium"].mean(dim=["chain", "draw"]).values], axis=0),
                np.concatenate([idata.posterior["nrf_inc_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["nrf_inc_germanium"].std(dim=["chain", "draw"]).values], axis=0)
            )
        self.post_nrf_fold = unp.uarray(
            np.concatenate([idata.posterior["nrf_fold_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["nrf_fold_germanium"].mean(dim=["chain", "draw"]).values], axis=0),
            np.concatenate([idata.posterior["nrf_fold_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["nrf_fold_germanium"].std(dim=["chain", "draw"]).values], axis=0)
        )
        self.post_ato = unp.uarray(
            np.concatenate([idata.posterior["Atomic_backg_scinti_inc"].mean(dim=["chain", "draw"]).values, idata.posterior["Atomic_backg_germanium_inc"].mean(dim=["chain", "draw"]).values], axis=0),
            np.concatenate([idata.posterior["Atomic_backg_scinti_inc"].std(dim=["chain", "draw"]).values, idata.posterior["Atomic_backg_germanium_inc"].std(dim=["chain", "draw"]).values], axis=0)
        )

        if self.bind_inst.ng_cont_bool:
            self.post_ng = unp.uarray(
                np.concatenate([idata.posterior["ng_inc_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["ng_inc_germanium"].mean(dim=["chain", "draw"]).values], axis=0),
                np.concatenate([idata.posterior["ng_inc_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["ng_inc_germanium"].std(dim=["chain", "draw"]).values], axis=0)
            )
            self.post_ng_fold = unp.uarray(
                np.concatenate([idata.posterior["ng_fold_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["ng_fold_germanium"].mean(dim=["chain", "draw"]).values], axis=0),
                np.concatenate([idata.posterior["ng_fold_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["ng_fold_germanium"].std(dim=["chain", "draw"]).values], axis=0)
            )
        if self.bind_inst.O_cont_bool:
            self.post_O = unp.uarray(
                np.concatenate([idata.posterior["O_inc_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["O_inc_germanium"].mean(dim=["chain", "draw"]).values], axis=0),
                np.concatenate([idata.posterior["O_inc_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["O_inc_germanium"].std(dim=["chain", "draw"]).values], axis=0)
            )
            self.post_O_fold = unp.uarray(
                np.concatenate([idata.posterior["O_fold_scinti"].mean(dim=["chain", "draw"]).values, idata.posterior["O_fold_germanium"].mean(dim=["chain", "draw"]).values], axis=0),
                np.concatenate([idata.posterior["O_fold_scinti"].std(dim=["chain", "draw"]).values, idata.posterior["O_fold_germanium"].std(dim=["chain", "draw"]).values], axis=0)
            )

        # idata_beam = az.from_netcdf(f"{self.PP.SRC_DIR}/../gen/{project}/Results/binding_results_{self.project}_run{self.runnr}_{self.savename}_beam.nc")


        self.res_beam_inc = unp.uarray(
            idata.posterior["pho_inc"].mean(dim=["chain", "draw"]).values, 
            idata.posterior["pho_inc"].std(dim=["chain", "draw"]).values) 
        
        self.res_beam_fold = unp.uarray(
            idata.posterior["pho_fold"].mean(dim=["chain", "draw"]).values, 
            idata.posterior["pho_fold"].std(dim=["chain", "draw"]).values) 

        
        self.beam_scala = unp.uarray(
            idata.posterior["scala"].mean(dim=["chain", "draw"]).values, 
            idata.posterior["scala"].std(dim=["chain", "draw"]).values) 
        
        self.steps = self.bind_inst.steps



        detectors = self.setup['detectors']

    def plot_incs(self,
                color_meas: str = 'black',
                color_fit: str = 'purple', 
                collist_hor: list = ['red', 'darkred', 'salmon', 'firebrick'],
                collist_vert: list = ['blue', 'darkblue', 'cyan', 'teal'],
                color_ato: str = 'gold',
                color_ng: str = 'green',
                beam_prof: bool = True,
                beam_prof_with_unc: bool = False, 
                beam_deconv: str = False,
                ato: bool = True, 
                fonts_size: int = 25, 
                plot_shape: tuple = (15, 20), 
                plot_arr: tuple = None, 
                dets: list = None) -> None:
            

        if plot_arr == None:
            plot_arr = (len(self.setup['detectors']), 1)

        if dets == None:
            dets = self.setup['detectors']


        if beam_deconv:


            set_font(25)
            fig, ax = plt.subplots(figsize=plot_shape)
            axis_ticks(ax)


            beam_inc_line, = ax.step(self.bind_inst.steps, unp.nominal_values(self.res_beam_inc), color='red', where='mid')
            beam_inc_err = ax.fill_between(self.steps, unp.nominal_values(self.res_beam_inc) + unp.std_devs(self.res_beam_inc), 
                                            unp.nominal_values(self.res_beam_inc) - unp.std_devs(self.res_beam_inc), 
                                            color = collist_hor[0], alpha=0.3, step='mid')

            beam_meas_line, = ax.step(self.steps, self.bind_inst.spec_meas_beam, color=color_meas, where='mid')
            beam_fold_line, = ax.step(self.steps, unp.nominal_values(self.res_beam_fold), color=color_fit, where='mid')
            beam_fold_err = ax.fill_between(self.steps, unp.nominal_values(self.res_beam_fold) + unp.std_devs(self.res_beam_fold), 
                                            unp.nominal_values(self.res_beam_fold) - unp.std_devs(self.res_beam_fold),
                                            color=color_fit, alpha=0.3, step='mid')


            ax.legend(handles=[(beam_inc_line, beam_inc_err),
                                (beam_fold_line, beam_fold_err),
                                beam_meas_line],
                                labels=['Deconvolved beam profile', 'Folded beam profile', 'Measured beam profile'],
                                handler_map={tuple: HandlerTuple(ndivide=None, pad = -2)},
                                handlelength = 2,)
                                # bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False) 

            plt.xlabel("Energy in keV", fontsize = fonts_size)
            plt.ylabel("Counts per 10 keV", fontsize = fonts_size)
            plt.tight_layout()
            plt.xlim(self.deconv_params[self.runnr]['low'], self.deconv_params[self.runnr]['high'])

            plt.show()
            return 
        
        idx_hor = np.concatenate([[self.bind_inst.mask_germanium[self.bind_inst.mask_hor_germanium]]
                   , [self.bind_inst.mask_scinti[self.bind_inst.mask_hor_scinti]]])
                

        det_names = [f"{types} {self.setup['detectors'][idx]} (Horizontal)"
                     if idx in idx_hor
                     else f"{types} {self.setup['detectors'][idx]} (Vertical)" 
                     for idx, types in enumerate(self.setup['types']) ] 


        set_font(25)
        fig, axes = plt.subplots(plot_arr[0], plot_arr[1], figsize=plot_shape, sharex=True)
        for idx_ax, det in enumerate(self.setup['detectors']):
            ax = axes[idx_ax]       
            axis_ticks(ax)

            scal = self.setup['eff_ratio'][idx_ax]

            # Data
            self.meas_data =  self.bind_inst.spec_meas_all_dets.eval()[idx_ax] 
            ax.step(self.steps, self.meas_data, color=color_meas, alpha=1, label='Measured data', where='mid')
            
            # Model fit
            fit_line, = ax.step(self.steps, unp.nominal_values(self.post_total[idx_ax]), 
                                color=color_fit, lw=1.5, label='Total fit', where='mid')
            fit_err = ax.fill_between(self.steps, unp.nominal_values(self.post_total[idx_ax]) + unp.std_devs(self.post_total[idx_ax]), 
                                      unp.nominal_values(self.post_total[idx_ax]) - unp.std_devs(self.post_total[idx_ax]), 
                                      color=color_fit, alpha=0.2, step='mid')
            
            # Atomic background
            if ato:
                ato_line, = ax.step(self.steps, unp.nominal_values(self.post_ato[idx_ax]) * scal * np.diag(self.bind_inst.rema_iso.eval()[idx_ax]), 
                                    color=color_ato, ls='-', label='Atomic bg', where='mid')
                ato_err = ax.fill_between(self.steps, (unp.nominal_values(self.post_ato[idx_ax]) + unp.std_devs(self.post_ato[idx_ax])) * scal * np.diag(self.bind_inst.rema_iso.eval()[idx_ax]), 
                                          (unp.nominal_values(self.post_ato[idx_ax]) - unp.std_devs(self.post_ato[idx_ax])) *scal * np.diag(self.bind_inst.rema_iso.eval()[idx_ax]), 
                                          color=color_ato, alpha=0.2, step='mid')
            
            
            
            m1_line, e1_line, m1_err, e1_err = [], [], [], []


            for state in range(len(self.setup['e_states'])):

                if idx_ax in self.bind_inst.mask_germanium: 
                    mask_m1 = self.bind_inst.mask_germanium[self.bind_inst.mask_hor_germanium]
                    mask_e1 = self.bind_inst.mask_germanium[self.bind_inst.mask_ver_germanium]
                else: 
                    mask_m1 = self.bind_inst.mask_scinti[self.bind_inst.mask_hor_scinti]
                    mask_e1 = self.bind_inst.mask_scinti[self.bind_inst.mask_ver_scinti]



                m1_line.append(ax.step(self.steps, unp.nominal_values(self.post_nrf[mask_m1, state]) * np.diag(self.bind_inst.rema_dip.eval()[idx_ax][0][state]) * scal, 
                                       color=collist_hor[state], alpha=0.8, where = 'mid')[0])
                m1_err.append(ax.fill_between(self.steps, (unp.nominal_values(self.post_nrf[mask_m1, state]) + unp.std_devs(self.post_nrf[mask_m1, state])) * np.diag(self.bind_inst.rema_dip.eval()[idx_ax][0][state]) * scal,
                                    (unp.nominal_values(self.post_nrf[mask_m1, state]) - unp.std_devs(self.post_nrf[mask_m1, state])) * np.diag(self.bind_inst.rema_dip.eval()[idx_ax][0][state]) * scal,
                                    color=collist_hor[state], alpha=0.2, step='mid'))
                    
                e1_line.append(ax.step(self.steps, unp.nominal_values(self.post_nrf[mask_e1, state]) * np.diag(self.bind_inst.rema_dip.eval()[idx_ax][1][state]) * scal,
                                    color=collist_vert[state], alpha=0.8, where = 'mid')[0])
                e1_err.append(ax.fill_between(self.steps, (unp.nominal_values(self.post_nrf[mask_e1, state]) + unp.std_devs(self.post_nrf[mask_e1, state])) * np.diag(self.bind_inst.rema_dip.eval()[idx_ax][1][state]) * scal,
                                    (unp.nominal_values(self.post_nrf[mask_e1, state]) - unp.std_devs(self.post_nrf[mask_e1, state])) * np.diag(self.bind_inst.rema_dip.eval()[idx_ax][1][state]) * scal,
                                    color=collist_vert[state], alpha=0.2, step='mid'))

                    
            if self.bind_inst.ng_cont_bool:
                ng_line = ax.step(self.steps, unp.nominal_values(self.post_ng[idx_ax]) * np.diag(self.bind_inst.rema_iso.eval()[idx_ax]) * scal, 
                                  color=color_ng, ls='-', where='mid')[0]
                ng_err = ax.fill_between(self.steps, (unp.nominal_values(self.post_ng[idx_ax]) + unp.std_devs(self.post_ng[idx_ax])) * np.diag(self.bind_inst.rema_iso.eval()[idx_ax]) * scal, 
                                        (unp.nominal_values(self.post_ng[idx_ax]) - unp.std_devs(self.post_ng[idx_ax])) * np.diag(self.bind_inst.rema_iso.eval()[idx_ax]) * scal,
                                        color=color_ng, alpha=0.2, step='mid')

            if self.bind_inst.O_cont_bool:
                O_line = ax.step(self.steps, unp.nominal_values(self.post_O[idx_ax]) * np.diag(self.bind_inst.rema_16O.eval()[idx_ax]) * scal, 
                                  color=color_ng, ls='-', where='mid')[0]
                O_err = ax.fill_between(self.steps, (unp.nominal_values(self.post_O[idx_ax]) + unp.std_devs(self.post_O[idx_ax])) * np.diag(self.bind_inst.rema_16O.eval()[idx_ax]) * scal, 
                                        (unp.nominal_values(self.post_O[idx_ax]) - unp.std_devs(self.post_O[idx_ax])) * np.diag(self.bind_inst.rema_16O.eval()[idx_ax]) * scal,
                                        color=color_ng, alpha=0.2, step='mid')

            if beam_prof or beam_prof_with_unc:
                beam_scal = self.meas_data.max() / np.max(unp.nominal_values(self.res_beam_inc)) 
                beam_line = ax.step(self.steps, unp.nominal_values(self.res_beam_inc) * beam_scal, ls = '--', color = 'grey', where = 'mid')[0]

                if beam_prof_with_unc: 
                    beam_err = ax.fill_between(self.steps, (unp.nominal_values(self.res_beam_inc) + unp.std_devs(self.res_beam_inc)) * beam_scal,
                                                (unp.nominal_values(self.res_beam_inc) - unp.std_devs(self.res_beam_inc)) * beam_scal, 
                                                color = 'grey', alpha = 0.2, step = 'mid') 



            ax.set_title(f"{det_names[idx_ax]}", fontsize = fonts_size)
            ax.set_ylabel(f"Counts per {self.bind_inst.bin_width} keV", fontsize = fonts_size)
            ax.set_ylim(0, self.meas_data.max() * 1.2)
            ax.set_xlim(self.steps.min(), self.steps.max())

            if idx_ax == 0:

                handles = [(fit_line, fit_err)]
                lables = ['Fit'] 
                
                if ato:
                    handles = handles + [(ato_line, ato_err)]
                    lables = lables + ['At. backg.']

                if self.bind_inst.ng_cont_bool:
                    handles = handles + [(ng_line, ng_err)]
                    lables = lables + ['$(n, \gamma)$ cont.']

                if self.bind_inst.O_cont_bool:
                    handles = handles + [(O_line, O_err)]
                    lables = lables + [r'\textsuperscript{16}O cont.']

                if beam_prof and not beam_prof_with_unc:
                    handles = handles + [beam_line]
                    lables = lables + [r'Beam profile (arb. scale)']
                
                if beam_prof_with_unc:
                    handles = handles + [(beam_line, beam_err)]
  
        return 

