setup = {
    'name' : 'Example', 
    'e_states' : [0, 884, 1070], 
    'j_states' : [0, 2, 0], 
    'detectors': ['L1', 'L3', '5', '7'], 
    'types': ['LaBr', 'LaBr', 'Clover', 'Clover'],
    'h5_indices' : [28, 29, (0, 1, 2, 3), (4, 5, 6, 7)], 
    'h5_indices_0deg' : 30, 
    'max_energy': 10000, 
}


"""

- name: Name of the project. Should match with the name of the directory
- e_states: Energy of the excited states that need to be considered in the deconvolution. 
  The ground state is denoted by 0. The energies should be given in keV. 
- j_states: Spins of the excited states given in 'e_states'. Length needs to be the same as for 'e_states'
- detectors: Names of detectors used in the deconvolution. Names should follow the 
  nomenclature used at the Cloverarray at HIGS, because this way, detectors at critical angles can be singled 
  out by the code automatically. 
  The HIGS nomenclature for detectors simply depends on the position and type 
  of the detectors. Detectors are differentiated between cross detectors (theta = 90°) and backwards angles 
  (theta ~ 130°). Backwards angles are always labled with a in front of the detector name. 
  The number of the detectors is determined by the azimuthal angle. phi = 0° is equivalent to detector '1'. 
  phi = 45° equates to detector two '2', phi = 90° to detector '3' and so on until detector '8'. 
  The name changes also depending on the detector type. LaBr and CeBr detectors are labeled with a 'L' or 'C' 
  prefix, respectively. HPGe and Clover detectors don't have any prefix. The detectors in the example describe a 
  cross configuration with two Clover and two LaBr detectors parallel and perpendicular to the horizontal 
  polarization plane
- types: Types of detectors. There are four possible types of detectors 
  implemented:'HPGe' (One-crystal high-purity germanium detectors), 'Clover' (four leave Clover detectors), 
  'LaBr' and 'CeBr' (scintillation detectors). Length needs to match length of 'detectors'.
- eff_ratio: Ratio of simulated and measured efficiency scale. Not important on 
  an absolute scale. But important to scale the detectors relative to each other, in case the simulation 
  deviates from the measured efficiencies for one detector only. Can be left as a list of ones if the deviation of 
  the measured efficiency to the simulated efficiency is the same for all detectors, or if there is no deviation at all.
- h5_indices: The input shape for the detector response matrices needs to 
  be in the .h5 format. Here multiple detector response matrices can be stored in a single file. Which index in the 
  .h5 file corresponds to which detector can be defined here.  
- h5_indices_0deg: Index of the detector response matrix of the 0-degree detector in the .h5 file. 
- max_energy: Maximum energy that you want to investigate with your deconvolution.


"""