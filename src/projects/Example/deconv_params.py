deconv_params = {
    179 : {
        'low' : 4600,
        'high' : 6600,
        'beam' : 6170,
        'cal' : [
            [-2.475583e03, 7.530190e-1],
            [ -2880.713883016002 , 0.9221887479445257 , -1.5004698144231412e-05 ],
            [0, 1],
            [0, 1]
            ],
        'cut_dets_high' : [1.04, 1.04, 1.04, 1.04],
        'cut_dets_low' : [0.95, 0.95, 0.95, 0.95],
        'cal_0deg' : [0, 1], 
        'cut_dets_0deg': [0.95, 1.04],
        '0deg_nr' : 178,
    }, 

    # 181 : {
    # 'low' : 4600,
    # 'high' : 6600,
    # ....
    #  


}


'''

Each item in the deconv_params dictionary is named after the run number. 
- 'low': Lower bound of the deconvolution range. In keV.
- 'high': Upper bound of the deconvolution range. In keV.
- 'beam': Approximate mean energy of the beam profile. In keV.
- 'cal': Calibration parameters of the measured spectra.
         Can be linear or quadratic funtion. First entry is the 
         offset, second the slope. 
- 'cut_dets_high': Upper limit of the prior from the top-down 
                   deconvolution. Needs to have the same length as 
                   the setup['detectors'] array.  
                   Given in percent with 
                   respect to the beam energy. 
- 'cut_dets_low':  Lower limit of the prior from the top-down 
                   deconvolution. Needs to have the same length as 
                   the setup['detectors'] array.  
                   Given in percent with 
                   respect to the beam energy. 
- 'cal_0deg': Calibration parameter for the 0-degree detector
- 'cut_dets_high': Upper limit of the prior from the top-down 
                   deconvolution. Given in percent with 
                   respect to the beam energy
- '0deg_nr': Run number of the 0-degree detector measurement

'''