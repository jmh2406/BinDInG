## BinDInG
BinDInG- Bin-wise Determination of Incident Gammas

The BinDIng code allows the bin-wise reconstruction of incident photons based on a given set of measured spectra  
and detector response matrices based on Bayesian statistics via [PyMC](https://www.pymc.io/projects/docs/en/stable/learn.html). 
Mulitple detectors of arbitrary resolution and with different decay contributions can be implemented and the subtraction of 
contaminations from $(n,\gamma)$-lines or other NRF lines (e.g. 16O) is feasible. Contributions from branching transitions 
to low-lying excited states of the nucleus of interest are included in the model as well. 
Because the spectra are deconvolved bin-wise, no assumption about the shape of the beam profile is needed. 
Albeit the large model space, compilation times are quite short with 5-10 minutes for a typical deconvolution, which is 
due to the fact that top-down deconvolutions are used to determine the priors for the forward folding model. 
The background is modeled with constant background based on the highest-energy bins of the measured spectrum in 
combination with an atomic scattering background model.  

# Installation and Setup

