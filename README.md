## BinDInG
BinDInG- Bin-wise Determination of Incident Gammas

### General description 
The BinDIng code allows the bin-wise reconstruction of incident photons based on a given set of measured spectra  
and detector response matrices based on Bayesian statistics via [PyMC](https://www.pymc.io/projects/docs/en/stable/learn.html). 
Mulitple detectors of arbitrary resolution and with different decay contributions can be implemented and the subtraction of 
contaminations from $(n,\gamma)$-lines or other NRF lines (e.g. $^\mathrm{16}\mathrm{O}$) is feasible. Contributions from branching transitions  
to low-lying excited states of the nucleus of interest are included in the model as well. 
Because the spectra are deconvolved bin-wise, no assumption about the shape of the beam profile is needed. 
Albeit the large model space, compilation times are quite short with 5-10 minutes for a typical deconvolution, which is 
due to the fact that top-down deconvolutions are used to determine the priors for the forward folding model. 
The background is modeled with constant background based on the highest-energy bins of the measured spectrum in 
combination with an atomic scattering background model.  

### Structure 

BinDInG consists of the 'src' directory that contains a 'scripts' and 'projects' directory. 
The 'scripts' directory contains all the scripts and executables for using the code. 
The 'projects' directory can be used to setup new deconvolution projects. 
Per default, only an 'Example' directory is contained here. 
The content of the project directories is explained below. 

Every file created by the code can be found in the 'gen' directory next to the 'src' directory.
Here, the created files are sorted with respect to the project they belong to. 

### Installation and Setup

Installation should simply require executing:
```python 
    pip install git+https://github.com/jmh2406/BinDInG.git
```

You can execute 

```bash 
    ./make_new_project.sh <new_project>
```

to create a new project directory. A project directory consists of three files, setup.py, deconv_params.py and paths.py. 
All three files need to be adjusted to the needs of your deconvolution. 

In the paths.py file, the path to your measured spectra and detector response matrices needs to be defined. Also the structure and naming logic of your detector response matrices and spectra can be adjusted here. 

The setup.py file contains a dictionary with parameters that are constant for an entire project, e.g. the detector names or 
the excited states of the nucleus of interest. 

The deconv_params.py file contains all the parameters that can change from run to run. The name of each dictionary item is
the runnumber. 

Deconvolutions can be executed with the exe_binding.py executable. Change the parameters of the 
binding class in the exe_binding.py executable to vary which run and with which options you want run. 

View the results of your deconvolution in the deconv_playground.ipynb notebook. Here, top-down deconvolutions can be 
conducted as well, in order to test the calibration of your measured spectra and the validity of your detector response matrices. 

For questions, you can contact Julian Hauf (jhauf@ikp.tu-darmstadt.de)




