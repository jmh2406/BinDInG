#!/usr/bin/env python
from binding_bins import binding


a = binding(project = 'Example', 
            runnr = 179, 
            savename = 'bins',  
            bin_width = 10, 
            )
a.sample_model()
