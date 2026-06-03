#!/usr/bin/env python
from binding_bins import binding
from deconv_params import deconv_params

# runnrs = [176, 179, 181, 186, 216, 219, 221, 225, 228]

# for j in runnrs: 
#     print(f'Run number: {j}')
#     a = binding(j, 'bins',  10, branchings=True, O_cont = False, ng_cont= False)
#     a.sample_model()

a = binding(193, 'bins',  10, branchings=True, O_cont = True, ng_cont= True)
a.sample_model()
