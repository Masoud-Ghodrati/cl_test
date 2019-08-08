# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:30:17 2019

@author: masoudg
"""
# import some packages
import matplotlib.pyplot as plt
import numpy as np
import re

#%% set some pathes
result_path = 'C:/MyFolder/coles_test'

def read_errorfile(result_path, file_name):
    
    # Write errors to file
    with open(result_path + '/' + file_name) as file:
        er = file.read().split(',')
        nwrmsle  = []
        rsquared = []
        for ind, line in enumerate(er):
            val = float(re.findall('\d*\.?\d+', line )[0])
            if ind%2 ==0:
                rsquared.append(val)
            else:
                nwrmsle.append(val)
                
    rsquared = np.array(rsquared)
    nwrmsle  = np.array(nwrmsle)
    return rsquared, nwrmsle

#%% ploting
n_months    = 10
n_days      = 15
train_weeks = np.arange(n_days, n_months*30, n_days)/7;

# plot the results
_, nwrmsle = read_errorfile(result_path, 'errors_rf_reg.txt')

fig, ax    = plt.subplots(nrows=1, ncols=1, sharex='col', sharey='row', figsize=(20, 7))
ax         = plt.subplot(1, 2, 1)
plt.plot(train_weeks, nwrmsle[::4], 'b', linewidth=2)
plt.xlabel('# weeks trained')
plt.ylabel('nwrmsle')
plt.title('Random forest')

#_, nwrmsle = read_errorfile(result_path, 'errors_lin_reg.txt')
#ax         = plt.subplot(1, 2, 2)
#plt.plot(train_weeks, nwrmsle[::4])
#plt.xlabel('# weeks trained')
#plt.ylabel('nwrmsle')
#plt.title('Linear regression')
#
fig.savefig(result_path + '/scores')



