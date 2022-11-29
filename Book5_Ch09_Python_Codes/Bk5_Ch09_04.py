

###############
# Authored by Weisheng Jiang
# Book 5  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

num_data  = 500
# generate random data that follows normal distribution
data = np.random.normal(loc=-2, scale=3, size=(num_data,1))
loc_array = np.cumsum(np.ones_like(data))

mu_data = data.mean()
sigma_data = data.std()

for sigma_band_factor in [1,2]:

    inside_data  = np.copy(data)
    outside_data = np.copy(data)
    
    plus_sigma   = mu_data + sigma_band_factor*sigma_data
    minus_sigma  = mu_data - sigma_band_factor*sigma_data

    outside_data[(outside_data >= minus_sigma) & (outside_data <= plus_sigma)] = np.nan
    inside_data[(inside_data >= plus_sigma) | (inside_data <= minus_sigma)] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), gridspec_kw={'width_ratios': [3, 1]})
    
    ax1.plot(loc_array, inside_data,  marker = '.', color = 'b', linestyle = 'None')
    ax1.plot(loc_array, outside_data, marker = 'x', color = 'r', linestyle = 'None')

    ax1.fill_between(loc_array, 0*loc_array + plus_sigma, 0*loc_array + minus_sigma, color = '#DBEEF4')

    ax1.axhline(y = mu_data,     color = 'r', linestyle = '--')
    ax1.axhline(y = plus_sigma,  color = 'r', linestyle = '--')
    ax1.axhline(y = minus_sigma, color = 'r', linestyle = '--')
    ax1.set_ylim([np.floor(data.min()) - 1,np.ceil(data.max()) + 1])
    ax1.set_xlim([loc_array.min(),loc_array.max()])
    
    ax2 = sns.distplot(data, rug=True, rug_kws={"color": "k", 
                                                    "alpha": 0.5, 
                                                    "height": 0.06, 
                                                    "lw": 0.5}, 
                       vertical=True, bins = 15)

    ax2.set_ylim([np.floor(data.min()) - 1,np.ceil(data.max()) + 1])
    ax2.axhline(y = mu_data,     color = 'r', linestyle = '--')
    ax2.axhline(y = plus_sigma,  color = 'r', linestyle = '--')
    ax2.axhline(y = minus_sigma, color = 'r', linestyle = '--')
    print(np.count_nonzero(~np.isnan(outside_data)))
    
