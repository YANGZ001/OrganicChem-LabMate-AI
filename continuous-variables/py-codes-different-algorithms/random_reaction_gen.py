import numpy as np
import pandas as pd
import itertools


'''
Section below creates lists for your reaction parameters. Change names of lists where appropriate
'''
#Chemical space 1
#For bigger lists use np.arange(min_value, max_value, step)
naphthalen2yl_trifluoromethanesulfonate = [0.15] # in mmol
_2methylcyclohexan1one = [0.15, 0.25, 0.35, 0.45] # in mmol
Ligand_to_metal = [0.9, 1.05, 1.2, 1.35, 1.5]# No unit
# Ligand_to_metal
Cat_loading = [5, 7.5, 10, 12.5 , 15] #mol%
Base = [0.75, 1, 1.25, 1.5, 2] #equiv
i_PrOH = [1, 1.2, 1.4, 1.6, 1.8] #equiv.
Time = [14, 16, 18, 20, 22] # in hrs
Temperature = [110, 120, 130, 140, 150] # in C
concentration = [0.25, 0.35, 0.45, 0.55, 0.65] # in M

#Chemical space 2 extended
#For bigger lists use np.arange(min_value, max_value, step)
naphthalen2yl_trifluoromethanesulfonate = [0.15] # in mmol
_2methylcyclohexan1one = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75] # in mmol
Ligand_to_metal = [0.9, 1.05, 1.2, 1.35, 1.5]# No unit
# Ligand_to_metal
Cat_loading = [5, 7.5, 10, 12.5 , 15] #mol%
Base = [0.75, 1, 1.25, 1.5, 2, 3, 4, 5] #equiv
i_PrOH = [1, 1.2, 1.4, 1.6, 1.8, 2.5, 5, 7.5 ,10] #equiv.
Time = [14, 16, 18, 20, 22] # in hrs
Temperature = [110, 120, 130, 140, 150] # in C
concentration = [0.25, 0.35, 0.45, 0.55, 0.65] # in M




'''
The following lines create all combos possible for the values listed above and saves as text file. Change names where appropriate.
'''
combinations = list(itertools.product(naphthalen2yl_trifluoromethanesulfonate, _2methylcyclohexan1one, Ligand_to_metal, Cat_loading, Base, i_PrOH, Time, Temperature,concentration ))
df = pd.DataFrame(combinations)
df.to_csv('all_combos_yz.csv', header = ['nap (mmol)', '2meth (mmol)', 'Ligan2Metal', 'Cat (mol%)', 'base (equiv.)', 'i_PrOH (equiv)', 'Time (Hrs)', 'Temperature (C)', 'concentration (M)' ])
print('its done!')
#
# '''
# Below, 10 random reaction are selected from all possible combinations. The reactions are stored in a text file. Change names of header as appropriate.
# '''
#
# random_data = df.sample(n=10, random_state=1)
# df_random_data = pd.DataFrame(random_data)
# df_random_data.to_csv('train_data_yz.csv', header = ['nap (mmol)', '2meth (mmol)', 'Ligan2Metal', 'Cat (mol%)', 'base (equiv.)', 'i_PrOH (equiv)', 'Time (Hrs)', 'Temperature (C)', 'concentration (M)'])
