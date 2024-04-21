from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI

#from tigramite.independence_tests import ParCorr
from tigramite.independence_tests.parcorr import ParCorr

from tigramite.models import LinearMediation
from matplotlib import pyplot as plt
import numpy as np

from tigramite.toymodels import structural_causal_processes as toys

'''
seed=111
random_state = np.random.default_rng(seed=seed)
data = random_state.standard_normal((500, 3))
for t in range(1, 500):
    data[t, 0] += 0.6*data[t-1, 0]
    data[t, 1] += 0.6*data[t-1, 1] - 0.36*data[t-2, 0]
    data[t, 2] += 0.6*data[t-1, 2]
            
var_names = [r'$X^0$', r'$X^1$', r'$X^2$']
dataframe = pp.DataFrame(data, var_names=var_names)
#'''
seed = 42
np.random.seed(seed)     # Fix random seed
def lin_f(x): return x
links_coeffs = {0: [((0, -1), 0.7, lin_f), ((1, -1), -0.8, lin_f)],
                1: [((1, -1), 0.8, lin_f), ((3, -1), 0.8, lin_f)],
                #2: [((2, -1), 0.5, lin_f), ((1, -2), 0.5, lin_f), ((3, -3), 0.6, lin_f)],
                2: [((2, -1), 0.5, lin_f), ((1, -2), 0.5, lin_f), ((3, -3), 0.6, lin_f)],
                3: [((3, -1), 0.4, lin_f)],
                }
T = 1000     # time series length
data, _ = toys.structural_causal_process(links_coeffs, T=T, seed=seed)
T, N = data.shape

# Initialize dataframe object, specify time axis and variable names
var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
dataframe = pp.DataFrame(data, 
                         datatime = {0:np.arange(len(data))}, 
                         var_names=var_names)
#'''
                         
parcorr = ParCorr(significance='analytic', mask_type=None)

pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

pcmci.verbosity = 0
    
results = pcmci.run_pcmci(tau_min=1, tau_max=1, pc_alpha=None)
print('results',results)

#print('p_,mattrix',results['p_matrix'])

q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
#print('q_matrix', q_matrix)

graph = pcmci.get_graph_from_pmatrix(results['p_matrix'], 0.05, 1, 1)

#sig_parents = pcmci.return_significant_links(pq_matrix=q_matrix, val_matrix=results['val_matrix'], alpha_level=0.3)
sig_parents = pcmci.return_parents_dict(graph=graph, val_matrix=results['val_matrix'])
print(sig_parents)
#all_parents = sig_parents['parents']
#link_matrix = sig_parents['link_matrix']

linear_mediator = LinearMediation(dataframe=dataframe, mask_type = 'y', data_transform = None)
linear_mediator.fit_model(all_parents = sig_parents, tau_max=1)  
val_matrix = linear_mediator.get_val_matrix()
print(val_matrix)

#fig = plt.figure(figsize=(8, 6), frameon=False)
#ax = fig.add_subplot(111, frame_on=False)
tp.plot_graph(graph=graph, val_matrix=val_matrix, var_names=var_names, node_aspect=0.5, node_size=0.5) #fig_ax=(fig,ax)
#tp.plot_graph(graph=results['graph'])#, node_aspect=0.07, node_size=5.0)#, val_matrix=val_matrix, var_names=var_names, node_aspect=0.1, node_size=5, show_colorbar=False)

plt.show()

'''
pcmci.print_significant_links(p_matrix = results['p_matrix'], q_matrix = q_matrix, val_matrix = results['val_matrix'], alpha_level = 0.05)
        
sig_parents = pcmci.return_significant_parents(pq_matrix=q_matrix, val_matrix=results['val_matrix'], alpha_level=0.05)

all_parents = sig_parents['parents']
link_matrix = sig_parents['link_matrix']

# Calculate beta coefficient
linear_mediator = LinearMediation(dataframe=dataframe, mask_type = 'y', data_transform = None)
linear_mediator.fit_model(all_parents = all_parents, tau_max=1)  

val_matrix = linear_mediator.get_val_matrix()

fig = plt.figure(figsize=(8, 6), frameon=False)
ax = fig.add_subplot(111, frame_on=False)
tp.plot_graph(val_matrix=val_matrix,link_matrix=link_matrix, var_names=var_names, fig_ax=(fig,ax))

plt.show()
'''
