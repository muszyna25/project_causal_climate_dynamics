from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
from tigramite.models import LinearMediation
from matplotlib import pyplot as plt
import numpy as np

seed=1
random_state = np.random.default_rng(seed=seed)
data = random_state.standard_normal((500, 2))
for t in range(1, 500):
    data[t, 0] += 0.6*data[t-1, 0]
    data[t, 1] += 0.6*data[t-1, 1] - 0.36*data[t-2, 0]
            
var_names = [r'$X^0$', r'$X^1$']
dataframe = pp.DataFrame(data, var_names=var_names)

parcorr = ParCorr(significance='analytic', mask_type=None)

pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

pcmci.verbosity = 0
        
results = pcmci.run_pcmci(tau_min=1, tau_max=1, pc_alpha=None)

q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

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
