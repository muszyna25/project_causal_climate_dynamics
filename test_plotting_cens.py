from warnings import filterwarnings
filterwarnings("ignore")

from src.DataProcessingClass import DataProcessingClass
from src.CausalityClass import CausalityClass


# Create a class instance for data preprocessing
dpc = DataProcessingClass()

# Load climate indices from netCDF files.
D = dpc.load_netcdf_files()

ca = CausalityClass(D)
dataframe = ca.test_syntethic_example()

# Build the causal graph with PCMCI algorithm
sig_causal_parents = ca.construct_causal_graph(dataframe)

# Fit the linear estimator
val_matrix, link_matrix = ca.linear_estimator(sig_causal_parents, dataframe)

# Plot causal effect network and save it to a png format
ca.plot_cens_graph(val_matrix, link_matrix, save=False)

'''
# Create a class instance for data preprocessing
dpc = DataProcessingClass()

# Load climate indices from netCDF files.
D = dpc.load_netcdf_files()

dpc.transform_precip(D, ind_name='precip', apply_transform=True)

# Select index
ind_name = 'wc'

dpc.detrend_index(D, ind_name)

# Remove means
dpc.remove_means(D)

# Apply min max normalisation
dpc.min_max_norm(D)

# Select datapoint for May-September from each year for 1940-2022
dpc.select_datapoints(D)

# Create a class instance with climate indices
ca = CausalityClass(D)

# subselect dict
from operator import itemgetter
ca.var_names = ['precip', 'wc']
D = dict(zip(ca.var_names, itemgetter(*ca.var_names)(D)))

# Construct a custom dataframe format used by Tigramite package
dataframe, var_names = ca.generate_dataframe(D) 

# Build the causal graph with PCMCI algorithm
sig_causal_parents = ca.construct_causal_graph(dataframe)

# Fit the linear estimator
val_matrix, link_matrix = ca.linear_estimator(sig_causal_parents, dataframe)

# Get the beta coefficients for the causal graph
ca.get_links_beta_coeffs()

ca.linear_mediator.get_coefs()

# Plot causal effect network and save it to a png format
ca.plot_cens_graph(val_matrix, link_matrix, save=False)

'''