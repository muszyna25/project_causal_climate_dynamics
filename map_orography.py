from warnings import filterwarnings
filterwarnings("ignore")

from src.DataProcessingClass import DataProcessingClass
from src.CausalityClass import CausalityClass


# Create a class instance for data preprocessing
dpc = DataProcessingClass()

# Load climate indices from netCDF files.
#D = dpc.load_netcdf_files()

dpc.plot_orography_map(save=True)

