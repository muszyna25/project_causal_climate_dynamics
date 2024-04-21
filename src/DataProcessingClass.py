import os
import numpy as np
from glob import glob
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from xarray import open_dataset
from yaml import safe_load
from scipy import signal
import cartopy.crs as ccrs
import cartopy as cart
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

class DataProcessingClass:
    """ DataProcessingClass

        This is the class that implements all functions needed to read/preprocess climate indices.
        It also includes functions that perform: time series detrending, statistical testing for stationarity
        of time series data, etc.
    """

    ###############################################
    def __init__(self):
        """__init__ 
        
        Creates instances of the class and initilises their atributes.
        
        """

        self.path = self.read_yaml_file()['path']
        self.indexes = sorted(self.read_yaml_file()['indexes'], key=lambda x: x)
        self.sign_lev = self.read_yaml_file()['significance_level']
        self.nmonths = self.read_yaml_file()['nmonths']
        self.range_months = self.read_yaml_file()['range_months']
        self.n_breakup_months = self.read_yaml_file()['n_breakup_months']
        self.save_path = self.read_yaml_file()['save_path']
        self.box_hp_wc = self.read_yaml_file()['box_hp_wc']
        self.box_mhc = self.read_yaml_file()['box_mhc']
        self.region = self.read_yaml_file()['region']

    ###############################################
    def read_yaml_file(self):
        """read_yaml_file 

        This function reads in the program configuration from yaml file.

        Returns
        -------
        A string of objects
            Returns a string of objects read fromt the yaml file.
        """

        with open('config.yml', 'r') as f:
            config = safe_load(f)
        return config
    
    ###############################################
    def get_list_of_files(self, path, pattern='*.nc'):
        """get_list_of_files

        This function finds netcdf files in the dir
        and returns them alphabetically sorted.

        Parameters
        ----------
        path : str
            A path to a dir.
        pattern : str, optional
            A type of file as a regular expression, by default '*.nc'

        Returns
        -------
        A list of strings
            Returns a list of file names. 
        """

        print('Input dir path:', self.path)
        print('Selected indices: ', self.indexes)

        l_files = [os.path.basename(x) for x in glob(path + pattern)]
        l_files = sorted(l_files, key=lambda x: x)
        print(l_files)

        new_l_files = []
        for fn in l_files:
            print(os.path.splitext(fn)[0])
            if os.path.splitext(fn)[0] not in self.indexes:
                print('not in', fn)
            else:
                new_l_files.append(fn)

        print('List of files', *new_l_files, sep='\n')

        return new_l_files
    
    ###############################################
    def load_netcdf_files(self):
        """load_netcdf_files 

        This function loads netcdf files from the dir
        and returns a dictionary contating them.

        Returns
        -------
        dict
            Returns a dictionary.
        """

        l_fns = self.get_list_of_files(self.path)
        data = [open_dataset(self.path + f, engine='netcdf4') for f in l_fns]
        D = dict(map(lambda i,j : (i,j) , self.indexes,data))

        return D
    
    ###############################################
    def detrend_index(self, D, ind_name):
        """detrend_index 

        This function remove a linear trend from 1D time series 
        using scipy.signal.detrend().

        Parameters
        ----------
        D : dict
            A dictionary containing climate indices.
        ind_name : str
            A name of an index.
        n_months : int
            A length of a time between breakup points for scipy.signal.detrend().
        """

        tmp_var = D[ind_name].variables[ind_name].values

        breakps = [b for b in range(0, tmp_var.shape[0], self.n_breakup_months)]

        D[ind_name].variables[ind_name].values = signal.detrend(tmp_var, axis=-1, type='linear', bp=breakps)

    ###############################################
    def transform_precip(self, D, ind_name, apply_transform=False, show=False):
        """transform_precip 

        This function applies the logarithmic transformation to precipitation-based index,
        and it plots a histogram of the index data.

        Parameters
        ----------
        D : dict
            A dictionary containing climate indices.
        ind_name : str
            A name of an index.
        apply_transform : bool, optional
            A flag whether to apply the logarithmic transformation to precipitation-based index, by default False.
        """
    
        tmp_var = D[ind_name].variables[ind_name].values

        if apply_transform:

            # Log transform
            tp_max = np.max(tmp_var)
            tmp_log_var = np.log(tmp_var*(np.e-1)/tp_max + 1)

            if show:
                _, axs = plt.subplots(2, 1)
                axs[0].hist(tmp_var, label='Original ' + ind_name)
                axs[0].legend()

                axs[1].hist(tmp_log_var, label= ind_name + ' after log transform')
                axs[1].legend()
                plt.show()

            D[ind_name].variables[ind_name].values = tmp_log_var
        else:
            plt.hist(tmp_var, label='Original\t' + ind_name)
            plt.legend()
            plt.show()

    ###############################################
    def adf_kpss_tests(self, D):
        """adf_kpss_tests 

        This function uses the Augmented Dickey-Fuller unit root test
        and Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
        Both tests use statsmodels.tsa.stattools package.

        Parameters
        ----------
        D : dict
            A dictionary containing climate indices.
        """

        for k in list(D.keys()):
            print('Index:', k)
            dftest1 = adfuller(D[k].variables[k].values, store=True)
            dftest2 = kpss(D[k].variables[k].values, nlags='legacy', store=True)

            # Combined tests for stationarity of an index.
            if dftest1[1] < self.sign_lev  and dftest2[1] > self.sign_lev :
                print('Significance level:', self.sign_lev )
                print('ADF test: test statistics; p-value: ', round(dftest1[0],30), round(dftest1[1],30))
                print('KPSS test: test statistics; p-value: ', round(dftest2[0],30), dftest2[1])
                print('Yes, the index is stationary.\n')
            else:
                print('No, the index is not stationary.\n')

    ###############################################        
    def remove_means(self, D):
        """remove_means 

        This function removes climatological and seasonal means from climate indices.

        Parameters
        ----------
        D : dict
            A dictionary containing climate indices.
        """

        _n_months = self.nmonths
        d_keys = list(D.keys())
        
        # Remove climatological means
        for i in range(_n_months):
            for k in d_keys:
                tmp = D[k].variables[k].values[i::_n_months]
                D[k].variables[k].values[i::_n_months] = tmp - np.mean(tmp, axis = 0)
        
        ## Remove seasonal mean
        n_ind_size = D[d_keys[0]].variables[d_keys[0]].values.shape[0]
        mons = self.range_months

        for j in range(0, n_ind_size, _n_months):
            for k in d_keys:
                mu_tmp = np.mean(D[k].variables[k].values[j+mons[0]:j+mons[1]], axis=0)
                D[k].variables[k].values[j+mons[0]:j+mons[1]] = D[k].variables[k].values[j+mons[0]:j+mons[1]] - mu_tmp
        
    ###############################################
    def min_max_norm(self, D):
        """min_max_norm 

        Min-max normalisation of climate indices to a common range.

        Parameters
        ----------
        D : dict
            A dictionary containing climate indices.
        """
        
        d_keys = list(D.keys())
        for k in d_keys:
            tmp = D[k].variables[k].values
            D[k].variables[k].values = (2.0 * (tmp - np.amin(tmp))/(np.amax(tmp) - np.amin(tmp))) - 1.0

    ###############################################        
    def select_datapoints(self, D):
        """select_datapoints 

        This function selects months from each index, e.g. MJJAS.

        Parameters
        ----------
        D : dict
            A dictionary containing climate indices.
        """

        months = list(range(self.range_months[0], self.range_months[1] + 1))
        d_keys = list(D.keys())
        k_enso = 'enso'

        for k in d_keys:
            if k != k_enso:
                tmp = D[k]
                D[k] = tmp.sel(time=tmp.time.dt.month.isin(months))
            else: D[k] = D[k_enso].sel(index=D[k_enso].index.dt.month.isin(months))

    ###############################################
    def test_pairs_lag_corr(self, D, n_lags=-1):
        """test_pairs_lag_corr 

        This function tests pairs of indices for direct associations.

        Parameters
        ----------
        D : dict
            A dictionary containing climate indices.
        n_lags : int, optional
            Number of lags., by default -1
        """
        
        d_keys = list(D.keys())
        for k1 in d_keys:
            x = D[k1].variables[k1].values
            for k2 in d_keys:
                if k2 != k1:
                    y = D[k2].variables[k2].values 
                    print('Pair:', k1 + '-->' + k2)
                    cc = self.lag_cross_corr(x, y, n_lags)
                    print('\n')

    ###############################################
    def lag_cross_corr(self, x, y, lag=-1):
        """lag_corr 

        This function calculates lagged (Pearson) cross-correlation
        coefficients between two time series.

        Parameters
        ----------
        x : numpy array
            A leading time series.
        y : numpy array
            A lagging time series.
        lag : int, optional
            Number of lags., by default -1

        Returns
        -------
        A list of lagged (Pearson) cross-correlatio coefficients.
        """

        if len(x)!=len(y):
            raise('Indices are of different lengths.')

        if np.isscalar(lag):
            if abs(lag) >= len(x):
                raise('Maximum lag equal or larger than the index size.')
            if lag<0:
                lag=-np.arange(abs(lag)+1)
            elif lag==0:
                lag=[0,]
            else:
                lag=np.arange(lag+1)    
        else:
            lag=np.asarray(lag)

        ccs=[]
        print('Computing lagged cross-correlations at lags:', lag)
        for i in lag:
            if i < 0:
                cc = pearsonr(x[:i],y[-i:])
                print(i, cc)
                ccs.append(cc)
            elif i == 0:
                cc = pearsonr(x,y)
                print(i, cc)
                ccs.append(cc)
            elif i > 0:
                cc = pearsonr(x[i:],y[:-i])
                print(i, cc)
                ccs.append(cc)

        ccs = np.asarray(ccs)

        return ccs

    ######################################
    def plot_orography_map(self, fn, save=True):
        """plot_orography_map _summary_

        This function plots boxes associated with SMHP/RWC and MHC indices
        over the ERA5 orography.

        Parameters
        ----------
        save : bool, optional
            _description_, by default True
        """
        
        da = open_dataset(self.path + 'orography/era5_invariant_orog.nc')
        print(da)

        fig = plt.figure(figsize=(10,6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.coastlines()
        ax.coastlines('10m')
        ax.set_aspect('equal')    
        ax.set_extent(self.region)

        das = da.orog.where(da.orog >= 0)
        
        p = das.plot(ax=ax, levels=18, robust=True, add_colorbar=False, cmap='YlOrBr')

        da.orog.plot(ax=ax, 
                     levels=18, 
                     robust=True, 
                     transform=ccrs.PlateCarree(), 
                     add_colorbar=False, 
                     cmap='YlOrBr')
        
        cb = plt.colorbar(p, 
                          orientation="horizontal", 
                          pad=0.07, 
                          shrink=0.6, 
                          aspect=20, 
                          extend='max', 
                          spacing='proportional')
        
        cb.set_label(label='Elevation (m)', size='large', weight='bold')
        cb.ax.tick_params(labelsize='large')
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, zorder=115)
        gl.xlines = True
        gl.ylines = True
        gl.top_labels = True
        gl.right_labels = True
        gl.left_labels = True
        
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}

        ### Plot boxes over the region
        
        box = self.box_hp_wc
        b = patches.Rectangle((box[2], box[0]), 
                              abs(box[2] - box[3]), 
                              abs(box[1] - box[0]), 
                              facecolor='None', 
                              edgecolor='w', 
                              linewidth=3, 
                              linestyle='--' , 
                              zorder=102)
        ax.add_patch(b)
        
        box1 = self.box_mhc
        b1 = patches.Rectangle((box1[2], box1[0]), 
                               abs(box1[2] - box1[3]), 
                               abs(box1[1] - box1[0]), 
                               facecolor='None', 
                               edgecolor='lime', 
                               linewidth=3, 
                               linestyle='--', 
                               zorder=101)
        ax.add_patch(b1)

        ax.add_feature(cart.feature.OCEAN, zorder=100, edgecolor='k')

        left, width = 0.17, 0.45
        bottom, height = 0.25, 1.31
        right = left + width
        top = bottom + height
        ax.text(0.5*(left+right), 
                0.5*(bottom+top), 
                s='SMHP & RWC',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14, 
                color='w',
                transform=ax.transAxes, 
                weight='bold', 
                zorder=105)
        
        left1, width1 = 0.22, 0.52
        bottom1, height1 = 0.25, 0.35
        right1 = left1 + width1
        top1 = bottom1 + height1
        ax.text(0.5*(left1+right1), 
                0.5*(bottom1+top1), 
                s='MHC',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14, 
                color='lime',
                transform=ax.transAxes, 
                weight='bold', 
                zorder=105)
        
        ax.set_title('')
        plt.show()
        
        if save == True:
            fig.savefig(self.save_path + fn + '.png', dpi=300, bbox_inches='tight', transparent=True)
        
    ######################################
    def get_links_formatted(self):
        """get_links_formatted

        This function produces a readable format of causal links from tigramites format.

        Returns
        -------
        dict
            Readable format of causal links.
        """

        fn = [os.path.basename(x) for x in glob(self.path + '*.npz')][0]
        fd = np.load(self.path+fn)
        all_data = [fd[key] for key in fd]
        data = all_data[2]
        names = all_data[1]

        #Get all connection x:abc --> y:efg
        beg = list(range(0,len(names))) # from
        end = list(range(0,len(names))) # to

        links = {}
        maxlag = 1
        sym = '\n \u2191 \n'
        for i in beg:
            for j in end:
                if i == j:
                    continue
                v = data[:,j,i,maxlag]
                if all(v == 0): continue
                k = names[i] + sym + names[j] # i->j coefficient of the connection
                v = np.delete(v, np.where(v == 0.0))
                links[k] = v
        
        return links
    
    ######################################
    def plot_coeffs_distribution(self, links, fn, save=False):
        """plot_coeffs_distribution

        This function plots box plots for path coeffs. 

        Parameters
        ----------
        links : dict
            Readable format of causal links.
        fn : str
            A npz file name.
        save : bool, optional
            _description_, by default False
        """
    
        fig, axs = plt.subplots(1, 1, figsize=(11,8))
        axs.set_ylim(-0.6,0.6)
        sh = 0.50
        _labels, vals = list(links.keys()), links.values() 
            
        all_vals = [v for v in vals]
        for av, l in zip(all_vals, _labels):
            print('mean value:', l, np.mean(av), sep='\n')

        medianprops = dict(linestyle='-.', linewidth=0.001)

        axs.boxplot(x=all_vals, 
                        positions= np.asarray(list(range(0,len(all_vals)))) + sh, 
                        widths=0.5, 
                        showmeans=True, 
                        whis=(0,100), 
                        meanprops=dict(markerfacecolor='grey', markeredgecolor='k'), 
                        patch_artist=True, 
                        boxprops=dict(facecolor='b', color='k'),
                        medianprops=medianprops)
          
        axs.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.9)
        
        axs.set_xticks(list(range(0,len(_labels))))

        axs.set_xticklabels(_labels,fontdict={'fontsize':16}, 
                            transform=Affine2D().translate(0.5,-0.62) + axs.transData)
        
        axs.set_xlabel('Causal Links', fontsize=16)
        axs.set_ylabel('Avg. Path Coeff.', fontsize=16)
        axs.tick_params(labelsize=16)

        plt.axhline(0.0,color = 'y',linewidth=1.5, zorder=0, linestyle='dashed')
        plt.tight_layout()  
        
        if save == True:
            print(self.save_path + fn + '.png')
            fig.savefig(self.save_path + fn + '.png', dpi=300, bbox_inches='tight', transparent=True)
    
        plt.show()

    ######################################
    def plot_metrics_chart_plots(self, fname, save=False):
        """plot_metrics_chart_plots

        This function plots averages of two causal metrics:
        Average Causal Effect, Average Causal Susceptibility

        Parameters
        ----------
        fname : str
            A npz file name.
        save : bool, optional
            _description_, by default False
        """

        fn = [os.path.basename(x) for x in glob(self.path + '*.npz')][0]
        
        # Calculated metrics' names
        metrics = ['Average Causal Effect', 'Average Causal Susceptibility']
        n_entries=5
        for mes_i in range(3,n_entries):
            all_reanalyses = []

            fd = np.load(self.path+fn)
            all_data = [fd[key] for key in fd]
            m = all_data[mes_i]
            labels = all_data[1]

            data = np.mean(m,axis=0)
            print('mean values:', tuple(zip(labels, data)))
            all_reanalyses.append(data)
    
            fig, _axs = plt.subplots(1, 2, figsize=(13,9))
            axs = _axs.flatten()
            axs[1].set_visible(False)
            
            for i in range(0, len(axs)-1):
                x = np.arange(0,3*len(labels),3)  # the label locations
                width = 0.85  # the width of the bars
                print(np.round(all_reanalyses,10))
                
                if i == 0:
                    axs[i].bar(x, 
                               np.round(all_reanalyses[0],2), 
                               width,  
                               edgecolor='black', 
                               color='g')
                    
                    axs[i].set_ylabel(metrics[mes_i-3], fontsize=16)

                axs[i].set_axisbelow(True)
                axs[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.9)
                axs[i].set_xticks(x)
                axs[i].set_xticklabels(labels,fontsize=16)
                axs[i].tick_params(axis='y', labelsize=16)
                axs[i].tick_params(axis='x', labelsize=16)

                if mes_i == 3: 
                    axs[i].set_yticks(np.arange(0, 0.22, 0.05), minor=False)
                    axs[i].set_ylim(0.0,0.2) #ace
                else:
                    axs[i].set_yticks(np.arange(0, 0.35, 0.05), minor=False)
                    axs[i].set_ylim(0.0,0.3) #acs
                
            plt.xticks(size = 14)
            plt.subplots_adjust(wspace=0.23, hspace=0.3)

            if save == True:
                fig.savefig(self.save_path + fname + metrics[mes_i-2] + '.png', dpi=300, bbox_inches='tight', transparent=True)

            plt.show()