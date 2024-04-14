from glob import glob
from os.path import basename
from matplotlib import pyplot as plt
from xarray import open_dataset
import numpy as np
import os
import xarray as xr

#...........................
def get_list_of_files(path):
        # find netcdf files only
        pattern = "*.nc"
        # split file names by '_'
        elem = '_' 
        #path = os.path.join(self.dir_path, self.dataset, self.variable, 'jjas',"")
        #path = os.path.join(self.dir_path, self.dataset, self.variable, self.part, "") 
        print(path)

        l_files = [os.path.basename(x) for x in glob.glob(path + pattern)]
        # sort all file names by given element, for example, xxx_xxx_0000.nc 
        l_files = sorted(l_files, key=lambda x: x.split(elem)[-1])
        #l_files = sorted(l_files, key=lambda x: x.split(elem)[-2])
        print('List:', l_files)

        return l_files

 #...........................
def load_pl_netcdf_file(path, levels, lfs):

        # Load all netcdf files in a givend dir.
        L = []
        for pl in levels:
            D = []
            for f in lfs:
                # Read netcdf file and extract precipitation fields
                print('Loading:', f)
                #print(self.dir_path, self.dataset, self.variable, f, "")
                #path = os.path.join(self.dir_path, self.dataset, self.variable, self.part, f)
                #print(path)
                ds = xr.open_dataset(path)

                if not levels:
                    da = ds
                else:
                    da = ds.isel(level=pl)
                    #da = ds.isel(pressure=pl)
                    print('Loading levels: ', pl)
                    dai= da.sel(time=da.time.dt.month.isin(self.timespan))
                    print('Selected data', dai)
                D.append(dai)
                ds.close()

            print('D list:', D[:3])
            X = xr.concat(D, dim='time')
            print('Concat: ', X)
            L.append(X)
        #Z = xr.align(L[0],L[1])
        return L
