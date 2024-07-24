import xarray as xr
import sys

file_ = sys.argv[1]

vars_ = ['u10m', 'v10m', 't2m', 'tp06']

ds = xr.open_dataset(file_)[vars_].isel(time=slice(0,7)).sel(lat=slice(47, 49)).sel(lon=slice(1.5, 3.5))
ds.to_netcdf(file_[:-3]+'_paris_36hr.nc')
