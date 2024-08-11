# import xarray as xr

# ds = xr.open_dataset('graphcast_2024_07_26_paris_36hr.nc')
# #print(ds)

# ds = xr.open_dataset('graphcast_2024_07_28_paris_36hr.nc')

# #print(ds)

# ds = xr.open_dataset('graphcast_2024_07_26_paris_36hr_pred.nc')

# print(ds)

import xarray as xr
import numpy as np
import subprocess

# Load the dataset
ds = xr.open_dataset('graphcast_2024_07_28_paris_36hr.nc')

# Define block size
block_size = 9

# Get the dimensions of the dataset
lat_dim, lon_dim = ds.dims['lat'], ds.dims['lon']

# Function to save a block as a new NetCDF file
def save_block(block, lat_idx, lon_idx):
    file_name = f'graphcast_2024_07_28_paris_36hr_block_{lat_idx}_{lon_idx}.nc'
    block.to_netcdf(file_name)
    print(f'Saved {file_name}')

# Loop over the dataset with a stride of 1
for lat_start in range(lat_dim - block_size + 1):
    for lon_start in range(lon_dim - block_size + 1):
        lat_end = lat_start + block_size + 1
        lon_end = lon_start + block_size + 1
        
        block = ds.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end))
        save_block(block, lat_start, lon_start)
        
        # Optionally run inference.py on each new NetCDF file
        subprocess.run(['python', 'inference.py', f'graphcast_2024_07_28_paris_36hr_block_{lat_start}_{lon_start}.nc'])