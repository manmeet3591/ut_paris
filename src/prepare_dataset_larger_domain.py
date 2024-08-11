import xarray as xr
import os
import glob
import sys
import numpy as np
path = '/mnt/washington/'

combine_data = True

if combine_data:
    ds_combined = xr.open_mfdataset('noaa_aorc_washington_????.nc', chunks={'time': 100})
    ds_input_combined = xr.open_mfdataset('/mnt/washington_larger/graphcast_????_??_??_washington_36hr.nc', combine='nested', concat_dim='new_dim')
    #ds_combined.to_netcdf('noaa_aorc_washington_APCP_surface_2015_2023.nc')
    print('Read the data successfully')



global_min_input = {'u10m': -21.397995, 'v10m': -21.892258, 't2m': 240.48112, 'tp06': -0.00036333402, 'q1000': -0.00010691816}
global_max_input = {'u10m': 20.468126, 'v10m': 22.318855, 't2m': 310.62543, 'tp06': 0.08382863, 'q1000': 0.023028348}

# global_min_input['q1000'] = ds_input_combined['q1000'].compute()
# global_max_input['q1000'] = ds_input_combined['q1000'].compute()

# print(ds_input_combined['tp06'].min(skipna=True).values)
# print(ds_input_combined['tp06'].max(skipna=True).values)

# print(ds_input_combined['t2m'].min(skipna=True).values)
# print(ds_input_combined['t2m'].max(skipna=True).values)

# print(ds_input_combined['u10m'].min(skipna=True).values)
# print(ds_input_combined['u10m'].max(skipna=True).values)

# print(ds_input_combined['v10m'].min(skipna=True).values)
# print(ds_input_combined['v10m'].max(skipna=True).values)


# sys.exit()
# Load the AORC datasets
ds_aorc_apcp = ds_combined #xr.open_dataset('/mnt/noaa_aorc_washington_APCP_surface_2017_2023.nc')#.sel(time=slice(start_time, end_time))
ds_aorc_t2m   = ds_combined #xr.open_dataset('/mnt/noaa_aorc_washington_TMP_2maboveground_2017_2023.nc')#.sel(time=slice(start_time, end_time))
ds_aorc_u10  = ds_combined # xr.open_dataset('/mnt/noaa_aorc_washington_UGRD_10maboveground_2017_2023.nc')#.sel(time=slice(start_time, end_time))
ds_aorc_v10  = ds_combined # xr.open_dataset('/mnt/noaa_aorc_washington_VGRD_10maboveground_2017_2023.nc')#.sel(time=slice(start_time, end_time))
ds_aorc_q100 = ds_combined

global_min_target = {'APCP_surface': 0.0, 'TMP_2maboveground': 236.50000352412462, 'UGRD_10maboveground': -24.2000003606081, 'VGRD_10maboveground': -33.80000050365925, 'SPFH_2maboveground': 9.999999747378752e-05}
global_max_target = {'APCP_surface': 221.30000329762697, 'TMP_2maboveground': 313.60000467300415, 'UGRD_10maboveground': 29.700000442564487, 'VGRD_10maboveground': 33.5000004991889, 'SPFH_2maboveground': 0.029199999262345955}


# Update global min and max for target data
# global_min_target['APCP_surface'] = min(global_min_target['APCP_surface'], ds_aorc_apcp['APCP_surface'].min().item())
# global_max_target['APCP_surface'] = max(global_max_target['APCP_surface'], ds_aorc_apcp['APCP_surface'].max().item())
# global_min_target['TMP_2maboveground'] = min(global_min_target['TMP_2maboveground'], ds_aorc_t2m['TMP_2maboveground'].min().item())
# global_max_target['TMP_2maboveground'] = max(global_max_target['TMP_2maboveground'], ds_aorc_t2m['TMP_2maboveground'].max().item())
# global_min_target['UGRD_10maboveground'] = min(global_min_target['UGRD_10maboveground'], ds_aorc_u10['UGRD_10maboveground'].min().item())
# global_max_target['UGRD_10maboveground'] = max(global_max_target['UGRD_10maboveground'], ds_aorc_u10['UGRD_10maboveground'].max().item())
# global_min_target['VGRD_10maboveground'] = min(global_min_target['VGRD_10maboveground'], ds_aorc_v10['VGRD_10maboveground'].min().item())
# global_max_target['VGRD_10maboveground'] = max(global_max_target['VGRD_10maboveground'], ds_aorc_v10['VGRD_10maboveground'].max().item())

# global_min_target['SPFH_2maboveground'] =  ds_combined['SPFH_2maboveground'].compute()
# global_max_target['SPFH_2maboveground'] = ds_combined['SPFH_2maboveground'].max()

print(global_min_input, global_max_input, global_min_target, global_max_target)



# Define the path and the pattern
path = '/mnt/washington_larger/'  # Ensure the path is absolute
pattern = 'graphcast_????_??_??_washington_36hr.nc'

# Construct the full pattern
full_pattern = os.path.join(path, pattern)

# # Debug print to check the full pattern
# print(f'Full pattern: {full_pattern}')

# Use glob to find all files matching the pattern
matching_files = glob.glob(full_pattern)

print(f'Global min values for target: {global_min_target}')
print(f'Global max values for target: {global_max_target}')

import torch

# Function to perform min-max normalization with global min and max values
def min_max_normalize(data, global_min, global_max):
    return (data - global_min) / (global_max - global_min)

# Loop through the matching files again to normalize and save
for file_path in matching_files:
    if os.path.exists(file_path):
        try:
            # Perform operations on the file
            print(f'Processing file: {file_path}')
            ds_gc = xr.open_dataset(file_path).isel(history=0).isel(time=slice(1,7))
            ds_gc['lon'] = ds_gc['lon'] - 360
            
            #print('read file')
            
            start_time = ds_gc.time.values[0]
            end_time = ds_gc.time.values[-1]

            # Load the AORC datasets
            ds_aorc_apcp = ds_combined.sel(time=slice(start_time, end_time))
            ds_aorc_t2m   = ds_combined.sel(time=slice(start_time, end_time))
            ds_aorc_u10  = ds_combined.sel(time=slice(start_time, end_time))
            ds_aorc_v10  = ds_combined.sel(time=slice(start_time, end_time))
            ds_aorc_spfh  = ds_combined.sel(time=slice(start_time, end_time))
            
            # print('Read NOAA AORC data')
            # break
            
            # Interpolate ds_gc to the grid of ds_aorc_apcp
            ds_gc_interp = ds_gc.interp(lat=ds_aorc_apcp.latitude, lon=ds_aorc_apcp.longitude)

            # Ensure the input variables are in the same shape
            input_data = xr.Dataset({
                'u10m': ds_gc_interp['u10m'],
                'v10m': ds_gc_interp['v10m'],
                't2m': ds_gc_interp['t2m'],
                'tp06': ds_gc_interp['tp06'],
                'q1000': ds_gc_interp['q1000']
            })
            
            # Ensure the target variables are in the same shape
            target_data = xr.Dataset({
                'UGRD_10maboveground': ds_aorc_u10['UGRD_10maboveground'],
                'VGRD_10maboveground': ds_aorc_v10['VGRD_10maboveground'],
                'TMP_2maboveground': ds_aorc_t2m['TMP_2maboveground'],
                'APCP_surface': ds_aorc_apcp['APCP_surface'],
                'SPFH_2maboveground': ds_aorc_spfh['SPFH_2maboveground']
            })

            # Perform min-max normalization using global min and max
            input_data_norm = xr.Dataset({var: min_max_normalize(input_data[var], global_min_input[var], global_max_input[var]) for var in input_data})
            target_data_norm = xr.Dataset({var: min_max_normalize(target_data[var], global_min_target[var], global_max_target[var]) for var in target_data})

            # Convert the input and target data to PyTorch tensors
            input_tensor = torch.tensor(input_data_norm.to_array().values)
            target_tensor = torch.tensor(target_data_norm.to_array().values)
            #print('nans in input, nans in target, input.shape, target.shape ', torch.sum(torch.isnan(input_tensor)), torch.sum(torch.isnan(target_tensor)), input_tensor.shape, target_tensor.shape)
            # Define the output path for saving the .pt files
            input_output_path = os.path.join('/mnt/training_data_large/', f'{os.path.basename(file_path).split(".")[0]}_input.pt')
            target_output_path = os.path.join('/mnt/training_data_large/', f'{os.path.basename(file_path).split(".")[0]}_target.pt')
            #print(input_output_path)

            # Save the tensors to .pt files
            torch.save(input_tensor, input_output_path)
            torch.save(target_tensor, target_output_path)
            print(f'Saved normalized input data to {input_output_path}')
            print(f'Saved normalized target data to {target_output_path}')
            #break
        except:
            continue