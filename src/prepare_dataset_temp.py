import xarray as xr
import os
import glob

path = '/mnt/washington/'

global_min_input = {'u10m': -11.2865629196167, 'v10m': -12.6307373046875, 't2m': 252.3680419921875, 'tp06': -0.00011615798575803638}
global_max_input = {'u10m': 12.917617797851562, 'v10m': 12.816701889038086, 't2m': 308.9913024902344, 'tp06': 0.0838286280632019}

# Load the AORC datasets
ds_aorc_apcp = xr.open_dataset('/mnt/noaa_aorc_washington_APCP_surface_2017_2023.nc')#.sel(time=slice(start_time, end_time))
ds_aorc_t2m   = xr.open_dataset('/mnt/noaa_aorc_washington_TMP_2maboveground_2017_2023.nc')#.sel(time=slice(start_time, end_time))
ds_aorc_u10  = xr.open_dataset('/mnt/noaa_aorc_washington_UGRD_10maboveground_2017_2023.nc')#.sel(time=slice(start_time, end_time))
ds_aorc_v10  = xr.open_dataset('/mnt/noaa_aorc_washington_VGRD_10maboveground_2017_2023.nc')#.sel(time=slice(start_time, end_time))


global_min_target = {'APCP_surface': 0.0, 'TMP_2maboveground': 247.20000368356705, 'UGRD_10maboveground': -20.500000305473804, 'VGRD_10maboveground': -20.000000298023224}
global_max_target = {'APCP_surface': 118.60000176727772, 'TMP_2maboveground': 313.4000046700239, 'UGRD_10maboveground': 23.100000344216824, 'VGRD_10maboveground': 24.800000369548798}

# Update global min and max for target data
# global_min_target['APCP_surface'] = min(global_min_target['APCP_surface'], ds_aorc_apcp['APCP_surface'].min().item())
# global_max_target['APCP_surface'] = max(global_max_target['APCP_surface'], ds_aorc_apcp['APCP_surface'].max().item())
# global_min_target['TMP_2maboveground'] = min(global_min_target['TMP_2maboveground'], ds_aorc_t2m['TMP_2maboveground'].min().item())
# global_max_target['TMP_2maboveground'] = max(global_max_target['TMP_2maboveground'], ds_aorc_t2m['TMP_2maboveground'].max().item())
# global_min_target['UGRD_10maboveground'] = min(global_min_target['UGRD_10maboveground'], ds_aorc_u10['UGRD_10maboveground'].min().item())
# global_max_target['UGRD_10maboveground'] = max(global_max_target['UGRD_10maboveground'], ds_aorc_u10['UGRD_10maboveground'].max().item())
# global_min_target['VGRD_10maboveground'] = min(global_min_target['VGRD_10maboveground'], ds_aorc_v10['VGRD_10maboveground'].min().item())
# global_max_target['VGRD_10maboveground'] = max(global_max_target['VGRD_10maboveground'], ds_aorc_v10['VGRD_10maboveground'].max().item())

# Define the path and the pattern
path = '/mnt/washington/'  # Ensure the path is absolute
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
            ds_aorc_apcp = xr.open_dataset('/mnt/noaa_aorc_washington_APCP_surface_2017_2023.nc').sel(time=slice(start_time, end_time))
            ds_aorc_t2m   = xr.open_dataset('/mnt/noaa_aorc_washington_TMP_2maboveground_2017_2023.nc').sel(time=slice(start_time, end_time))
            ds_aorc_u10  = xr.open_dataset('/mnt/noaa_aorc_washington_UGRD_10maboveground_2017_2023.nc').sel(time=slice(start_time, end_time))
            ds_aorc_v10  = xr.open_dataset('/mnt/noaa_aorc_washington_VGRD_10maboveground_2017_2023.nc').sel(time=slice(start_time, end_time))
            
            # print('Read NOAA AORC data')
            # break
            
            # Interpolate ds_gc to the grid of ds_aorc_apcp
            ds_gc_interp = ds_gc.interp(lat=ds_aorc_apcp.latitude, lon=ds_aorc_apcp.longitude)

            # Ensure the input variables are in the same shape
            input_data = xr.Dataset({
                # 'u10m': ds_gc_interp['u10m'],
                # 'v10m': ds_gc_interp['v10m'],
                't2m': ds_gc_interp['t2m'],
                # 'tp06': ds_gc_interp['tp06']
            })
            
            # Ensure the target variables are in the same shape
            target_data = xr.Dataset({
                # 'APCP_surface': ds_aorc_apcp['APCP_surface'],
                'TMP_2maboveground': ds_aorc_t2m['TMP_2maboveground'],
                # 'UGRD_10maboveground': ds_aorc_u10['UGRD_10maboveground'],
                # 'VGRD_10maboveground': ds_aorc_v10['VGRD_10maboveground']
            })

            # Perform min-max normalization using global min and max
            input_data_norm = xr.Dataset({var: min_max_normalize(input_data[var], global_min_input[var], global_max_input[var]) for var in input_data})
            target_data_norm = xr.Dataset({var: min_max_normalize(target_data[var], global_min_target[var], global_max_target[var]) for var in target_data})

            # Convert the input and target data to PyTorch tensors
            input_tensor = torch.tensor(input_data_norm.to_array().values)
            target_tensor = torch.tensor(target_data_norm.to_array().values)
            #print('nans in input, nans in target, input.shape, target.shape ', torch.sum(torch.isnan(input_tensor)), torch.sum(torch.isnan(target_tensor)), input_tensor.shape, target_tensor.shape)
            # Define the output path for saving the .pt files
            input_output_path = os.path.join('/mnt/training_data_temp/', f'{os.path.basename(file_path).split(".")[0]}_input.pt')
            target_output_path = os.path.join('/mnt/training_data_temp/', f'{os.path.basename(file_path).split(".")[0]}_target.pt')
            #print(input_output_path)

            # Save the tensors to .pt files
            torch.save(input_tensor, input_output_path)
            torch.save(target_tensor, target_output_path)
            print(f'Saved normalized input data to {input_output_path}')
            print(f'Saved normalized target data to {target_output_path}')
            #break
        except:
            continue