import xarray as xr
import fsspec
import numpy as np
import s3fs
import zarr
from tqdm import tqdm
from dask.distributed import Client

def main():
    base_url = 's3://noaa-nws-aorc-v1-1-1km'

    client = Client()

    # year = '1979'

    for year_ in tqdm(range(2017,2024)):
        year = str(year_)
        print(year)
        single_year_url = f'{base_url}/{year}.zarr/'

        #ds_single = xr.open_zarr(fsspec.get_mapper(single_year_url, anon=True), consolidated=True).sel(longitude=slice(-110, -90)).sel(latitude=slice(33, 45))
        ds_single = xr.open_zarr(fsspec.get_mapper(single_year_url, anon=True), consolidated=True).sel(longitude=slice(-88, -80)).sel(latitude=slice(32, 40))


        print(ds_single)
        ds_single.to_netcdf('noaa_aorc_32_40n_80_88w_'+year+'.nc')

if __name__ == "__main__":
    main()
