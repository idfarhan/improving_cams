import pandas as pd
import functools as ft
from more_itertools import sliced
import xarray as xr
from src.reference_grid import ref_grid
from src.oco_processing import oco_data_files, oco_gridding
from src.era5_processing import era_read_file
from src.cams_processing import cams_read_file, cams_0p75_read_file
from src.odiac_processing import odiac_read_file
from src.modis_processing import modis_ndvi_read_file
from src.landscan_processing import landscan_read_file
from src.gfed_processing import gfed_read_file

def prepare_training_data(oco_dir, era_dir, cams_file, cams_dir, odiac_dir, ndvi_dir, landscan_dir, gfed_dir, years, grid, chunk_size=50):
    oco_files = oco_data_files(oco_dir, years)
    for oco_file in oco_files:
        overpass_date, oco_df = oco_gridding(oco_file, grid)
        era_ds = era_read_file(era_dir, overpass_date)
        cams_ds = cams_0p75_read_file(cams_file, overpass_date)
        cams2_ds = cams_read_file(cams_dir, overpass_date)
        
        odiac_ds = odiac_read_file(odiac_dir, overpass_date)
        ndvi_ds = modis_ndvi_read_file(ndvi_dir, overpass_date)
        landscan_ds = landscan_read_file(landscan_dir, overpass_date)
        gfed_ds = gfed_read_file(gfed_dir, overpass_date)
        
        index_slices = sliced(range(len(oco_df)), chunk_size)
        for index_slice in index_slices:
            chunk = oco_df.iloc[index_slice].reset_index()
            chunk['time'] = pd.to_datetime(chunk['rounded_time'])
            chunk['longitude'] = chunk['longitude'].values
            chunk['latitude'] = chunk['latitude'].values
            chunk['time'] = pd.to_datetime(chunk['time']).values
            
            era_ds_interp = era_ds.interp(longitude=chunk['longitude'],
                                        latitude=chunk['latitude'],
                                        time=chunk['time']
                                        ).to_dataframe().reset_index()
            era_ds_interp.drop_duplicates(inplace=True)

            cams_ds_interp = cams_ds.interp(longitude=chunk['longitude'],
                                        latitude=chunk['latitude'],
                                        time=chunk['time']
                                        ).to_dataframe().reset_index()
            cams_ds_interp.drop_duplicates(inplace=True)
            #cams_ds_interp['cams'] = cams_ds_interp['cams'] * 1000000

            cams2_ds_interp = cams2_ds.interp(longitude=chunk['longitude'],
                                        latitude=chunk['latitude'],
                                        time=chunk['time']
                                        ).to_dataframe().reset_index()
            cams2_ds_interp.drop_duplicates(inplace=True)
            cams2_ds_interp['cams2'] = cams2_ds_interp['cams2'] * 1000000

            dfs_tmp = [chunk, era_ds_interp, cams_ds_interp, cams2_ds_interp]
            merged_tmp = ft.reduce(lambda left, right: pd.merge(left, right,
                                                                on=['longitude', 'latitude', 'time']), dfs_tmp)
            merged_tmp.drop('time', axis=1, inplace=True)
            
            odiac_ds_interp = odiac_ds.interp(x=chunk['longitude'],
                                        y=chunk['latitude']
                                        ).to_dataframe().reset_index()
            odiac_ds_interp.drop(['spatial_ref', 'band'], inplace=True, axis=1)
            odiac_ds_interp.rename(columns={'x':'longitude', 'y':'latitude'}, inplace=True)
            odiac_ds_interp.drop_duplicates(inplace=True)

            ndvi_ds_interp = ndvi_ds.interp(longitude=chunk['longitude'],
                                        latitude=chunk['latitude']
                                        ).to_dataframe().reset_index()
            ndvi_ds_interp.fillna(-3000, inplace=True)
            ndvi_ds_interp.drop_duplicates(inplace=True)

            landscan_ds_interp = landscan_ds.interp(x=chunk['longitude'],
                                                    y=chunk['latitude']
                                                    ).to_dataframe().reset_index()
            landscan_ds_interp.drop(['spatial_ref', 'band'], inplace=True, axis=1)
            landscan_ds_interp.rename(columns={'x':'longitude', 'y':'latitude'}, inplace=True)
            landscan_ds_interp.fillna(0, inplace=True)
            landscan_ds_interp.drop_duplicates(inplace=True)

            gfed_ds_interp = gfed_ds.interp(longitude=chunk['longitude'],
                                        latitude=chunk['latitude']
                                        ).to_dataframe().reset_index()
            gfed_ds_interp.fillna(-3000, inplace=True)
            gfed_ds_interp.drop_duplicates(inplace=True)
            
            dfs = [merged_tmp, odiac_ds_interp, ndvi_ds_interp, landscan_ds_interp, gfed_ds_interp]
            merged_df = ft.reduce(lambda left, right: pd.merge(left, right,
                                                                on=['longitude', 'latitude']), dfs)
            merged_df.drop('index', axis=1, inplace=True)

            merged_df.to_csv('training_data_new.csv', header=False, mode='a', index=False)
            
    return print(merged_df.columns.tolist())

def main():
    # Define directories and parameters
    oco_dir = '/Volumes/My Passport/datasets/oco2v11p1'
    era_dir = '/Volumes/My Passport/datasets/era_new/training_data'
    cams_file = '/Volumes/My Passport/datasets/cams/xco2_0p75/cams_xco2_0p75.nc'
    cams_dir = '/Volumes/My Passport/datasets/cams/latest'
    odiac_dir = '/Volumes/My Passport/datasets/odiac/1km'
    ndvi_dir = '/Volumes/My Passport/datasets/modis/MYD13C1v061'
    landscan_dir = '/Volumes/My Passport/datasets/landscan'
    gfed_dir = '/Volumes/My Passport/datasets/gfed/4.1'
    years = [2020]
    grid = ref_grid()
    
    # Call the function to prepare training data
    training_data = prepare_training_data(oco_dir, era_dir, cams_file,
                                          cams_dir, odiac_dir, ndvi_dir,
                                          landscan_dir, gfed_dir,
                                          years, grid)

if __name__ == "__main__":
    main()