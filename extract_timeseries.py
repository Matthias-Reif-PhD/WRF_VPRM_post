import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from scipy.interpolate import griddata
import argparse
import sys
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from collections import defaultdict



def compute_slope_aspect(hgt, lats, lons):
    lat_rad = np.radians(lats)
    dy = 111000  # meters per degree latitude
    dx = 111000 * np.cos(lat_rad)  # meters per degree longitude

    dz_dy = np.gradient(hgt, axis=0) / dy
    dz_dx = np.gradient(hgt, axis=1) / dx

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope = np.degrees(slope_rad)

    aspect = (np.degrees(np.arctan2(dz_dy, -dz_dx)) + 360) % 360
    return slope, aspect


def haversine_dist(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def find_nearest_grid_hgt_sa(
    lat_target, lon_target, lats, lons, location_pft, IVGTYP_vprm, hgt, hgt_site, radius
):
    slope, aspect = compute_slope_aspect(hgt, lats, lons)

    flat_idx = np.abs(lats - lat_target) + np.abs(lons - lon_target)
    min_flat_idx = np.argmin(flat_idx)
    lat_idx, lon_idx = np.unravel_index(min_flat_idx, lats.shape)
    target_slope = slope[lat_idx, lon_idx]
    target_aspect = aspect[lat_idx, lon_idx]

    pft_mask = IVGTYP_vprm == location_pft
    dist_km = haversine_dist(lat_target, lon_target, lats, lons)
    dist_mask = dist_km <= abs(radius)

    height_diff = np.abs(hgt - hgt_site)
    slope_diff = np.abs(slope - target_slope)
    aspect_diff = np.abs((aspect - target_aspect + 180) % 360 - 180)

    aspect_mask = aspect_diff <= 20
    slope_mask = slope_diff <= 10

    combined_mask = pft_mask & dist_mask & aspect_mask & slope_mask

    if not np.any(combined_mask):
        relaxed_mask = pft_mask & dist_mask
        if np.any(relaxed_mask):
            masked_height_diff = np.where(relaxed_mask, height_diff, np.inf)
            min_idx = np.unravel_index(np.argmin(masked_height_diff), hgt.shape)
            min_dist = dist_km[min_idx]
            return min_dist, min_idx
        else:
            fallback_flat_idx = np.argmin(dist_km)
            fallback_idx = np.unravel_index(fallback_flat_idx, lats.shape)
            fallback_dist = dist_km[fallback_idx]
            return fallback_dist, fallback_idx

    masked_height_diff = np.where(combined_mask, height_diff, np.inf)
    min_idx = np.unravel_index(np.argmin(masked_height_diff), hgt.shape)
    min_dist = dist_km[min_idx]

    return min_dist, min_idx, target_slope,target_aspect


def extract_datetime_from_filename(filename):
    """
    Extract datetime from WRF filename assuming format 'wrfout_d0x_YYYY-MM-DD_HH:MM:SS'.
    """
    base_filename = os.path.basename(filename)
    date_str = base_filename.split("_")[-2] + "_" + base_filename.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d_%H:%M:%S")


def extract_timeseries(wrf_path, start_date, end_date):

    run_Pmodel = False
    subday = ""
    if run_Pmodel:
        gpp_pmodel_path = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/gpp_pmodel/"
        migli_path = "/scratch/c7071034/DATA/RECO_Migli"
        subday = "subdailyC3_"
    
    wrf_path_dx_str = wrf_path.split("_")[-1]
    output_dir = "/scratch/c7071034/DATA/WRFOUT/csv"
    d0X = "wrfout_d01"
    if wrf_path_dx_str == "1km":
        d0X = "wrfout_d02"

    # Convert to datetime (but ignore time part for full-day selection)
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S").date()
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").date()

    # Collect all files
    all_files = sorted(glob.glob(os.path.join(wrf_path, f"{d0X}_*")))
    file_by_day = defaultdict(list)

    for f in all_files:
        dt = extract_datetime_from_filename(f)
        day = dt.date()
        if start_date_obj <= day <= end_date_obj:
            file_by_day[day].append((dt, f))

    # Filter for full days (24 hourly files starting from 00:00 to 23:00)
    file_list = []
    for day in sorted(file_by_day.keys()):
        files = sorted(file_by_day[day])
        if len(files) == 24 and all(dt.hour == i for i, (dt, _) in enumerate(files)):
            file_list.extend(f for _, f in files)

    # Define target locations (latitude, longitude)
    locations = [
        # {
        #     "name": "CH-Oe2_REF",
        #     "CO2_ID": "",
        #     "lat": 47.2863,
        #     "lon": 7.7343,
        #     "pft": 6,  # "CRO",
        #     "hgt_site": 452,
        # },
        {
            "name": "CH-Dav_REF",
            "CO2_ID": "",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        # {
        #     "name": "DE-Lkb_REF",
        #     "CO2_ID": "",
        #     "lat": 49.0996,
        #     "lon": 13.3047,
        #     "pft": 1,  # "ENF",
        #     "hgt_site": 1308,
        # },
        {
            "name": "IT-Lav_REF",
            "CO2_ID": "",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_REF",
            "CO2_ID": "",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_REF",
            "CO2_ID": "",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_REF",
            "CO2_ID": "",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        # {
        #     "name": "IT-Tor_REF",
        #     "CO2_ID": "",
        #     "lat": 45.8444,
        #     "lon": 7.5781,
        #     "pft": 7,  # "GRA",
        #     "hgt_site": 2160,
        # },
        # {
        #     "name": "CH-Lae_REF",
        #     "CO2_ID": "",
        #     "lat": 47.4781,
        #     "lon": 8.3644,
        #     "pft": 3,  # "MF",
        #     "hgt_site": 689,
        # },
        # {
        #     "name": "CH-Oe2_ALPS",
        #     "CO2_ID": "_REF",
        #     "lat": 47.2863,
        #     "lon": 7.7343,
        #     "pft": 6,  # "CRO",
        #     "hgt_site": 452,
        # },
        {
            "name": "CH-Dav_ALPS",
            "CO2_ID": "_REF",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        # {
        #     "name": "DE-Lkb_ALPS",
        #     "CO2_ID": "_REF",
        #     "lat": 49.0996,
        #     "lon": 13.3047,
        #     "pft": 1,  # "ENF",
        #     "hgt_site": 1308,
        # },
        {
            "name": "IT-Lav_ALPS",
            "CO2_ID": "_REF",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_ALPS",
            "CO2_ID": "_REF",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_ALPS",
            "CO2_ID": "_REF",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_ALPS",
            "CO2_ID": "_REF",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        # {
        #     "name": "IT-Tor_ALPS",
        #     "CO2_ID": "_REF",
        #     "lat": 45.8444,
        #     "lon": 7.5781,
        #     "pft": 7,  # "GRA",
        #     "hgt_site": 2160,
        # },
        # {
        #     "name": "CH-Lae_ALPS",
        #     "CO2_ID": "_REF",
        #     "lat": 47.4781,
        #     "lon": 8.3644,
        #     "pft": 3,  # "MF",
        #     "hgt_site": 689,
        # },
        # {
        #     "name": "CH-Oe2_SITE",
        #     "CO2_ID": "_2",
        #     "lat": 47.2863,
        #     "lon": 7.7343,
        #     "pft": 6,  # "CRO",
        #     "hgt_site": 452,
        # },
        {
            "name": "CH-Dav_SITE",
            "CO2_ID": "_2",
            "lat": 46.8153,
            "lon": 9.8559,
            "pft": 1,  # "ENF",
            "hgt_site": 1639,
        },
        # {
        #     "name": "DE-Lkb_SITE",
        #     "CO2_ID": "_3", # REMOVED DE-Lbk - Not enough data
        #     "lat": 49.0996,
        #     "lon": 13.3047,
        #     "pft": 1,  # "ENF",
        #     "hgt_site": 1308,
        # },
        {
            "name": "IT-Lav_SITE",
            "CO2_ID": "_4",
            "lat": 45.9562,
            "lon": 11.2813,
            "pft": 1,  # "ENF",
            "hgt_site": 1353,
        },
        {
            "name": "IT-Ren_SITE",
            "CO2_ID": "_3",
            "lat": 46.5869,
            "lon": 11.4337,
            "pft": 1,  # "ENF",
            "hgt_site": 1730,
        },
        {
            "name": "AT-Neu_SITE",
            "CO2_ID": "_4",
            "lat": 47.1167,
            "lon": 11.3175,
            "pft": 7,  # "GRA",
            "hgt_site": 970,
        },
        {
            "name": "IT-MBo_SITE",
            "CO2_ID": "_3",
            "lat": 46.0147,
            "lon": 11.0458,
            "pft": 7,  # "GRA",
            "hgt_site": 1550,
        },
        # {
        #     "name": "IT-Tor_SITE",
        #     "CO2_ID": "_2",
        #     "lat": 45.8444,
        #     "lon": 7.5781,
        #     "pft": 7,  # "GRA",
        #     "hgt_site": 2160,
        # },
        # {
        #     "name": "CH-Lae_SITE",
        #     "CO2_ID": "_2",
        #     "lat": 47.4781,
        #     "lon": 8.3644,
        #     "pft": 3,  # "MF",
        #     "hgt_site": 689,
        # },
    ]

    # Define the remapping dictionary for CORINE vegetation types
    corine_to_vprm = {
        24: 1,  # Coniferous Forest (Evergreen)
        23: 2,  # Broad-leaved Forest (Deciduous)
        25: 3,
        29: 3,  # Mixed Forest and Transitional Woodland-Shrub
        27: 4,
        28: 4,  # Moors and Heathland, Sclerophyllous Vegetation (Shrubland)
        35: 5,
        36: 5,
        37: 5,  # Wetlands: Inland Marshes, Peat Bogs, Salt Marshes
        12: 6,
        13: 6,
        14: 6,
        15: 6,
        16: 6,
        17: 6,
        19: 6,
        20: 6,
        21: 6,
        22: 6,  # Cropland
        18: 7,
        26: 7,  # Grassland: Pastures, Natural Grasslands
        # Others mapped to 8 (gray)
        1: 8,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 8,
        8: 8,
        9: 8,
        10: 8,
        11: 8,
        30: 8,
        31: 8,
        32: 8,
        33: 8,
        34: 8,
        38: 8,
        39: 8,
        40: 8,
        41: 8,
        42: 8,
        43: 8,
        44: 8,
    }

    # Initialize an empty DataFrame with time as the index and locations as columns
    columns = (
        [f"{location['name']}_GPP_WRF" for location in locations]
        + [f"{location['name']}_RECO_WRF" for location in locations]
        + [f"{location['name']}_T2_WRF" for location in locations]
    )
    if run_Pmodel:
        columns = (
            [f"{location['name']}_GPP_WRF" for location in locations]
            + [f"{location['name']}_RECO_WRF" for location in locations]
            + [f"{location['name']}_T2_WRF" for location in locations]
            + [f"{location['name']}_GPP_Pmodel" for location in locations]
            + [f"{location['name']}_RECO_Migli" for location in locations]
        )


    df_out = pd.DataFrame(columns=columns)
    # Process each WRF file (representing one timestep)
    for nc_f1 in file_list:
        nc_fid1 = nc.Dataset(nc_f1, "r")
        wrf_file = nc_f1.split("/")[-1]
        date_time = wrf_file.split("_")[2] + "_" + wrf_file.split("_")[3]
        file_end = wrf_path_dx_str + "_" + date_time
        print(f"Processing {file_end}")
        xlat = nc_fid1.variables["XLAT"][0]  # Assuming the first time slice
        xlon = nc_fid1.variables["XLONG"][0]
        WRF_T2 = nc_fid1.variables["T2"][0]
        hgt = nc_fid1.variables["HGT"][0]
        IVGTYP = nc_fid1.variables["IVGTYP"][0]
        IVGTYP_vprm = np.vectorize(corine_to_vprm.get)(
            IVGTYP[:, :]
        )  # Create a new array for the simplified vegetation categories
        dx = (xlat[0, 0] - xlat[1, 0]) * 111
        radius = dx * 20

        if run_Pmodel:
            # find file in migli_path which ends with date_time
            reco_migli_files = [
                f
                for f in sorted(glob.glob(os.path.join(migli_path, "reco_migli*")))
                if file_end in f
            ]
            if not reco_migli_files:
                raise FileNotFoundError(f"No reco_migli file found for {file_end}")
            reco_migli_file = reco_migli_files[0]
            reco_migli = xr.open_dataset(reco_migli_file)
            reco_migli = reco_migli["RECO_Migli"].values
            # find file in gpp which ends with date_time
            gpp_pmodel_file = [
                f
                for f in sorted(
                    glob.glob(os.path.join(gpp_pmodel_path, f"gpp_pmodel_{subday}*"))
                )
                if file_end in f
            ][0]

            gpp_pmodel = xr.open_dataset(gpp_pmodel_file)
            gpp_pmodel = gpp_pmodel["GPP_Pmodel"].values

        # Initialize lists to store data for the current timestep
        data_row = {col: None for col in df_out.columns}  # Map columns to values

        # Extract data for each location
        for location in locations:
            lat_target, lon_target = location["lat"], location["lon"]
            WRF_gee = nc_fid1.variables[f"EBIO_GEE{location['CO2_ID']}"][0, 0, :, :]
            WRF_res = nc_fid1.variables[f"EBIO_RES{location['CO2_ID']}"][0, 0, :, :]

                # Get nearest neighbour of GEE, RES, and T2 for the current location and append to the row

            dist_km, grid_idx , target_slope,target_aspect = find_nearest_grid_hgt_sa(
                lat_target,
                lon_target,
                xlat,
                xlon,
                location["pft"],
                IVGTYP_vprm,
                hgt,
                location["hgt_site"],
                radius,
            )
            # add dist to the large locations dict which contains all the locations
            for loc in locations:
                if loc["name"] == location["name"]:
                    loc["dist"] = dist_km
                    loc["hgt_wrf"] = hgt[grid_idx[0], grid_idx[1]]
                    loc["lat_wrf"] = xlat[grid_idx[0], grid_idx[1]]
                    loc["lon_wrf"] = xlon[grid_idx[0], grid_idx[1]]
                    break

            # Assign values to their respective columns
            data_row[f"{location['name']}_GPP_WRF"] = (
                WRF_gee[grid_idx[0], grid_idx[1]] / 3600
            )
            data_row[f"{location['name']}_RECO_WRF"] = (
                WRF_res[grid_idx[0], grid_idx[1]] / 3600
            )
            data_row[f"{location['name']}_T2_WRF"] = WRF_T2[
                grid_idx[0], grid_idx[1]
            ]
            if run_Pmodel:
                data_row[f"{location['name']}_GPP_Pmodel"] = gpp_pmodel[
                    grid_idx[0], grid_idx[1]
                ]
                data_row[f"{location['name']}_RECO_Migli"] = reco_migli[
                    grid_idx[0], grid_idx[1]
                ]

        # Append the current timestep data as a new row in the DataFrame
        temp_df_out = pd.DataFrame([data_row])
        # Filter out empty or all-NA DataFrames before concatenation to avoid FutureWarning
        frames_to_concat = [df for df in [df_out, temp_df_out] if not df.empty and not df.isna().all(axis=None)]
        if frames_to_concat:
            df_out = pd.concat(frames_to_concat, ignore_index=True)
        nc_fid1.close()

    # Set the time as the index of the DataFrame
    df_out.index = [extract_datetime_from_filename(f) for f in file_list]
    # Optionally, save the DataFrame to CSV
    output_filename = f"wrf_FLUXNET_sites_{wrf_path.split('_')[-1]}_{start_date.split('_')[0]}_{end_date.split('_')[0]}.csv"

    df_out.to_csv(
        os.path.join(
            output_dir,
            output_filename,
        )
    )
    # write another csv file with the distances but use only the _REF sites
    dist_rows = []
    for loc in locations:
        if "ref" in loc["name"]:
            dist_rows.append(
                {
                    "name": loc["name"],
                    "dist": loc["dist"],
                    "hgt_wrf": loc["hgt_wrf"],
                    "lat_wrf": loc["lat_wrf"],
                    "lon_wrf": loc["lon_wrf"],
                    "pft": loc["pft"],
                }
            )
    df_out_dist = pd.DataFrame(dist_rows, columns=["name", "dist", "hgt_wrf", "lat_wrf", "lon_wrf", "pft"])
    df_out_dist.to_csv(
        os.path.join(
            output_dir,
            f"distances_{wrf_path.split('_')[-1]}_{start_date.split('_')[0]}_{end_date.split('_')[0]}.csv",
        )
    )

    return


def main():

    if len(sys.argv) > 1:  # to run on cluster
        parser = argparse.ArgumentParser(description="Description of your script")
        parser.add_argument(
            "-s", "--start", type=str, help="Format: 2012-07-01 01:00:00"
        )
        parser.add_argument("-e", "--end", type=str, help="Format: 2012-07-01 01:00:00")
        args = parser.parse_args()
        start_date = args.start
        end_date = args.end
    else:  # to run locally 
        start_date = "2012-01-01 00:00:00"
        end_date = "2012-12-31 00:00:00"

    wrf_paths = [
        "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km",
        # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km",
        # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km",
        # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km",
        # "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km",
    ]
    for wrf_path in wrf_paths:
        extract_timeseries(wrf_path, start_date, end_date)

if __name__ == "__main__":
    main()
