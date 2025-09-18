import pandas as pd
import matplotlib.pyplot as plt
import glob
from datetime import datetime, timedelta
import pytz
import os
import numpy as np
from permetrics import RegressionMetric
from scipy.interpolate import RegularGridInterpolator
import netCDF4 as nc


def find_file_paths(base_dir, location):
    """
    Finds the correct file path for a given location based on the pattern.

    Args:
        base_dir (str): The base directory to search.
        location (str): The location code (e.g., "AT-Neu").

    Returns:
        str: The path of the matching file, or None if not found.
    """
    pattern = f"FLX_{location}_FLUXNET2015_FULLSET*/FLX_{location}_FLUXNET2015_FULLSET_HH*.csv"
    search_pattern = os.path.join(base_dir, pattern)
    matches = glob.glob(search_pattern)
    if matches:
        return matches[
            0
        ]  # Return the first match (or handle multiple matches if needed)
    else:
        print(f"No file found for location: {location}")
        return None


def read_FLUXNET_site(start_date, end_date, location, base_dir, var_flx):
    # Convert input dates to datetime objects with UTC timezone
    start_date_obj = start_date.replace(tzinfo=pytz.UTC)
    end_date_obj = end_date.replace(tzinfo=pytz.UTC)

    # Example: Find the file paths for all locations
    file_path = find_file_paths(base_dir, location)

    # Read CSV and create a DataFrame
    df_FLX_site = pd.read_csv(file_path, sep=",")

    # Drop the second row (index 0 is the header, index 1 contains units)
    df_FLX_site = df_FLX_site.drop(index=0)

    # Parse 'TIMESTAMP_START' from the raw format
    df_FLX_site["TIMESTAMP_START"] = pd.to_datetime(
        df_FLX_site["TIMESTAMP_START"], format="%Y%m%d%H%M", errors="coerce"
    )

    # Check for parsing errors
    if df_FLX_site["TIMESTAMP_START"].isna().any():
        print("Some 'TIMESTAMP_START' values could not be parsed.")
        print(df_FLX_site[df_FLX_site["TIMESTAMP_START"].isna()])

    # Convert to UTC timezone (assume 'Europe/Berlin' for initial localization)
    # df_FLX_site['TIMESTAMP_START'] = df_FLX_site['TIMESTAMP_START'].dt.tz_localize('Europe/Berlin', nonexistent='shift_forward', ambiguous='NaT')
    # df_FLX_site['TIMESTAMP_START'] = df_FLX_site['TIMESTAMP_START'].dt.tz_convert('UTC') # TODO this is not working, adjusting manually

    df_FLX_site["TIMESTAMP_START"] = df_FLX_site["TIMESTAMP_START"] - pd.Timedelta(
        hours=1
    )  # TODO: make sure time change is handled correcly
    df_FLX_site["TIMESTAMP_START"] = df_FLX_site["TIMESTAMP_START"].dt.tz_localize(
        "UTC"
    )

    # Filter the DataFrame based on the date range
    df_FLX_site = df_FLX_site[
        (df_FLX_site["TIMESTAMP_START"] >= start_date_obj)
        & (df_FLX_site["TIMESTAMP_START"] <= end_date_obj)
    ]

    # Debugging: Check the filtered data
    print("Filtered DataFrame shape:", df_FLX_site.shape)

    if df_FLX_site.empty:
        print("No data in the specified date range.")

    # Select relevant columns and clean
    df_FLX_site = df_FLX_site[["TIMESTAMP_START", var_flx, "NIGHT"]].copy()
    df_FLX_site = df_FLX_site.mask(df_FLX_site == -9999, np.nan)

    # Ensure GPP_NT_VUT_REF is negative
    if var_flx.startswith("GPP"):
        df_FLX_site[var_flx] = df_FLX_site[var_flx]
        df_FLX_site[var_flx] = np.where(
            df_FLX_site["NIGHT"] == 1, 0, df_FLX_site[var_flx]
        )
        df_FLX_site[var_flx] = df_FLX_site[var_flx].clip(lower=0)

    # Ensure TIMESTAMP_START is a datetime object
    df_FLX_site["TIMESTAMP_START"] = pd.to_datetime(df_FLX_site["TIMESTAMP_START"])

    # Set TIMESTAMP_START as the index
    df_FLX_site.set_index("TIMESTAMP_START", inplace=True)

    # Resample to 1-hour intervals and aggregate using the mean
    df_FLX_site_resampled = df_FLX_site.resample("1h").mean()

    # Reset the index if needed
    df_FLX_site_resampled.reset_index(inplace=True)
    return df_FLX_site_resampled


# Path to the directory containing the CSV files
timespan = "2012-01-01 00:00:00_2012-12-31 00:00:00"
sim_type = "_all"  # "", "_cloudy" or  "_all"
csv_dir = "/scratch/c7071034/DATA/WRFOUT/csv"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots"
base_dir_FLX = "/scratch/c7071034/DATA/Fluxnet2015/Alps"
fluxtypes = ["T2_WRF", "NEE_WRF", "GPP_WRF", "RECO_WRF"]
units = ["[°C]", "[μmol/m²/s]", "[μmol/m²/s]", "[μmol/m²/s]"]
res_dx = "1km"
plot_CAMS = True  # True -> need to load all CAMS files
# List all relevant CSV files
# if res_dx == "1km":
if sim_type == "_all":
    csv_files = glob.glob(
        f"{csv_dir}/wrf_FLUXNET_sites_{res_dx}*_{timespan}.csv"
    )  # be carefull, chack that there is no other data
else:
    csv_files = glob.glob(
        f"{csv_dir}/wrf_FLUXNET_sites_{res_dx}{sim_type}_{timespan}.csv"
    )

csv_files_sorted = sorted(
    csv_files,
    key=lambda x: int(
        x.split(f"_sites_")[1].split("km")[0]
    ),  # Extract distance as an integer
    reverse=True,  # Sort in descending order
)

# Initialize a dictionary to store dataframes for each resolution
dataframes = {}
# Read each CSV into a DataFrame and store in the dictionary
for csv_file in csv_files_sorted:
    resolution = res_dx  # Extract resolution from filename
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    dataframes[resolution] = df

start_date = df.index[0]
end_date = df.index[-1]
# Example: Plotting GPP (Gross Ecosystem Exchange) for all resolutions and locations

if res_dx == "1km":
    locations = [
        "CH-Dav",
        "IT-Lav",
        "IT-Ren",
        "AT-Neu",
        "IT-MBo",
    ]
    # locations = [ "IT-Ren"]
    locations_hgt = {
        "CH-Dav": 1639,
        "IT-Lav": 1353,
        "IT-Ren": 1730,
        "AT-Neu": 970,
        "IT-MBo": 1550,
    }
else:
    locations = [
        "IT-Tor",
        "IT-Lav",
        "AT-Neu",
        "CH-Cha",
        "CH-Dav",
        "CH-Fru",
        "CH-Lae",
        "CH-Oe1",
        "CH-Oe2",
        "DE-Lkb",
        "IT-Isp",
        "IT-La2",
        "IT-MBo",
        "IT-PT1",
        "IT-Ren",
    ]
    # locations = ["CH-Cha", "CH-Lae"]

    locations_hgt = {
        "AT-Neu": 970,
        "CH-Cha": 393,
        "CH-Dav": 1639,
        "CH-Fru": 982,
        "CH-Lae": 689,
        "CH-Oe1": 450,
        "CH-Oe2": 452,
        "DE-Lkb": 1308,
        "IT-Isp": 210,
        "IT-La2": 1350,
        "IT-Lav": 1353,
        "IT-MBo": 1550,
        "IT-PT1": 60,
        "IT-Ren": 1730,
        "IT-Tor": 2160,
    }


resolution_colors = {
    "1km": "blue",
    "3km": "darkgrey",
    "9km": "purple",
    "27km": "red",
    "54km": "green",
}
# read from
if sim_type == "_all":
    pft_site_match = pd.read_csv(f"{csv_dir}/distances_{res_dx}_{timespan}.csv")
else:
    pft_site_match = pd.read_csv(
        f"{csv_dir}/distances_{res_dx}{sim_type}_{timespan}.csv"
    )
model_lat_lon = []
for i in range(len(locations)):
    model_lat_lon.append(
        {
            "name": pft_site_match["name"][i],
            "lat": pft_site_match["lat_wrf"][i],
            "lon": pft_site_match["lon_wrf"][i],
            "veg_frac": pft_site_match["veg_frac_idx"][i],
        }
    )
# read from pft_site_match_at_3km.csv
# pft_site_match = pd.read_csv(
#     f"/home/c707/c7071034/Github/WRF_VPRM_post/pft_site_match_at_{res_dx}.csv"
# )
# model_lat_lon = []
# for i in range(len(locations)):
#     model_lat_lon.append(
#         {
#             "name": pft_site_match["site"][i],
#             "lat": pft_site_match["model_lat"][i],
#             "lon": pft_site_match["model_lon"][i],
#         }
#     )
# model_lat_lon
# load CAMS data
CAMS_data_dir_path = "/scratch/c7071034/DATA/CAMS/"

path_CAMS_file = os.path.join(
    CAMS_data_dir_path + "ghg-reanalysis_surface_2012_full.nc"
)

CAMS_data = nc.Dataset(path_CAMS_file)
times_CAMS = CAMS_data.variables["valid_time"]
gppbfas = CAMS_data.variables["gppbfas"][:]
rec_bfas = CAMS_data.variables["recbfas"][:]

CAMS_vars = ["fco2gpp", "fco2rec", "fco2nee", "t2m"]
factor_kgC = 1000000 / 0.04401  # conversion from kgCO2/m2/s to mgC/m2/s to  mumol/m2/s
factors = [factor_kgC, -factor_kgC, -factor_kgC, 1]

lat_CAMS = CAMS_data.variables["latitude"][:]
lon_CAMS = CAMS_data.variables["longitude"][:]


def get_int_var(lat_target, lon_target, lats, lons, var_CAMS):
    interpolator = RegularGridInterpolator((lats, lons), var_CAMS)
    interpolated_value = interpolator((lat_target, lon_target))
    return interpolated_value


df_CAMS_hourly_all = {}
for location_ll in model_lat_lon:
    lat_target, lon_target = location_ll["lat"], location_ll["lon"]

    data_rows = []
    new_time = start_date - timedelta(minutes=60)
    ts_CAMS = pd.DataFrame()
    j = 0
    for time_CAMS in times_CAMS:
        date_CAMS = datetime(1970, 1, 1) + timedelta(seconds=int(time_CAMS))
        formatted_time = date_CAMS.strftime("%Y-%m-%d %H:%M:%S")
        row = {"time": formatted_time}
        for CAMS_var, factor in zip(CAMS_vars, factors):
            var_CAMS = (
                CAMS_data.variables[CAMS_var][j, :, :].data * factor
            )  # convert unit to mmol m-2 s-1
            row[CAMS_var] = get_int_var(
                lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS
            )
            if CAMS_var == "fco2gpp":
                var_CAMS = var_CAMS * gppbfas[j, :, :].data
                row["fco2gpp_bfas"] = get_int_var(
                    lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS
                )
            if CAMS_var == "fco2rec":
                var_CAMS = var_CAMS * rec_bfas[j, :, :].data
                row["fco2rec_bfas"] = get_int_var(
                    lat_target, lon_target, lat_CAMS, lon_CAMS, var_CAMS
                )
            if CAMS_var == "fco2nee":
                row["fco2nee_bfas"] = row["fco2nee"]
                row["fco2nee"] = row["fco2rec"] - row["fco2gpp"]

        j += 1
        data_rows.append(row)

    df_CAMS = pd.DataFrame(data_rows)
    df_CAMS["time"] = pd.to_datetime(df_CAMS["time"])
    df_CAMS.set_index("time", inplace=True)
    df_CAMS = df_CAMS.astype(float)
    # df_CAMS_hourly = df_CAMS.resample('h').interpolate(method='linear')
    df_CAMS = df_CAMS[~df_CAMS.index.duplicated(keep="first")]  # drop duplicates
    df_CAMS_hourly = df_CAMS.resample("h").interpolate(method="linear")

    df_CAMS_hourly["t2m"] -= 273.15
    df_CAMS_hourly_all[location_ll["name"]] = df_CAMS_hourly

# Create plots for each location
consolidated_metrics_all = pd.DataFrame()
for location in locations:
    for fluxtype, unit in zip(fluxtypes, units):
        consolidated_metrics_df = pd.DataFrame(
            index=["RMSE", "R2"],  # Metrics as rows
        )
        variables = {
            f"SITE_{fluxtype}": "solid",
            f"ALPS_{fluxtype}": "dashdot",
            f"REF_{fluxtype}": "dashed",
        }
        # for Pmodel:
        #         variables = {f"SITE_{fluxtype}": "solid",f"ALPS_{fluxtype}": "dashdot"  , f"REF_{fluxtype}": "dashed" ,  "ALPS_GPP_Pmodel" : "dotted", "ALPS_RECO_Migli" : "dotted", "ALPS_NEE_PMigli" : "dotted"}

        if fluxtype == "RECO_WRF":
            var_flx = "RECO_NT_VUT_USTAR50"
            df_FLX_site = read_FLUXNET_site(
                start_date, end_date, location, base_dir_FLX, var_flx
            )
            var_CAMS_plot = "fco2rec"
        elif fluxtype == "GPP_WRF":
            var_flx = "GPP_NT_VUT_USTAR50"
            df_FLX_site = read_FLUXNET_site(
                start_date, end_date, location, base_dir_FLX, var_flx
            )
            var_CAMS_plot = "fco2gpp"
        elif fluxtype == "NEE_WRF":
            var_flx = "NEE_VUT_USTAR50"
            df_FLX_site = read_FLUXNET_site(
                start_date, end_date, location, base_dir_FLX, var_flx
            )
            var_CAMS_plot = "fco2nee"
        elif fluxtype == "T2_WRF":
            var_flx = "TA_F"
            df_FLX_site = read_FLUXNET_site(
                start_date, end_date, location, base_dir_FLX, var_flx
            )
            var_CAMS_plot = "t2m"
        # continue if df_FLX_site is empty
        if df_FLX_site.empty:
            print(f"Skipping {location} for {fluxtype} due to empty FLUXNET data.")
            continue

        plt.figure(figsize=(12, 7))
        for resolution, df in dataframes.items():
            if fluxtype == "NEE_WRF":
                df_loc_gee = df.filter(regex=f"^{location}_(.*_GPP_WRF)$")
                df_loc_res = df.filter(regex=f"^{location}_(.*_RECO_WRF)$")
                df_loc_gpp_pmodel = df.filter(regex=f"^{location}_(.*_GPP_Pmodel)$")
                df_loc_reco_migli = df.filter(regex=f"^{location}_(.*_RECO_Migli)$")
                df_loc = df_loc_res.add(df_loc_gee, fill_value=0)
                df_loc = df_loc.add(df_loc_gpp_pmodel, fill_value=0)
                df_loc = df_loc.add(df_loc_reco_migli, fill_value=0)
                nee_columns = (
                    {}
                )  # Dynamically create NEE columns by summing corresponding GPP and RECO columns
                nee_columns_P = {}
                for col in df_loc.columns:
                    if "_GPP_WRF" in col:
                        prefix = col.replace("_GPP_WRF", "")
                        if f"{prefix}_RECO_WRF" in df_loc.columns:
                            nee_columns[f"{prefix}_NEE_WRF"] = (
                                f"{prefix}_RECO_WRF",
                                f"{prefix}_GPP_WRF",
                            )
                for col in df_loc.columns:
                    if "_GPP_Pmodel" in col:
                        prefix = col.replace("_GPP_Pmodel", "")
                        if f"{prefix}_RECO_Migli" in df_loc.columns:
                            nee_columns_P[f"{prefix}_NEE_PMigli"] = (
                                f"{prefix}_RECO_Migli",
                                f"{prefix}_GPP_Pmodel",
                            )

                for nee_col, (res_col, gee_col) in nee_columns.items():
                    df_loc[nee_col] = df_loc[res_col] + df_loc[gee_col]
                df_loc = df_loc.drop(
                    columns=df_loc.filter(regex="(_GPP_WRF|_RECO_WRF)$").columns
                )
                for nee_col_P, (reco_col, gpp_col) in nee_columns_P.items():
                    df_loc[nee_col_P] = df_loc[reco_col] - df_loc[gpp_col]
                df_loc = df_loc.drop(
                    columns=df_loc.filter(regex="(_GPP_Pmodel|_RECO_Migli)$").columns
                )
            elif fluxtype == "T2_WRF":
                df_loc = df.filter(
                    regex=f"^{location}_REF(.*_{fluxtype})$"
                )  # Match location and variable pattern
                df_loc = df_loc - 273.15
            elif fluxtype == "GPP_WRF":
                df_loc = -df.filter(
                    regex=f"^{location}_(.*_{fluxtype})$"
                )  # turning GPP to positive values
                # add gpp from pmodel to df_loc
                df_loc_pmodel = df.filter(regex=f"^{location}_(.*ALPS_GPP_Pmodel)$")
                # replace nans with zero in pmodel
                df_loc_pmodel = df_loc_pmodel.fillna(0)
                df_loc = df_loc.add(df_loc_pmodel, fill_value=0)
            elif fluxtype == "RECO_WRF":
                df_loc = df.filter(regex=f"^{location}_(.*_{fluxtype})$")
                # add reco from migliavacca to df_loc
                df_loc_migli = df.filter(regex=f"^{location}_(.*ALPS_RECO_Migli)$")
                df_loc = df_loc.add(df_loc_migli, fill_value=0)
            print(f"{resolution}: {df_loc.columns.tolist()}")
            for column in df_loc.columns:
                variable = (
                    column.split("_")[-3]
                    + "_"
                    + column.split("_")[-2]
                    + "_"
                    + column.split("_")[-1]
                )  # Extract the variable type (e.g., ref_GPP_WRF, std_GPP_WRF)
                if variable in variables:
                    df_loc["hour"] = df_loc.index.hour
                    # Check if 'TIMESTAMP_START' is the index already and is a DatetimeIndex
                    # Only convert and set index if the column 'TIMESTAMP_START' exists
                    if "TIMESTAMP_START" in df_FLX_site.columns:
                        df_FLX_site["TIMESTAMP_START"] = pd.to_datetime(
                            df_FLX_site["TIMESTAMP_START"]
                        ).dt.tz_localize(None)
                        df_FLX_site = df_FLX_site.set_index("TIMESTAMP_START")

                    # Now filter by df_loc index
                    df_FLX_site = df_FLX_site[df_FLX_site.index.isin(df_loc.index)]

                    # Align index/order exactly
                    df_FLX_site = df_FLX_site.loc[df_loc.index]

                    hourly_avg = df_loc.groupby("hour")[column].mean()
                    # mask out nan values from both time series
                    # filter values in df_FLX_site based on the dates of df_loc
                    if df_loc[column].isna().any():
                        mask = ~df_loc[column].isna()
                        clean_values_FLX = df_FLX_site[var_flx][mask.tolist()].tolist()
                        clean_values_loc = df_loc[column][mask].tolist()
                        evaluator = RegressionMetric(
                            clean_values_FLX,
                            clean_values_loc,
                        )
                    else:
                        evaluator = RegressionMetric(
                            df_FLX_site[var_flx].tolist(),
                            df_loc[column].tolist(),
                        )
                    metrics = evaluator.get_metrics_by_list_names(["RMSE", "R2"])
                    # save metric data for csv
                    for metric, value in metrics.items():
                        consolidated_metrics_df.loc[
                            metric, column + "_" + resolution
                        ] = value  # Assign metric value to the appropriate column

                    linestyle = variables[variable]
                    color_i = resolution_colors[resolution]
                    if (
                        variable == "std_GPP_Pmodel"
                        or variable == "std_RECO_Migli"
                        or variable == "std_NEE_PMigli"
                    ):
                        color_i = "purple"

                    # diff_hgt = locations_hgt[location] - int(
                    #     site_match[f"model_hgt_NN"].iloc[0]
                    # )
                    # print(f"diff_ght={diff_hgt} at location {location}")

                    plt.plot(
                        hourly_avg,  # Skipping the first value as requested
                        label=(
                            f"{resolution}_{variable} RMSE={metrics['RMSE']:.2f} R2={metrics['R2']:.2f}"
                        ),
                        linestyle=linestyle,
                        color=color_i,
                    )

        if plot_CAMS:
            if location in df_CAMS_hourly_all:
                df_CAMS_hourly = df_CAMS_hourly_all[location]
            df_CAMS_hourly["hour"] = df_CAMS_hourly.index.hour
            # # Ensure both DataFrames have the same timezone awareness
            df_FLX_site.index = df_FLX_site.index.tz_localize(None)
            df_CAMS_hourly.index = df_CAMS_hourly.index.tz_localize(None)
            # Merge df_FLX_site and df_CAMS_hourly on their timestamps
            df_merged = pd.merge(
                df_FLX_site,
                df_CAMS_hourly,
                left_index=True,
                right_index=True,
                suffixes=("_FLX", "_CAMS"),
            )
            cols_to_fix = [
                "fco2gpp",
                "fco2gpp_bfas",
                "fco2rec",
                "fco2rec_bfas",
                "fco2nee",
                "fco2nee_bfas",
                "t2m",
            ]
            df_merged[cols_to_fix] = df_merged[cols_to_fix].apply(
                pd.to_numeric, errors="coerce"
            )
            df_merged[cols_to_fix] = df_merged[cols_to_fix].interpolate(method="linear")
            cams_l = df_merged[var_CAMS_plot]

            evaluator = RegressionMetric(
                df_FLX_site[var_flx].tolist(),
                cams_l.tolist(),
            )
            metrics = evaluator.get_metrics_by_list_names(["RMSE", "R2"])
            # save metric data for csv
            for metric, value in metrics.items():
                consolidated_metrics_df.loc[
                    metric, var_CAMS_plot + "_" + resolution
                ] = value  # Assign metric value to the appropriate column

            if var_CAMS_plot == "t2m":
                hourly_avg_CAMS = df_CAMS_hourly.interpolate(method="linear")
                hourly_avg_CAMS = df_CAMS_hourly.groupby("hour")[var_CAMS_plot].mean()
                plt.plot(
                    hourly_avg_CAMS,
                    label=f"CAMS {var_CAMS_plot}  R2={metrics['R2']:.2f} ",
                    color="orange",
                )
            else:
                hourly_avg_CAMS = df_CAMS_hourly.interpolate(method="linear")
                hourly_avg_CAMS = df_CAMS_hourly.groupby("hour")[var_CAMS_plot].mean()
                plt.plot(
                    hourly_avg_CAMS,
                    label=f"CAMS {var_CAMS_plot}",
                    color="orange",
                    linestyle="dashed",
                )
                hourly_avg_CAMS_bfas = df_CAMS_hourly.groupby("hour")[
                    var_CAMS_plot + "_bfas"
                ].mean()
                plt.plot(
                    hourly_avg_CAMS_bfas,
                    label=f"CAMS {var_CAMS_plot}_bfas corr. RMSE={metrics['RMSE']:.2f}  R2={metrics['R2']:.2f} ",
                    color="orange",
                )

        df_FLX_site["hour"] = df_FLX_site.index.hour
        hourly_avg_FLX = df_FLX_site.groupby("hour")[var_flx].mean()
        plt.plot(
            hourly_avg_FLX,
            label=f"FLUXNET {var_flx}",
            linestyle="solid",
            color="black",
        )

        factor_fontsize = 1.8
        plt.xlabel("UTC [h]", fontsize=14 * factor_fontsize)
        if unit == "[°C]":
            plt.ylabel(r"T$_{2m}$ [°C]", fontsize=14 * factor_fontsize)
        else:
            flux_label = fluxtype.split("_")[0]
            plt.ylabel(f"{flux_label} [μmol/m²/s]", fontsize=14 * factor_fontsize)
        plt.tick_params(labelsize=12 * factor_fontsize)
        plt.legend(
            loc="center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=2,
            fontsize=8 * factor_fontsize,
        )
        plt.xticks([0, 6, 12, 18, 24])
        plt.grid()
        plt.tight_layout()
        if sim_type == "":
            sim_type = "_clear_sky"
            plt.savefig(
                f"{outfolder}/{location}_{fluxtype}_comparison_hourly_{resolution}{sim_type}_{timespan}.pdf",
                bbox_inches="tight",
            )  # Save plot as PNG
        else:
            plt.savefig(
                f"{outfolder}/{location}_{fluxtype}_comparison_hourly_{resolution}{sim_type}_{timespan}.pdf",
                bbox_inches="tight",
            )  # Save plot as PNG
        plt.close()

        # append consolidated_metrics_df to consolidated_metrics_all
        consolidated_metrics_all = pd.concat(
            [consolidated_metrics_all, consolidated_metrics_df], axis=1
        )

        print(
            f"finished: {location}_{fluxtype}_comparison_hourly_{resolution}_{timespan}"
        )

consolidated_metrics_all.to_csv(
    f"{outfolder}/Validation_FLUXNET_hourly{sim_type}_{timespan}.csv"
)
