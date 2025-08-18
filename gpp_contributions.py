import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import binary_erosion, distance_transform_edt
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

def styled_imshow_plot(data, vmin, vmax, cmap, label, filename):
    fig, ax = plt.subplots(
        figsize=(12, 15), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    im = ax.imshow(
        data,
        extent=[lons_fine.min(), lons_fine.max(), lats_fine.min(), lats_fine.max()],
        cmap=cmap,
        origin="lower",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(
        im, ax=ax, orientation="vertical", shrink=0.3, fraction=0.046, pad=0.06
    )
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label, fontsize=20)

    gl = ax.gridlines(
        draw_labels=True, linewidth=1.5, color="black", alpha=0.2, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 20}
    gl.ylabel_style = {"size": 20}

    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.5)

    plt.tight_layout()
    if save_plot:
        plt.savefig(f"{plots_folder}{filename}_{datetime}h.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def generate_coastal_mask(
    veg_type: np.ndarray, buffer_km: float = 50.0, grid_spacing_km: float = 3.0
) -> np.ndarray:
    """
    Returns a new landmask where any point within `buffer_km` of the coastline is set to 0 (masked).

    Args:
        veg_type: 2D array with 44 for water.
        buffer_km: Distance from coastline to mask, in km.
        grid_spacing_km: Grid spacing in km (e.g., 3 for WRF 1km grid).

    Returns:
        new_landmask: same shape as input, with coastal zone masked out.
    """
    # Create landmask: 1 for land, 0 for water
    landmask = np.ones_like(veg_type, dtype=np.uint8)
    landmask[veg_type == 44] = 0
    land_binary = landmask.astype(bool)
    eroded_land = binary_erosion(land_binary)
    coastline = land_binary & (~eroded_land)

    # Compute distance (in grid cells) from coastline
    distance = distance_transform_edt(~coastline) * grid_spacing_km

    # Mask everything within `buffer_km` from coastline
    new_landmask = landmask.copy()
    new_landmask[distance <= buffer_km] = 0

    return new_landmask

def proj_on_finer_WRF_grid(
    lats_coarse, lons_coarse, var_coarse, lats_fine, lons_fine, WRF_var_1km, method_in
):
    proj_var = griddata(
        (lats_coarse.flatten(), lons_coarse.flatten()),
        var_coarse.flatten(),
        (lats_fine, lons_fine),
        method=method_in,
    ).reshape(WRF_var_1km.shape)
    return proj_var




############# INPUT ############
plots_folder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/areafluxes_"
save_plot = False
dx = "_54km"
interp_method = "nearest"  # 'linear', 'nearest', 'cubic'
STD_TOPO = 200
##############################

shares_norm_df = pd.DataFrame(columns=["Tscale", "Wscale", "Pscale", "Rnorm"]) # , "EVI" TODO: fix EVI
for time in range(5,19):

    datetime = f"2012-07-27_{time:02d}"
    date = datetime.split("_")[0]
    subfolder = ""  # "" or "_cloudy" TODO _rainy
    wrfinput_path_1km = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_1km{subfolder}/wrfout_d02_{datetime}:00:00"
    wrfinput_path_d01 = f"/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS{dx}{subfolder}/wrfout_d01_{datetime}:00:00"
    vprm_input_path_1km = f"/scratch/c7071034/DATA/VPRM_input/vprm_corine_1km/vprm_input_d02_{date}_00:00:00.nc"
    vprm_input_path_d01 = f"/scratch/c7071034/DATA/VPRM_input/vprm_corine{dx}/vprm_input_d01_{date}_00:00:00.nc"


    # Load the NetCDF file
    conv_factor = 1 / 3600
    nc_fid1km = nc.Dataset(wrfinput_path_1km, "r")
    nc_fid54km = nc.Dataset(wrfinput_path_d01, "r")
    GPP_WRF_1km = -nc_fid1km.variables["EBIO_GEE"][0, 0, 10:-10, 10:-10] * conv_factor
    GPP_WRF_d01 = -nc_fid54km.variables["EBIO_GEE"][0, 0, :, :] * conv_factor
    SWDOWN_1km = nc_fid1km.variables["SWDOWN"][0, 10:-10, 10:-10]
    SWDOWN_d01 = nc_fid54km.variables["SWDOWN"][0]
    T2_1km = nc_fid1km.variables["T2"][0, 10:-10, 10:-10] - 273.15
    T2_d01 = nc_fid54km.variables["T2"][0] - 273.15
    lats_fine = nc_fid1km.variables["XLAT"][0, 10:-10, 10:-10]
    lons_fine = nc_fid1km.variables["XLONG"][0, 10:-10, 10:-10]
    stdh_topo_1km = nc_fid1km.variables["VAR"][0, 10:-10, 10:-10]
    stdh_mask = stdh_topo_1km >= STD_TOPO
    lats_d01 = nc_fid54km.variables["XLAT"][0, :, :]
    lons_d01 = nc_fid54km.variables["XLONG"][0, :, :]
    veg_type = nc_fid54km.variables["IVGTYP"][0, :, :]
    new_landmask = generate_coastal_mask(veg_type, buffer_km=30.0, grid_spacing_km=50.0)

    # --- Load vvprm_input ---
    ds = xr.open_dataset(vprm_input_path_1km)
    ds_d01 = xr.open_dataset(vprm_input_path_d01)
    # ['Times', 'XLONG', 'XLAT', 'EVI_MIN', 'EVI_MAX', 'EVI', 'LSWI_MIN', 'LSWI_MAX', 'LSWI', 'VEGFRA_VPRM']
    veg_frac_map = ds["VEGFRA_VPRM"].isel(
        south_north=slice(10, -10), west_east=slice(10, -10)
    )
    veg_frac_map_d01 = ds_d01["VEGFRA_VPRM"]

    evi_map = ds["EVI"].isel(
        south_north=slice(10, -10), west_east=slice(10, -10)
    )
    evi_map_d01 = ds_d01["EVI"]
    evi_min_map = ds["EVI_MIN"].isel(
        south_north=slice(10, -10), west_east=slice(10, -10)
    )
    evi_min_map_d01 = ds_d01["EVI_MIN"]
    evi_max_map = ds["EVI_MAX"].isel(
        south_north=slice(10, -10), west_east=slice(10, -10)
    )
    evi_max_map_d01 = ds_d01["EVI_MAX"]
    lswi_map = ds["LSWI"].isel(
        south_north=slice(10, -10), west_east=slice(10, -10)
    )
    lswi_map_d01 = ds_d01["LSWI"]
    lswi_min_map = ds["LSWI_MIN"].isel(
        south_north=slice(10, -10), west_east=slice(10, -10)
    )
    lswi_min_map_d01 = ds_d01["LSWI_MIN"]
    lswi_max_map = ds["LSWI_MAX"].isel(
        south_north=slice(10, -10), west_east=slice(10, -10)
    )
    lswi_max_map_d01 = ds_d01["LSWI_MAX"]
    # --- define parameters ---
    SWDOWN_TO_PAR = 0.505
    PAR0_of_PFT = {
        "ENF": 316.96,
        "DBF": 310.78,
        "MF": 428.83,
        "SHB": 363.00,
        "SAV": 682.00,
        "CRO": 595.86,
        "GRA": 406.28,
        "OTH": 0.00,
    }
    lambda_of_PFT = { 
        "ENF": 0.304,
        "DBF": 0.216,
        "MF": 0.114,
        "SHB": 0.0874,
        "SAV": 0.114,
        "CRO": 0.140,
        "GRA": 0.448,
        "OTH": 0.00,
    } 
    Tvar_of_PFT = {
        "ENF": (14.250, 0, 40),
        "DBF": (23.580, 0, 40),
        "MF": (17.440, 0, 40),
        "SHB": (20.000, 0, 40),
        "SAV": (20.000, 0, 40),
        "CRO": (22.000, 0, 40),
        "GRA": (15.880, 0, 40),
        "OTH": (0.000, 0, 40),
    }

    # calculate scaled values for 1km resolution
    RADscale_1km = np.zeros_like(SWDOWN_1km)
    Tscale_1km = np.zeros_like(SWDOWN_1km)
    Wscale_1km = np.zeros_like(SWDOWN_1km)
    Pscale_1km = np.zeros_like(SWDOWN_1km)
    EVI_1km = np.zeros_like(SWDOWN_1km)
    #Residual_1km = np.zeros_like(SWDOWN_1km)
    RADscale_d01 = np.zeros_like(SWDOWN_d01)
    Tscale_d01 = np.zeros_like(SWDOWN_d01)
    Wscale_d01 = np.zeros_like(SWDOWN_d01)
    Pscale_d01 = np.zeros_like(SWDOWN_d01)
    EVI_d01 = np.zeros_like(SWDOWN_d01)
    #Residual_d01 = np.zeros_like(SWDOWN_d01)
    GPP_test_1km = np.zeros_like(SWDOWN_1km)
    GPP_test_d01 = np.zeros_like(SWDOWN_d01)
    eps = 1e-7  # numerical safeguard

    for m in range(7):
        vegfrac = np.nan_to_num(veg_frac_map[0, m, :, :].values, nan=0.0)

        # --- RADscale ---
        PAR0 = PAR0_of_PFT[list(Tvar_of_PFT.keys())[m]]
        if PAR0 > 0:
            RADscale_temp = (
                (1 / (1 + (SWDOWN_1km * SWDOWN_TO_PAR) / PAR0))
                * SWDOWN_1km
                * SWDOWN_TO_PAR
            ) 
            RADscale_temp = np.nan_to_num(RADscale_temp, nan=0.0)
            RADscale_1km += RADscale_temp * vegfrac / np.nanmax(RADscale_temp)
        
        # --- Tscale ---
        a1 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][1]
        a2 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][2]
        a3 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][0]
        Tscale_tmp = np.where((a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3 ** 2))
        Tscale_tmp = np.nan_to_num(Tscale_tmp, nan=0.0)
        Tscale_1km += np.where(Tscale_tmp < 0, 0, Tscale_tmp) * vegfrac

        # --- Wscale ---
        if m == 3 or m == 6:  # cropland / grassland
            num = lswi_map[0, m, :, :].values - lswi_min_map[0, m, :, :].values
            den = lswi_max_map[0, m, :, :].values - lswi_min_map[0, m, :, :].values
            frac = np.where(np.abs(den) < eps, 0, num / den)
            Wscale_temp_1km = frac
        else:
            Wscale_temp_1km = (1 + lswi_map[0, m, :, :].values) / (1 + lswi_max_map[0, m, :, :].values)
        Wscale_temp_1km = np.nan_to_num(Wscale_temp_1km, nan=0.0)
        Wscale_1km += Wscale_temp_1km * vegfrac
        

        # --- Pscale ---
        if m == 0:  # evergreen
            Pscale_temp_1km = np.zeros_like(SWDOWN_1km)
        elif m == 4 or m == 6:  # savanna / grassland
            Pscale_temp_1km = (1 + lswi_map[0, m, :, :].values) / 2.0 
        else:
            evithresh = evi_min_map[0, m, :, :].values + 0.55 * (
                evi_max_map[0, m, :, :].values - evi_min_map[0, m, :, :].values
            )
            Pscale_temp_1km = np.where(
                evi_map[0, m, :, :].values >= evithresh,
                1.0,
                (1 + lswi_map[0, m, :, :].values) / 2.0,
            ) 
        Pscale_temp_1km = np.nan_to_num(Pscale_temp_1km, nan=0.0)
        Pscale_1km += Pscale_temp_1km * vegfrac
        
        # --- EVI ---
        EVI_temp_1km = np.nan_to_num(evi_map[0, m, :, :].values, nan=0.0)
        EVI_1km += EVI_temp_1km * vegfrac

        # --- Residual ---
        GPP_temp_1km = lambda_of_PFT[list(lambda_of_PFT.keys())[m]] * Tscale_tmp  * Wscale_temp_1km * Pscale_temp_1km * RADscale_temp * evi_map[0, m, :, :]
        GPP_temp_1km[np.isnan(GPP_temp_1km)] = 0  # set nan values of GPP_temp_1km to zeros
        GPP_test_1km += GPP_temp_1km * vegfrac
        #Residual_1km += (lambda_of_PFT[list(lambda_of_PFT.keys())[m]] * RADscale_temp * evi_map[0, m, :, :]) * vegfrac

        ###############################  Domain d01 #################################
        vegfrac_d01 = veg_frac_map_d01[0, m, :, :].values
        # --- RADscale ---
        PAR0 = PAR0_of_PFT[list(Tvar_of_PFT.keys())[m]]
        if PAR0 > 0:
            RADscale_temp_d01 = (
                (1 / (1 + (SWDOWN_d01 * SWDOWN_TO_PAR) / PAR0))
                * SWDOWN_d01
                * SWDOWN_TO_PAR
            ) 
            RADscale_d01 += RADscale_temp_d01 * vegfrac_d01 / np.nanmax(RADscale_temp_d01)

        # --- Tscale ---
        a1 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][1]
        a2 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][2]
        a3 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][0]
        Tscale_tmp_d01 = np.where((a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3 ** 2))
        Tscale_tmp_d01[np.isnan(Tscale_tmp_d01)] = 0  # set nan values of Tscale_d01 to zeros
        Tscale_d01 += np.where(Tscale_tmp_d01 < 0, 0, Tscale_tmp_d01) * vegfrac_d01

        # --- Wscale ---
        if m == 3 or m == 6:  # cropland / grassland
            num = lswi_map_d01[0, m, :, :].values - lswi_min_map_d01[0, m, :, :].values
            den = lswi_max_map_d01[0, m, :, :].values - lswi_min_map_d01[0, m, :, :].values
            frac = np.where(np.abs(den) < eps, 0, num / den)
            Wscale_temp_d01 = frac
        else:
            Wscale_temp_d01 = (1 + lswi_map_d01[0, m, :, :].values) / (1 + lswi_max_map_d01[0, m, :, :].values)
        Wscale_temp_d01[np.isnan(Wscale_temp_d01)] = 0 #  set nan values of Wscale_d01 to zeros
        Wscale_d01 += Wscale_temp_d01 * vegfrac_d01
        
        # --- Pscale ---
        if m == 0:  # evergreen
            Pscale_temp_d01 = np.zeros_like(SWDOWN_d01)
        elif m == 4 or m == 6:  # savanna / grassland
            Pscale_temp_d01 = (1 + lswi_map_d01[0, m, :, :].values) / 2.0 
        else:
            evithresh = evi_min_map_d01[0, m, :, :].values + 0.55 * (
                evi_max_map_d01[0, m, :, :].values - evi_min_map_d01[0, m, :, :].values
            )
            Pscale_temp_d01 = np.where(
                evi_map_d01[0, m, :, :].values >= evithresh,
                1.0,
                (1 + lswi_map_d01[0, m, :, :].values) / 2.0,
            ) 
        Pscale_temp_d01[np.isnan(Pscale_temp_d01)] = 0
        Pscale_d01 += Pscale_temp_d01 * vegfrac_d01
        # --- EVI ---
        EVI_temp_d01 = np.nan_to_num(evi_map_d01[0, m, :, :].values, nan=0.0)
        EVI_d01 += EVI_temp_d01 * vegfrac_d01

        # --- Residual ---
        GPP_temp_d01 = lambda_of_PFT[list(lambda_of_PFT.keys())[m]] * Tscale_tmp_d01 * Wscale_temp_d01 * Pscale_temp_d01 * RADscale_temp_d01 * evi_map_d01[0, m, :, :]
        GPP_temp_d01[np.isnan(GPP_temp_d01)] = 0  # set nan values of GPP_temp_d01 to zeros
        GPP_test_d01 += GPP_temp_d01 * vegfrac_d01
        # Residual_d01 += (lambda_of_PFT[list(lambda_of_PFT.keys())[m]] * RADscale_temp_d01 * evi_map_d01[0, m, :, :]) * vegfrac_d01

    proj_landmask_d01 = proj_on_finer_WRF_grid(
        lats_d01,
        lons_d01,
        new_landmask,
        lats_fine,
        lons_fine,
        SWDOWN_1km,
        interp_method,
    )

    proj_SWDOWN_d01 = proj_on_finer_WRF_grid(
        lats_d01,
        lons_d01,
        SWDOWN_d01,
        lats_fine,
        lons_fine,
        SWDOWN_1km,
        interp_method,
    )

    proj_RADscale_d01 = proj_on_finer_WRF_grid(
        lats_d01,
        lons_d01,
        RADscale_d01,
        lats_fine,
        lons_fine,
        RADscale_1km,
        interp_method,
    )
    proj_Tscale_d01 = proj_on_finer_WRF_grid(
        lats_d01,
        lons_d01,
        Tscale_d01,
        lats_fine,
        lons_fine,
        Tscale_1km,
        interp_method,
    )

    proj_Wscale_d01 = proj_on_finer_WRF_grid(
        lats_d01,
        lons_d01,
        Wscale_d01,
        lats_fine,
        lons_fine,
        Wscale_1km,
        interp_method,
    )

    proj_Pscale_d01 = proj_on_finer_WRF_grid(
        lats_d01,
        lons_d01,
        Pscale_d01,
        lats_fine,
        lons_fine,
        Pscale_1km,
        interp_method,
    )
    proj_EVI_d01 = proj_on_finer_WRF_grid(
        lats_d01,
        lons_d01,
        EVI_d01,
        lats_fine,
        lons_fine,
        EVI_1km,
        interp_method,
    )

    # --- Apply masks ---
    # convert true and false to 1 and zero
    stdh_mask_numeric  = stdh_mask.astype(int)
    common_mask = (proj_landmask_d01 * stdh_mask_numeric == 0)
    all_fields = [RADscale_1km, Tscale_1km, Wscale_1km, Pscale_1km,EVI_1km, GPP_WRF_1km,GPP_test_1km, 
                proj_RADscale_d01, proj_Tscale_d01,proj_Wscale_d01,proj_Pscale_d01, proj_EVI_d01] # Residual_1km

    for arr in all_fields:
        arr[common_mask] = np.nan

    GPP_diff = GPP_test_1km- GPP_WRF_1km
    print("GPP_test_1km: ",np.nanmean(GPP_test_1km))
    print("GPP_WRF_1km: ",np.nanmean(GPP_WRF_1km))
    print("GPP_diff: ",np.nanmean(GPP_diff))

    # # set all 0 values to nan for plotting
    # RADscale_1km[RADscale_1km == 0] = np.nan
    # Wscale_1km[Wscale_1km == 0] = np.nan
    # Pscale_1km[Pscale_1km == 0] = np.nan
    # Residual_1km[Residual_1km == 0] = np.nan
    # RADscale_d01[RADscale_d01 == 0] = np.nan
    # Pscale_d01[Pscale_d01 == 0] = np.nan
    # Wscale_d01[Wscale_d01 == 0] = np.nan
    # Residual_d01[Residual_d01 == 0] = np.nan



    dRADscale = proj_RADscale_d01 - RADscale_1km
    dTscale = proj_Tscale_d01 - Tscale_1km
    dWscale = proj_Wscale_d01 - Wscale_1km
    dPscale = proj_Pscale_d01 - Pscale_1km
    dEVI = proj_EVI_d01 - EVI_1km

    # dResidual = proj_Residual_d01 - Residual_1km

    if save_plot:
        styled_imshow_plot(
            dTscale,
            np.nanmin(dTscale),
            np.nanmax(dTscale),
            "YlOrRd",
            r"T$_{scale}$ [-]",
            "dTscale",
        )
        styled_imshow_plot(
            dWscale,
            np.nanmin(dWscale),
            np.nanmax(dWscale),
            "YlOrRd",
            r"W$_{scale}$ [-]",
            "dWscale",
        )
        styled_imshow_plot(
            dPscale,
            np.nanmin(dPscale),
            np.nanmax(dPscale),
            "YlOrRd",
            r"P$_{scale}$ [-]",
            "dPscale",
        )
        styled_imshow_plot(
            dEVI,
            np.nanmin(dEVI),
            np.nanmax(dEVI),
            "YlOrRd",
            "[-]",
            "dEVI",
        )
        styled_imshow_plot(
            Tscale_1km,
            np.nanmin(Tscale_1km),
            np.nanmax(Tscale_1km),
            "YlOrRd",
            r"T$_{scale}$ [-]",
            "Tscale",
        )
        styled_imshow_plot(
            Wscale_1km,
            np.nanmin(Wscale_1km),
            np.nanmax(Wscale_1km),
            "YlOrRd",
            r"W$_{scale}$ [-]",
            "Wscale",
        )
        styled_imshow_plot(
            Pscale_1km,
            np.nanmin(Pscale_1km),
            np.nanmax(Pscale_1km),
            "YlOrRd",
            r"P$_{scale}$ [-]",
            "Pscale",
        )
        styled_imshow_plot(
            EVI_1km,
            np.nanmin(EVI_1km),
            np.nanmax(EVI_1km),
            "YlOrRd",
            "[-]",
            r"EVI",
        )

    dRADscale_mean = np.nanmean(proj_RADscale_d01 - RADscale_1km)
    dTscale_mean = np.nanmean(proj_Tscale_d01 - Tscale_1km)
    dWscale_mean = np.nanmean(proj_Wscale_d01 - Wscale_1km)
    dPscale_mean = np.nanmean(proj_Pscale_d01 - Pscale_1km)
    dEVI_mean = np.nanmean(proj_EVI_d01 - EVI_1km)

    #dResidual_mean = np.nanmean(proj_Residual_d01 - Residual_1km)

    print("dRADscale", dRADscale_mean)
    print("dTscale", dTscale_mean)
    print("dWscale", dWscale_mean)
    print("dPscale", dPscale_mean)
    print("dEVI", dEVI_mean)
    # print("dResidual", dResidual_mean)

    def print_minmax(name, arr):
        arr = np.array(arr, dtype=float)
        print(f"{name:10s} min={np.nanmin(arr):.4e}, max={np.nanmax(arr):.4e}")

    print("--- Fine resolution (1 km) ---")
    print_minmax("RAD_1km", RADscale_1km)
    print_minmax("T_1km", Tscale_1km)
    print_minmax("W_1km", Wscale_1km)
    print_minmax("P_1km", Pscale_1km)
    print_minmax("EVI_1km", EVI_1km)

    print("--- Coarse resolution (d01) ---")
    print_minmax("RAD_d01", proj_RADscale_d01)
    print_minmax("T_d01", proj_Tscale_d01)
    print_minmax("W_d01", proj_Wscale_d01)
    print_minmax("P_d01", proj_Pscale_d01)
    print_minmax("EVI_d01", proj_EVI_d01)

    # TODO compare to GPP
    # --- Normalized LMDI decomposition (min-max normalization) ---
    # --- Helper: logarithmic mean ---
    def log_mean(a, b):
        """Logarithmic mean L(a,b) = (a-b)/(ln a - ln b)"""
        out = np.full_like(a, np.nan, dtype=float)
        mask = (a > 0) & (b > 0) & (a != b)
        out[mask] = (a[mask] - b[mask]) / (np.log(a[mask]) - np.log(b[mask]))
        # handle case a == b > 0 → limit is a
        mask_eq = (a == b) & (a > 0)
        out[mask_eq] = a[mask_eq]
        return out

    # --- Helper: enforce positivity ---
    def enforce_positive(x, eps=1e-6):
        x = np.array(x, dtype=float)
        mask = ~np.isnan(x) & (x <= 0)
        x[mask] = eps
        return x

    # --- Enforce positivity for LMDI stability ---
    # RADscale_1km     = enforce_positive(RADscale_1km)
    # Tscale_1km       = enforce_positive(Tscale_1km)
    # Wscale_1km       = enforce_positive(Wscale_1km)
    # Pscale_1km       = enforce_positive(Pscale_1km)
    # proj_RADscale_d01 = enforce_positive(proj_RADscale_d01)
    # proj_Tscale_d01   = enforce_positive(proj_Tscale_d01)
    # proj_Wscale_d01   = enforce_positive(proj_Wscale_d01)
    # proj_Pscale_d01   = enforce_positive(proj_Pscale_d01)

    # --- Define dictionaries of drivers ---
    drivers_fine = {
        "Tscale": Tscale_1km,
        "Wscale": Wscale_1km,
        "Pscale": Pscale_1km,
        "Rnorm": RADscale_1km,
        #"EVI": EVI_1km
    }
    drivers_coarse = {
        "Tscale": proj_Tscale_d01,
        "Wscale": proj_Wscale_d01,
        "Pscale": proj_Pscale_d01,
        "Rnorm": proj_RADscale_d01,
        #"EVI": proj_EVI_d01
    }

    # --- Normalized LMDI decomposition (min-max normalization) ---
    drivers_fine_norm = {k: (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) for k, v in drivers_fine.items()}
    drivers_coarse_norm = {k: (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) for k, v in drivers_coarse.items()}

    # Pixel-wise normalized LMDI contributions
    contrib_map_norm = {}
    for key in drivers_fine_norm.keys():
        coarse = drivers_coarse_norm[key]
        fine   = drivers_fine_norm[key]
        mask = (coarse > 0) & (fine > 0)
        contrib_map_norm[key] = np.full_like(coarse, np.nan, dtype=float)
        w = log_mean(coarse, fine)
        contrib_map_norm[key][mask] = w[mask] * np.log(coarse[mask] / fine[mask])

    # Domain mean normalized contributions
    C_norm = {k: np.nanmean(v) for k, v in contrib_map_norm.items()}
    total_norm = np.nansum(list(C_norm.values()))
    shares_norm = {k: 100 * v / total_norm for k, v in C_norm.items()}
    shares_norm_df.loc[datetime] = shares_norm

    # 
# Ensure the index is sorted and string type for plotting
shares_norm_df.index = shares_norm_df.index.astype(str)
shares_norm_df = shares_norm_df.sort_index()

# Plot as stacked bar
ax = shares_norm_df.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 5),
    colormap='tab20'
)

plt.ylabel('Normalized LMDI Share (%)')
plt.xlabel('Time')
plt.title('Normalized LMDI Contribution Shares Over Time')
plt.legend(title='Driver', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Annotate each segment with its percentage
for i, idx in enumerate(shares_norm_df.index):
    cum = 0
    for col in shares_norm_df.columns:
        height = shares_norm_df.loc[idx, col]
        if height > 0:
            plt.text(
                i, cum + height / 2, f"{height:.1f}%", 
                ha='center', va='center', fontsize=8, color='black'
            )
        cum += height

plt.show()

shares_norm_df.to_csv("shares_norm_timeseries.csv")


# Original Fortran Code
# DO j=jts,min(jte,jde-1)
# DO i=its,min(ite,ide-1)

#    GEE_frac= 0.
#    RESP_frac= 0.

#    Tair= T2(i,j)-273.15
#    veg_frac_loop: DO m=1,7

#         if (vprm_in(i,m,j,p_vegfra_vprm)<1.e-8) CYCLE  ! Then fluxes are zero

#         a1= Tair-Tmin(m)
#         a2= Tair-Tmax(m)
#         a3= Tair-Topt(m)

#         ! Here a1 or a2 can't be negative
#         if (a1<0. .OR. a2>0.) then
#             Tscale= 0.
#         else
#             Tscale=a1*a2/(a1*a2 - a3**2)
#         end if

#         if (Tscale<0.) then
#             Tscale=0.
#         end if

#        ! modification due to different dependency on ground water
#         if (m==4 .OR. m==7) then  ! grassland and shrubland are xeric systems
#             if (vprm_in(i,m,j,p_lswi_max)<1e-7) then  ! in order to avoid NaN for Wscale
#                 Wscale= 0.
#             else
#                 Wscale= (vprm_in(i,m,j,p_lswi)-vprm_in(i,m,j,p_lswi_min))/(vprm_in(i,m,j,p_lswi_max)-vprm_in(i,m,j,p_lswi_min))
#             end if
#         else
#             Wscale= (1.+vprm_in(i,m,j,p_lswi))/(1.+vprm_in(i,m,j,p_lswi_max))
#         end if

#         ! effect of leaf phenology
#         if (m==1) then  ! evegreen
#             Pscale= 1.
#         else if (m==5 .OR. m==7) then  ! savanna or grassland
#             Pscale= (1.+vprm_in(i,m,j,p_lswi))/2.
#         else                           ! Other vegetation types
#             evithresh= vprm_in(i,m,j,p_evi_min) + 0.55*(vprm_in(i,m,j,p_evi_max)-vprm_in(i,m,j,p_evi_min))
#             if (vprm_in(i,m,j,p_evi)>=evithresh) then  ! Full canopy period
#                Pscale= 1.
#             else
#                Pscale=(1.+vprm_in(i,m,j,p_lswi))/2.  ! bad-burst to full canopy period
#             end if
#         end if

#         RADscale= 1./(1. + RAD(i,j)/rad0(m))
#         GEE_frac= lambda(m)*Tscale*Pscale*Wscale*RADscale* vprm_in(i,m,j,p_evi)* RAD(i,j)*vprm_in(i,m,j,p_vegfra_vprm) + GEE_frac

#         RESP_frac= (alpha(m)*Tair + RESP0(m))*vprm_in(i,m,j,p_vegfra_vprm) + RESP_frac

# # calculate Xscale
# Tscale = np.zeros_like(evi_map)
# Wscale = np.zeros_like(evi_map)
# Pscale = np.zeros_like(evi_map)
# for m in range(6):
#     a1 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][0]
#     a2 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][1]
#     a3 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][2]
#     Tscale = np.where((a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3 ** 2))
#     Tscale = np.where(Tscale < 0, 0, Tscale)

#     if m == 4 or m == 7:
#         Wscale = np.where(lswi_max_map < 1e-7, 0, (lswi_map - lswi_min_map) / (lswi_max_map - lswi_min_map))
#     else:
#         Wscale = (1 + lswi_map) / (1 + lswi_max_map)

#     # effect of leaf phenology
#     if m == 1:  # evergreen
#         Pscale = 1.
#     elif m == 5 or m == 7:  # savanna or grassland
#         Pscale = (1 + lswi_map) / 2.
#     else:  # Other vegetation types
#         evithresh = lswi_min_map + 0.55 * (lswi_max_map - lswi_min_map)
#         if lswi_map >= evithresh:  # Full canopy period
#             Pscale = 1.
#         else:
#             Pscale = (1 + lswi_map) / 2.  # bad-burst to full canopy period