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

############# INPUT ############
save_plot_maps = False
save_plot2 = True
print_output = True
SWDOWN_TO_PAR = 1 # 0.505
plots_folder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/components_"
date = "2012-03-27" # "2012-06-28"
dx_all = ["_54km","_9km"] # 
interp_method = "nearest"  # 'linear', 'nearest', 'cubic'
STD_TOPO = 200
##############################

def plot_lin_pert_results(contribs_grid, residual, driver_names=None):
    """
    Plot contributions and residual from linear perturbation analysis.

    contribs_grid: array (6, ny, nx)
    residual: array (ny, nx)
    driver_names: list of 6 strings
    """
    if driver_names is None:
        driver_names = ['Lambda','T','W','P','R','E']

    n_drivers = contribs_grid.shape[0]
    ny, nx = contribs_grid.shape[1], contribs_grid.shape[2]

    fig, axes = plt.subplots(2, 4, figsize=(16,8))
    axes = axes.flatten()

    # Plot each driver contribution
    vmin = np.nanmin(contribs_grid)
    vmax = np.nanmax(contribs_grid)
    for i in range(n_drivers):
        im = axes[i].imshow(contribs_grid[i], origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i].set_title(driver_names[i])
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Plot residual
    im = axes[n_drivers].imshow(residual, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[n_drivers].set_title('Residual')
    axes[n_drivers].axis('off')
    fig.colorbar(im, ax=axes[n_drivers], fraction=0.046, pad=0.04)

    # Hide any remaining axes
    for j in range(n_drivers+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{plots_folder}_lin_pertubation_panels_{datetime}h_{dx}.pdf", bbox_inches="tight")

def linear_perturbation_analysis(
    GPP_1km, GPP_d01,
    Lambda_1km, Lambda_d01,
    T_1km, T_d01,
    W_1km, W_d01,
    P_1km, P_d01,
    R_1km, R_d01,
    E_1km, E_d01,
    regularize=True, alpha=1e-6
):
    # Flatten and convert to float
    GPP_1km = np.asarray(GPP_1km, dtype=np.float64)
    GPP_d01 = np.asarray(GPP_d01, dtype=np.float64)
    dG = (GPP_d01 - GPP_1km).ravel()

    drivers = [Lambda_d01-Lambda_1km,
               T_d01-T_1km,
               W_d01-W_1km,
               P_d01-P_1km,
               R_d01-R_1km,
               E_d01-E_1km]

    D = np.stack([np.asarray(d, dtype=np.float64).ravel() for d in drivers], axis=1)

    # Remove NaNs
    mask = np.all(np.isfinite(D), axis=1) & np.isfinite(dG)
    D_valid = D[mask]
    dG_valid = dG[mask]

    # Solve
    if regularize and alpha > 0.0:
        A = np.linalg.solve(np.dot(D_valid.T, D_valid) + alpha*np.eye(6),
                            np.dot(D_valid.T, dG_valid))
    else:
        A, *_ = np.linalg.lstsq(D_valid, dG_valid, rcond=None)

    # Contributions
    contribs_flat = D_valid * A  # (N_valid,6)

    # Fill full grid
    ny, nx = GPP_1km.shape
    contribs_grid = np.full((6, ny, nx), np.nan)
    residual = np.full((ny, nx), np.nan)
    valid_idx = np.where(mask)[0]
    for i in range(6):
        contribs_grid[i].flat[valid_idx] = contribs_flat[:, i]
    residual.flat[valid_idx] = dG_valid - contribs_flat.sum(axis=1)

    return A, contribs_grid, residual

def styled_imshow_plot_d01(data, vmin, vmax, cmap, label, filename):
    fig, ax = plt.subplots(
        figsize=(12, 15), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    im = ax.imshow(
        data,
        extent=[lons_d01.min(), lons_d01.max(), lats_d01.min(), lats_d01.max()],
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

    plt.savefig(f"{plots_folder}{filename}_{datetime}h.pdf", bbox_inches="tight")



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

lin_pert_mean_diffs_df = pd.DataFrame(columns=["dlambda", "dTscale", "dWscale", "dPscale","dPAR" , "dEVI","Residual"])
mean_diffs_df = pd.DataFrame(columns=["dlambda", "dTscale", "dWscale", "dPscale","dPAR" , "dEVI","dGPP","dGPP_validate","dGPP_lin_per"])
records = []
for dx in dx_all:
    for time in range(0,24):

        datetime = f"{date}_{time:02d}"
        date = datetime.split("_")[0]
        subfolder = "_pram_err"  # "" or "_cloudy" TODO _rainy
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
        eps = 1e-6  # numerical safeguard
        ds = xr.open_dataset(vprm_input_path_1km)
        ds_d01 = xr.open_dataset(vprm_input_path_d01)
        # ['Times', 'XLONG', 'XLAT', 'EVI_MIN', 'EVI_MAX', 'EVI', 'LSWI_MIN', 'LSWI_MAX', 'LSWI', 'VEGFRA_VPRM']
        veg_frac_map = ds["VEGFRA_VPRM"].isel(
            south_north=slice(10, -10), west_east=slice(10, -10)
        ).values
        veg_frac_map_d01 = ds_d01["VEGFRA_VPRM"].values
        veg_frac_map = np.nan_to_num(veg_frac_map, nan=0.0)
        veg_frac_map_d01 = np.nan_to_num(veg_frac_map_d01, nan=0.0)

        evi_map = ds["EVI"].isel(
            south_north=slice(10, -10), west_east=slice(10, -10)
        ).values
        evi_map_d01 = ds_d01["EVI"].values
        evi_min_map = ds["EVI_MIN"].isel(
            south_north=slice(10, -10), west_east=slice(10, -10)
        ).values
        evi_min_map_d01 = ds_d01["EVI_MIN"].values
        evi_max_map = ds["EVI_MAX"].isel(
            south_north=slice(10, -10), west_east=slice(10, -10)
        ).values
        evi_max_map_d01 = ds_d01["EVI_MAX"].values
        
        evi_map = np.nan_to_num(evi_map, nan=eps)
        evi_min_map = np.nan_to_num(evi_min_map, nan=eps)
        evi_max_map = np.nan_to_num(evi_max_map, nan=eps) 
        evi_map_d01 = np.nan_to_num(evi_map_d01, nan=eps)
        evi_min_map_d01 = np.nan_to_num(evi_min_map_d01, nan=eps)
        evi_max_map_d01 = np.nan_to_num(evi_max_map_d01, nan=eps)

        lswi_map = ds["LSWI"].isel(
            south_north=slice(10, -10), west_east=slice(10, -10)
        ).values
        lswi_map_d01 = ds_d01["LSWI"].values
        lswi_min_map = ds["LSWI_MIN"].isel(
            south_north=slice(10, -10), west_east=slice(10, -10)
        ).values
        lswi_min_map_d01 = ds_d01["LSWI_MIN"].values
        lswi_max_map = ds["LSWI_MAX"].isel(
            south_north=slice(10, -10), west_east=slice(10, -10)
        ).values
        lswi_max_map_d01 = ds_d01["LSWI_MAX"].values
        
        lswi_map = np.nan_to_num(lswi_map, nan=eps-1.0)
        lswi_min_map = np.nan_to_num(lswi_min_map, nan=eps-1.0)
        lswi_max_map = np.nan_to_num(lswi_max_map, nan=eps-1.0) 
        lswi_map_d01 = np.nan_to_num(lswi_map_d01, nan=eps-1.0)
        lswi_min_map_d01 = np.nan_to_num(lswi_min_map_d01, nan=eps-1.0)
        lswi_max_map_d01 = np.nan_to_num(lswi_max_map_d01, nan=eps-1.0)

        # --- define parameters ---
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

        # init arrays
        PAR_1km = np.zeros_like(SWDOWN_1km)
        Tscale_1km = np.zeros_like(SWDOWN_1km)
        Wscale_1km = np.zeros_like(SWDOWN_1km)
        Pscale_1km = np.zeros_like(SWDOWN_1km)
        EVI_1km = np.zeros_like(SWDOWN_1km)
        Lambda_1km = np.zeros_like(SWDOWN_1km)
        GPP_validate_1km = np.zeros_like(SWDOWN_1km)

        PAR_d01 = np.zeros_like(SWDOWN_d01)
        Tscale_d01 = np.zeros_like(SWDOWN_d01)
        Wscale_d01 = np.zeros_like(SWDOWN_d01)
        Pscale_d01 = np.zeros_like(SWDOWN_d01)
        EVI_d01 = np.zeros_like(SWDOWN_d01)        
        Lambda_d01 = np.zeros_like(SWDOWN_d01)        
        GPP_validate_d01 = np.zeros_like(SWDOWN_d01)

        for m in range(7):
            # --- vegetation fraction ---
            vegfrac = veg_frac_map[0, m, :, :]
            vegfrac = np.where(np.isnan(vegfrac), 0.0, vegfrac)   
            vegfrac = np.where(vegfrac < eps, 0.0, vegfrac)   

            # --- Lambda --- 
            Lambda_temp_1km = lambda_of_PFT[list(lambda_of_PFT.keys())[m]] * vegfrac / SWDOWN_TO_PAR
            Lambda_1km += Lambda_temp_1km

            # --- EVI ---
            EVI_temp_1km = evi_map[0, m, :, :]
            EVI_temp_1km = EVI_temp_1km 
            EVI_1km += EVI_temp_1km * vegfrac

            # --- PAR ---
            PAR0 = PAR0_of_PFT[list(Tvar_of_PFT.keys())[m]]
            if PAR0 > 0:
                PAR_temp = (
                    (1 / (1 + (SWDOWN_1km ) / (PAR0* SWDOWN_TO_PAR)))
                ) * SWDOWN_1km 
                if np.any(np.isnan(PAR_temp)):
                    # print("Count of NaNs in PAR_temp:", np.sum(np.isnan(PAR_temp)))
                    PAR_temp = np.nan_to_num(PAR_temp, nan=0.0)

                PAR_1km += PAR_temp * vegfrac 
            
            # --- Tscale ---
            a1 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][1]
            a2 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][2]
            a3 = T2_1km - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][0]
            Tscale_tmp = np.where((a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3 ** 2))
            Tscale_1km += np.where(Tscale_tmp < 0, 0, Tscale_tmp) * vegfrac

            # --- Wscale ---
            if m == 3 or m == 6:  # cropland / grassland
                num = lswi_map[0, m, :, :] - lswi_min_map[0, m, :, :]
                den = lswi_max_map[0, m, :, :] - lswi_min_map[0, m, :, :]
                den_nonzero = np.where(den < eps, eps, den)
                Wscale_temp_1km = num / den_nonzero
            else:
                Wscale_temp_1km = (1 + lswi_map[0, m, :, :]) / (1 + lswi_max_map[0, m, :, :])
            Wscale_temp_1km[np.isnan(Wscale_temp_1km)] = 0 
            Wscale_1km += Wscale_temp_1km * vegfrac

            # --- Pscale ---
            if m == 0:  # evergreen
                Pscale_temp_1km = np.ones_like(SWDOWN_1km)
            elif m == 4 or m == 6:  # savanna / grassland
                Pscale_temp_1km = (1 + lswi_map[0, m, :, :]) / 2.0 
            else:
                evithresh = evi_min_map[0, m, :, :] + 0.55 * (
                    evi_max_map[0, m, :, :] - evi_min_map[0, m, :, :]
                )
                Pscale_temp_1km = np.where(
                    evi_map[0, m, :, :] >= evithresh,
                    1.0,
                    (1 + lswi_map[0, m, :, :]) / 2.0,
                ) 
            if np.any(np.isnan(Pscale_temp_1km)):
                # print(f"Percentage of NaNs in Pscale_temp_1km for {list(lambda_of_PFT.keys())[m]}:", np.sum(np.isnan(Pscale_temp_1km)) / Pscale_temp_1km.size * 100)
                Pscale_temp_1km = np.nan_to_num(Pscale_temp_1km, nan=0.0)

            Pscale_1km += Pscale_temp_1km * vegfrac
            
            # --- GPP for comparison ---
            GPP_temp_1km = Lambda_temp_1km * Tscale_tmp  * Wscale_temp_1km * Pscale_temp_1km * PAR_temp * EVI_temp_1km
            GPP_temp_1km[GPP_temp_1km < 0] = 0
            GPP_validate_1km += GPP_temp_1km


            ###############################  Domain d01 #################################
            vegfrac_d01 = veg_frac_map_d01[0, m, :, :]
            vegfrac_d01 = np.where(np.isnan(vegfrac_d01), 0.0, vegfrac_d01)   
            vegfrac_d01 = np.where(vegfrac_d01 < eps, 0.0, vegfrac_d01)   

            # --- Lambda --- 
            Lambda_temp_d01 = lambda_of_PFT[list(lambda_of_PFT.keys())[m]] * vegfrac_d01 / SWDOWN_TO_PAR
            Lambda_d01 += Lambda_temp_d01
            
            # --- EVI ---
            EVI_temp_d01 = evi_map_d01[0, m, :, :]
            EVI_d01 += EVI_temp_d01 * vegfrac_d01

            # --- PAR ---
            PAR0 = PAR0_of_PFT[list(Tvar_of_PFT.keys())[m]]
            if PAR0 > 0:
                PAR_temp_d01 = (
                    (1 / (1 + (SWDOWN_d01 ) / (PAR0* SWDOWN_TO_PAR)))
                ) * SWDOWN_d01 
            PAR_d01 += PAR_temp_d01 * vegfrac_d01 

            # --- Tscale ---
            a1 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][1]
            a2 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][2]
            a3 = T2_d01 - Tvar_of_PFT[list(Tvar_of_PFT.keys())[m]][0]
            Tscale_tmp_d01 = np.where((a1 < 0) | (a2 > 0), 0, a1 * a2 / (a1 * a2 - a3 ** 2))
            Tscale_tmp_d01[np.isnan(Tscale_tmp_d01)] = 0  # set nan values of Tscale_d01 to zeros
            Tscale_d01 += np.where(Tscale_tmp_d01 < 0, 0, Tscale_tmp_d01) * vegfrac_d01

            # --- Wscale ---
            if m == 3 or m == 6:  # cropland / grassland
                num = lswi_map_d01[0, m, :, :] - lswi_min_map_d01[0, m, :, :]
                den = lswi_max_map_d01[0, m, :, :] - lswi_min_map_d01[0, m, :, :]
                den_nonzero = np.where(den < eps, eps, den)
                Wscale_temp_d01 = num / den_nonzero
            else:
                Wscale_temp_d01 = (1 + lswi_map_d01[0, m, :, :]) / (1 + lswi_max_map_d01[0, m, :, :])
            Wscale_temp_d01[np.isnan(Wscale_temp_d01)] = 0 
            Wscale_d01 += Wscale_temp_d01 * vegfrac_d01

            # --- Pscale ---
            if m == 0:  # evergreen
                Pscale_temp_d01 = np.ones_like(SWDOWN_d01)
            elif m == 4 or m == 6:  # savanna / grassland
                Pscale_temp_d01 = (1 + lswi_map_d01[0, m, :, :]) / 2.0 
            else:
                evithresh = evi_min_map_d01[0, m, :, :] + 0.55 * (
                    evi_max_map_d01[0, m, :, :] - evi_min_map_d01[0, m, :, :]
                )
                Pscale_temp_d01 = np.where(
                    evi_map_d01[0, m, :, :] >= evithresh,
                    1.0,
                    (1 + lswi_map_d01[0, m, :, :]) / 2.0,
                ) 
            Pscale_d01 += Pscale_temp_d01 * vegfrac_d01

            # --- GPP comparison ---
            GPP_temp_d01 = Lambda_temp_d01 * Tscale_tmp_d01 * Wscale_temp_d01 * Pscale_temp_d01 * PAR_temp_d01 * EVI_temp_d01
            GPP_temp_d01[GPP_temp_d01 < 0] = 0
            GPP_validate_d01 += GPP_temp_d01


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

        proj_PAR_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            PAR_d01,
            lats_fine,
            lons_fine,
            PAR_1km,
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
        proj_Lambda_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            Lambda_d01,
            lats_fine,
            lons_fine,
            Lambda_1km,
            interp_method,
        )
        proj_GPP_validate_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            GPP_validate_d01,
            lats_fine,
            lons_fine,
            GPP_validate_1km,
            interp_method,
        )
        proj_GPP_WRF_d01 = proj_on_finer_WRF_grid(
            lats_d01,
            lons_d01,
            GPP_WRF_d01,
            lats_fine,
            lons_fine,
            GPP_WRF_1km,
            interp_method,
        )

        # --- Apply masks ---
        # convert true and false to 1 and zero
        stdh_mask_numeric  = stdh_mask.astype(int)
        common_mask = (proj_landmask_d01 * stdh_mask_numeric == 0)
        all_fields = [PAR_1km, Tscale_1km, Wscale_1km, Pscale_1km,EVI_1km, GPP_WRF_1km,GPP_validate_1km, 
                    proj_PAR_d01, proj_Tscale_d01,proj_Wscale_d01,proj_Pscale_d01, proj_EVI_d01,proj_GPP_validate_d01,proj_GPP_WRF_d01] # Residual_1km

        for arr in all_fields:
            arr[common_mask] = np.nan

        GPP_diff_1km = GPP_validate_1km - GPP_WRF_1km
        GPP_diff_d01 = proj_GPP_validate_d01 - proj_GPP_WRF_d01

        if print_output:
            print("GPP_validate_1km: ",np.nanmean(GPP_validate_1km))
            print("GPP_WRF_1km: ",np.nanmean(GPP_WRF_1km))
            print("GPP_diff: ",np.nanmean(GPP_diff_1km))
            print("GPP_validate_1km NaN count: ",np.isnan(GPP_validate_1km).sum())
            print("GPP_WRF_1km NaN count: ",np.isnan(GPP_WRF_1km).sum())
            
            print("GPP_validate_d01: ",np.nanmean(proj_GPP_validate_d01))
            print("GPP_WRF_d01: ",np.nanmean(proj_GPP_WRF_d01))
            print("GPP_diff_d01: ",np.nanmean(GPP_diff_d01))
            print("GPP_validate_d01 NaN count: ",np.isnan(proj_GPP_validate_d01).sum())
            print("GPP_WRF_d01 NaN count: ",np.isnan(proj_GPP_WRF_d01).sum())

        dGPP = proj_GPP_WRF_d01 - GPP_WRF_1km
        dGPP_validate = proj_GPP_validate_d01 - GPP_validate_1km
        dPAR = proj_PAR_d01 - PAR_1km
        dTscale = proj_Tscale_d01 - Tscale_1km
        dWscale = proj_Wscale_d01 - Wscale_1km
        dPscale = proj_Pscale_d01 - Pscale_1km
        dEVI = proj_EVI_d01 - EVI_1km
        dLambda = proj_Lambda_d01 - Lambda_1km
        dPAR_mean = np.nanmean(dPAR)
        dTscale_mean = np.nanmean(dTscale)
        dWscale_mean = np.nanmean(dWscale)
        dPscale_mean = np.nanmean(dPscale)
        dEVI_mean = np.nanmean(dEVI)
        dLambda_mean = np.nanmean(dLambda)
        dGPP_mean = np.nanmean(dGPP)
        dGPP_validate_mean = np.nanmean(dGPP_validate)
        
        
        # --- linear_perturbation_analysis ---
        alphas, contribs, residual = linear_perturbation_analysis(
            GPP_validate_1km, proj_GPP_validate_d01,
            Lambda_1km, proj_Lambda_d01,
            Tscale_1km, proj_Tscale_d01,
            Wscale_1km, proj_Wscale_d01,
            Pscale_1km, proj_Pscale_d01,
            PAR_1km,proj_PAR_d01,
            EVI_1km, proj_EVI_d01,
            regularize=True, alpha=1e-6
        )
        # Then alphas gives your sensitivities; contribs[0] is ΔGPP_Lambda, contribs[1] is ΔGPP_T, etc.; residual is the unexplained fraction.
        driver_names = ["dlambda", "dTscale", "dWscale", "dPscale","dPAR" , "dEVI"]
            
        mean_contribs = np.nanmean(contribs, axis=(1,2))  # shape (6,)
        mean_residual = np.nanmean(residual)  
        lin_pert_mean_diffs_df.loc[datetime] = [mean_contribs[0],mean_contribs[1],mean_contribs[2],mean_contribs[3],mean_contribs[4],mean_contribs[5],mean_residual]

        lin_pert_mean = mean_contribs.sum()+mean_residual
        mean_diffs_df.loc[datetime] = [dLambda_mean, dTscale_mean, dWscale_mean, dPscale_mean, dPAR_mean, dEVI_mean,dGPP_mean,dGPP_validate_mean,lin_pert_mean]


        if print_output:               
            for name, val in zip(driver_names, mean_contribs):
                print(f"{name}: {val:.3f} [μmol/m²/s]")
            print(f"Residual: {mean_residual:.3f} [μmol/m²/s]")

        plot_lin_pert_results(contribs, residual,driver_names)

        if save_plot_maps:
            

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
                dWscale,
                np.nanmin(dWscale),
                np.nanmax(dWscale),
                "YlOrRd",
                r"W$_{scale}$ [-]",
                "dWscale",
            )
            styled_imshow_plot(
                dTscale,
                np.nanmin(dTscale),
                np.nanmax(dTscale),
                "YlOrRd",
                r"T$_{scale}$ [-]",
                "dTscale",
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
            # plot proj_Wscale_d01
            styled_imshow_plot_d01(
                Wscale_d01,
                np.nanmin(Wscale_d01),
                np.nanmax(Wscale_d01),
                "YlOrRd",
                r"W$_{scale}$ [-]",
                "Wscale_d01",
            )
        
    if save_plot2:

        # --- Subplot 1: Line plot of mean contributions ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(lin_pert_mean_diffs_df.index, lin_pert_mean_diffs_df[driver_names], marker='o')
        axes[0].plot(lin_pert_mean_diffs_df.index, lin_pert_mean_diffs_df['Residual'], marker='x', linestyle='--', color='k', label='Residual')
        axes[0].set_ylabel('Liner Perturbation ΔGPP [μmol/m²/s]')
        axes[0].legend(driver_names + ['Residual'], loc='upper left')
        # axes[0].set_title(f'Domain-averaged linear contributions over time {dx[1:]}')
        axes[0].grid(True)
        # --- Subplot 2: PAR bar + scale variables lines ---
        ax1 = axes[1]
        bars_df = mean_diffs_df[['dPAR']]
        bars_df.plot(kind='bar', ax=ax1, color='tab:purple', label='dPAR', alpha=0.5)
        ax1.set_ylabel('dPAR [μmol/m²/s]', color='tab:purple')
        ax1.tick_params(axis='y', labelcolor='tab:purple')
        # Second y-axis for scale variables
        ax2 = ax1.twinx()
        ax2.plot(mean_diffs_df.index, mean_diffs_df['dlambda'], color='tab:blue', label='dlambda')
        ax2.plot(mean_diffs_df.index, mean_diffs_df['dTscale'], color='tab:orange', label='dTscale')
        ax2.plot(mean_diffs_df.index, mean_diffs_df['dWscale'], color='tab:green', label='dWscale')
        ax2.plot(mean_diffs_df.index, mean_diffs_df['dPscale'], color='tab:red', label='dPscale')
        ax2.plot(mean_diffs_df.index, mean_diffs_df['dEVI'], color='tab:brown', label='dEVI')
        ax2.set_ylabel('Average of Scale Variables [dimensionless]')
        ax2.tick_params(axis='y')
        ax2.grid(True)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        # plt.show()

        plt.savefig(f"{plots_folder}lin_pert_mean_diffs_{dx[1:]}-1km_{date}.pdf", dpi=300)


    print(f"finished {dx}")