import os
import pandas as pd
import numpy as np
import xarray as xr
from pyrealm.pmodel import PModel, PModelEnvironment, SubdailyScaler, SubdailyPModel
from scipy.ndimage import uniform_filter1d


def migliavacca_LinGPP(
    T_ref, T0, E0, k_mm, k2, alpha_p, alpha_lai, max_lai, R_lai0, GPP, P, T_A
):
    GPP_gC_per_day = GPP * 12 * 86400 / 10**6
    R_ref = R_lai0 + alpha_lai * max_lai + k2 * GPP_gC_per_day
    f_T = np.exp(E0 * (1 / (T_ref - T0) - 1 / (T_A - T0)))
    f_P = (alpha_p * k_mm + P * (1 - alpha_p)) / (k_mm + P * (1 - alpha_p))

    reco_LinGPP_gC_per_day = R_ref * f_T * f_P
    reco_LinGPP_mumol_per_sec = reco_LinGPP_gC_per_day * 10**6 / (86400 * 12)
    return reco_LinGPP_mumol_per_sec


fixed_boundaries = {  # RLAI and alphaLAI are from literature while the others are tuned with fluxnet data
    "1": {
        "RLAI": 1.02,
        "alphaLAI": 0.42,
        "E0(K)": 128.69993785818411,
        "K (mm)": 7.601665827633045,
        "alpha_p": 0.48730251852457174,
        "k2": 0.79,
    },
    "2": {
        "RLAI": 1.27,
        "alphaLAI": 0.34,
        "E0(K)": 126.3734980228889,
        "K (mm)": 3.4082270937649963,
        "alpha_p": 0.7126491712262502,
        "k2": 0.2674530834652109,
    },
    "3": {
        "RLAI": 0.78,
        "alphaLAI": 0.44,
        "E0(K)": 49.549479391838965,
        "K (mm)": 6.026176584021398,
        "alpha_p": 0.8792355364845186,
        "k2": 0.4859112192032395,
    },
    "4": {
        "RLAI": 0.42,
        "alphaLAI": 0.57,
        "k2": 0.354,
        "E0(K)": 156.746,
        "alpha_p": 0.850,
        "K (mm)": 0.097,
    },
    "5": {
        "RLAI": 0.42,
        "alphaLAI": 0.57,
        "k2": 0.654,
        "E0(K)": 81.537,
        "alpha_p": 0.474,
        "K (mm)": 0.567,
    },
    "6": {
        "RLAI": 0.25,
        "alphaLAI": 0.40,
        "E0(K)": 137.94375730711272,
        "K (mm)": 3.00018616618101,
        "alpha_p": 1.0,
        "k2": 0.3891423973169539,
    },
    "7": {
        "RLAI": 0.41,
        "alphaLAI": 1.14,
        "E0(K)": 174.73276222784446,
        "K (mm)": 2.442778948014844,
        "alpha_p": 0.6826592853906017,
        "k2": 0.49396913212403937,
    },
    "8": {
        "RLAI": 0,
        "alphaLAI": 0,
        "E0(K)": 0,
        "K (mm)": 0,
        "alpha_p": 0,
        "k2": 0,
    },
}

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


# Load WRF dataset
# wrf_path = "/home/madse/Downloads/Fluxnet_Data/wrfout_d01_2012-07-01_12:00:00.nc"  # Replace with your file path
wrf_paths = [
    #   "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250107_155336_ALPS_3km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20250105_193347_ALPS_9km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241229_112716_ALPS_27km",
    "/scratch/c7071034/DATA/WRFOUT/WRFOUT_20241227_183215_ALPS_54km",
]

modis_path = "/scratch/c7071034/DATA/MODIS/MODIS_FPAR/"
migli_path = "/scratch/c7071034/DATA/RECO_Migli"

# pmodel parameters
days_mem = 37
half_wdth = 90
window_cent = 13
gC_to_mumol = 0.0833  # 1 µg C m⁻² s⁻¹ × (1 µmol C / 12.01 µg C) × (1 µmol CO₂ / 1 µmol C) = 0.0833 µmol CO₂ m⁻² s⁻¹

for wrf_path in wrf_paths:
    wrf_path_dx_str = wrf_path.split("_")[-1]
    # list all files in the wrf_path
    files = os.listdir(wrf_path)
    files = [f for f in files if f.startswith("wrfout_d01")]
    files.sort()
    timesteps = len(files)
    # get datetime from first file
    datetimestart = files[0].split("_")[2] + " " + files[0].split("_")[3]
    wrf_ds = xr.open_dataset(wrf_path + "/" + files[0])
    temp = wrf_ds["T2"].to_numpy()
    l, m, n = temp.shape
    fpar_modis_arr = np.zeros((timesteps, m, n))
    ppfd_arr = np.zeros((timesteps, m, n))
    tc_arr = np.zeros((timesteps, m, n))  # Store temperature time series
    co2_arr = np.zeros((timesteps, m, n))  # Store CO₂ time series
    patm_arr = np.zeros((timesteps, m, n))  # Store atmospheric pressure
    vpd_arr = np.zeros((timesteps, m, n))  # Store vapor pressure deficit
    rainc_arr = np.zeros((timesteps, m, n))  # Store rain rate
    lai_wrf_arr = np.zeros((timesteps, m, n))  # Store LAI

    IVGTYP = wrf_ds["IVGTYP"].to_numpy()
    IVGTYP_vprm = np.vectorize(corine_to_vprm.get)(
        IVGTYP[:, :]
    )  # Create a new array for the simplified vegetation categories
    # mask where IVGTYP_vprm is 8
    IVGTYP_mask = np.where(IVGTYP_vprm == 8, True, False)

    t = 0
    for file in files[1:]:
        day = int(file.split("_")[2].split("-")[2])

        if day > 30:
            continue
        wrf_ds = xr.open_dataset(wrf_path + "/" + file)

        # Load variables from WRF dataset
        temp = wrf_ds["T2"].to_numpy() - 273.15  # Convert to Celsius
        patm = wrf_ds["PSFC"].to_numpy()  # Pa
        co2 = wrf_ds["CO2_BIO"].isel(bottom_top=0).to_numpy()  # ppmv
        qvapor = (
            wrf_ds["QVAPOR"].isel(bottom_top=0).to_numpy()
        )  # Water vapor mixing ratio (kg/kg) at the surface level
        psfc = wrf_ds["PSFC"].isel(Time=0).to_numpy()  # Surface pressure (Pa)
        t2 = wrf_ds["T2"].isel(Time=0).to_numpy()  # Temperature at 2m (K)

        # Calculate actual vapor pressure (ea) in kPa
        ea = (qvapor * psfc) / (0.622 + qvapor)  # Pa
        # Calculate saturation vapor pressure (es) in kPa
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3)) * 1000  # convert to Pa
        # Calculate VPD
        vpd = np.maximum(0, es - ea)  # Force non-negative VPD

        # Load required variables from WRF dataset
        vegfra = wrf_ds["VEGFRA"].to_numpy()  # Vegetation fraction (0 to 1)
        albedo = wrf_ds["ALBEDO"].to_numpy()  # Albedo (0 to 1)
        swdown = wrf_ds["SWDOWN"].to_numpy()  # Downward shortwave radiation (W/m^2)
        ppfd = (
            swdown * 2.30785
        )  # Shortwave radiation (W/m²) × 0.505 -> PAR (W/m²) × 4.57 -> 2.3*x ~ PPFD (umol/m²/s)
        xlat = wrf_ds["XLAT"].to_numpy()  # Latitude (degrees)
        xlon = wrf_ds["XLONG"].to_numpy()  # Longitude (degrees)
        xlat = xlat[0, :, :]
        xlon = xlon[0, :, :]
        fapar_wrf = (1 - albedo) * (vegfra / 100)  # Calculate fAPAR

        # get modis fpar
        modis_path_in = f"{modis_path}fpar_interpol/interpolated_fpar_{wrf_path_dx_str}_2012-07-{day:02d}T12:00:00.nc"
        modis_ds = xr.open_dataset(modis_path_in)
        fpar_modis = modis_ds["Fpar_500m"].to_numpy()  # fAPAR from MODIS
        # where modis is nan use values from fapar_wrf
        fpar_modis = np.where(
            np.isnan(fpar_modis), fapar_wrf, fpar_modis
        )  # TODO: how is this handled in literature? There is no fPAR Data around cities, so these areas could also be masked out, but I would habe to modify the landcover maps...
        # set fpar_modis to 0 where IVGTYP_vprm is 8
        fpar_modis[IVGTYP_mask] = 0

        # Ensure proper dimensions and clean invalid data
        temp[temp < -25] = np.nan  # Mask temperatures below -25°C
        vpd = np.clip(vpd, 0, np.inf)  # Force VPD ≥ 0

        # # Run P-model environment
        # env = PModelEnvironment(tc=temp, co2=co2, patm=patm, vpd=vpd)
        # env.summarize()

        # # Estimate productivity
        # model = PModel(env)
        # model.estimate_productivity(fpar_modis, ppfd)
        # gC_to_mumol = 0.0833  # 1 µg C m⁻² s⁻¹ × (1 µmol C / 12.01 µg C) × (1 µmol CO₂ / 1 µmol C) = 0.0833 µmol CO₂ m⁻² s⁻¹
        # data = model.gpp[0, :, :] * gC_to_mumol
        # # save data in netcdf
        # date_time = file.split("_")[2] + "_" + file.split("_")[3]
        # modis_path_out = (
        #     f"{modis_path}gpp_pmodel/gpp_pmodel_{wrf_path_dx_str}_{date_time}.nc"
        # )
        # xr.DataArray(data, name="GPP_Pmodel").to_netcdf(
        #     modis_path_out, format="NETCDF4_CLASSIC"
        # )

        tc_arr[t, :, :] = temp[0, :, :]
        co2_arr[t, :, :] = co2[0, :, :]
        patm_arr[t, :, :] = patm[0, :, :]
        vpd_arr[t, :, :] = vpd[0, :, :]

        fpar_modis_arr[t, :, :] = fpar_modis[0, :, :]
        ppfd_arr[t, :, :] = ppfd[0, :, :]

        # get vars for migliavacca_LinGPP
        rainc = wrf_ds["RAINC"].to_numpy()
        rainc_arr[t, :, :] = rainc[0, :, :]
        lai_wrf = wrf_ds["LAI"].to_numpy()
        lai_wrf_arr[t, :, :] = lai_wrf[0, :, :]

        t += 1

    env_arr = PModelEnvironment(tc=tc_arr, co2=co2_arr, patm=patm_arr, vpd=vpd_arr)
    env_arr.summarize()

    # calculate GPP with acclimation
    datetimes = pd.date_range(
        start=datetimestart, periods=timesteps, freq="h"
    ).to_numpy()

    fsscaler = SubdailyScaler(datetimes)
    fsscaler.set_window(
        window_center=np.timedelta64(window_cent, "h"),
        half_width=np.timedelta64(half_wdth, "m"),
    )

    subdailyC3_arr = SubdailyPModel(
        env=env_arr,
        fapar=fpar_modis_arr,
        ppfd=ppfd_arr,
        fs_scaler=fsscaler,
        alpha=1 / days_mem,
        allow_holdover=True,
    )
    subdailyC3_arr.gpp = subdailyC3_arr.gpp * gC_to_mumol

    t = 0
    for file in files[1:]:
        subdailyC3_gpp = subdailyC3_arr.gpp[t, :, :]
        # print(t, " ", np.nanmax(subdailyC3_gpp))
        # save data in netcdf
        date_time = file.split("_")[2] + "_" + file.split("_")[3]
        gpp_path_out = f"{modis_path}gpp_pmodel/gpp_pmodel_subdailyC3_{wrf_path_dx_str}_{date_time}.nc"
        xr.DataArray(subdailyC3_gpp, name="GPP_Pmodel").to_netcdf(
            gpp_path_out, format="NETCDF4_CLASSIC"
        )
        t += 1

    print(f"Saved GPP data to {gpp_path_out}")

    # Ensure rainc_arr is a NumPy array (3D: time, lat, lon)
    rainc_arr = np.asarray(rainc_arr)
    subdailyC3_arr = np.asarray(subdailyC3_arr.gpp)  # Extract GPP data if needed

    if rainc_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array (time, lat, lon), but got shape {rainc_arr.shape}"
        )

    if subdailyC3_arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array (time, lat, lon), but got shape {subdailyC3_arr.shape}"
        )

    window_size = 7  # Adjust to match time step frequency

    # Apply rolling mean
    rainc_avrg = uniform_filter1d(rainc_arr, size=window_size, axis=0, mode="nearest")
    subdailyC3_avrg = uniform_filter1d(
        subdailyC3_arr, size=window_size, axis=0, mode="nearest"
    )

    # Compute initial mean for first `window_size` time steps
    initial_mean_rainc = np.mean(rainc_arr[:window_size, :, :], axis=0)
    initial_mean_subdailyC3 = np.mean(subdailyC3_arr[:window_size, :, :], axis=0)

    # Fill missing values
    rainc_avrg = np.where(np.isnan(rainc_avrg), initial_mean_rainc, rainc_avrg)
    subdailyC3_avrg = np.where(
        np.isnan(subdailyC3_avrg), initial_mean_subdailyC3, subdailyC3_avrg
    )

    # Ensure no NaNs
    rainc_avrg = np.nan_to_num(rainc_avrg, nan=0)
    subdailyC3_avrg = np.nan_to_num(subdailyC3_avrg, nan=0)

    # settings
    T_ref = 288.15
    T0 = 227.13
    max_lai = np.max(lai_wrf_arr, axis=0)  # Get max LAI for each (lat, lon) location
    # TODO: create max_lai Map from MODIS LAI

    # Convert IVGTYP_vprm to integer type to use it as an index
    IVGTYP_vprm = IVGTYP_vprm.astype(int)

    # Create empty arrays with the same shape as IVGTYP_vprm
    shape = IVGTYP_vprm.shape
    E0_arr = np.zeros(shape)
    K_arr = np.zeros(shape)
    k2_arr = np.zeros(shape)
    alpha_p_arr = np.zeros(shape)
    alphaLAI_arr = np.zeros(shape)
    RLAI_arr = np.zeros(shape)

    # Fill arrays using vegetation type mapping
    for veg_type in range(1, 9):  # Loop through vegetation types 1 to 8
        mask = IVGTYP_vprm == veg_type  # Find locations for current vegetation type
        params = fixed_boundaries[str(veg_type)]  # Retrieve parameters

        E0_arr[mask] = params["E0(K)"]
        K_arr[mask] = params["K (mm)"]
        k2_arr[mask] = params["k2"]
        alpha_p_arr[mask] = params["alpha_p"]
        alphaLAI_arr[mask] = params["alphaLAI"]
        RLAI_arr[mask] = params["RLAI"]

    # Now you can use these arrays in your function call
    Reco_optimized = migliavacca_LinGPP(
        T_ref,
        T0,
        E0_arr,
        K_arr,
        k2_arr,
        alpha_p_arr,
        alphaLAI_arr,
        max_lai,
        RLAI_arr,
        subdailyC3_avrg,
        rainc_avrg,
        tc_arr + 273.15,
    )

    t = 0
    for file in files[1:]:
        subdailyC3_gpp = Reco_optimized[t, :, :]
        date_time = file.split("_")[2] + "_" + file.split("_")[3]
        reco_path_out = (
            f"{migli_path}/reco_migliavacca_subdailyC3_{wrf_path_dx_str}_{date_time}.nc"
        )
        xr.DataArray(subdailyC3_gpp, name="RECO_Migli").to_netcdf(
            reco_path_out, format="NETCDF4_CLASSIC"
        )
        t += 1
    print("Reco_optimized written to : ", reco_path_out)
