import warnings

warnings.filterwarnings("ignore")
from pyVPRM.lib.fancy_plot import *
from rioxarray import merge
import numpy as np
import os
import glob
import xarray as xr
import shutil
import cartopy.crs as ccrs
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import netCDF4 as nc
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


# Evergreen Forest:
#     vprm_class: 1
#     class_numbers:
#         - 24  # Coniferous Forest

# Deciduous Forest:
#     vprm_class: 2
#     class_numbers:
#         - 23  # Broad-leaved Forest

# Mixed Forest:
#     vprm_class: 3
#     class_numbers:
#         - 25  # Mixed Forest
#         - 29  # Transitional Woodland-Shrub

# Shrubland:
#     vprm_class: 4
#     class_numbers:
#         - 27  # Moors and Heathland
#         - 28  # Sclerophyllous Vegetation

# Wetlands:
#     vprm_class: 5
#     class_numbers:
#         - 35  # Inland Marshes
#         - 36  # Peat Bogs
#         - 37  # Salt marshes

# Cropland:
#     vprm_class: 6
#     class_numbers:
#         - 12  # Non-irrigated Arable Land
#         - 13  # Permanently Irrigated Land
#         - 15  # Vineyards
#         - 16  # Fruit Trees and Berry Plantations
#         - 17  # Olive Groves
#         - 19  # Annual Crops Associated with Permanent Crops
#         - 20  # Complex Cultivation Patterns
#         - 21  # Land Principally Occupied by Agriculture
#         - 22  # Agro-forestry areas
#         - 14  # Rice Fields

# Grassland:
#     vprm_class: 7
#     class_numbers:
#       - 18  # Pastures
#       - 26  # Natural Grasslands

# Other:
#     vprm_class: 8
#     class_numbers:
#       - 1   # Continuous Urban Fabric
#       - 2   # Discontinuous Urban Fabric
#       - 3   # Industrial or Commercial Units
#       - 4   # Road and Rail Networks
#       - 5   # Port Areas
#       - 6   # Airports
#       - 7   # Mineral Extraction Sites
#       - 8   # Dump Sites
#       - 9   # Construction Sites
#       - 10  # Green Urban Areas
#       - 11  # Sport and Leisure Facilities
#       - 30  # Beaches, Dunes, Sands
#       - 31  # Bare Rocks
#       - 32  # Sparsely Vegetated Areas
#       - 33  # Burnt Areas
#       - 34  # Glaciers and Perpetual Snow
#       - 38  # Salines
#       - 39  # Intertidal flats
#       - 40  # Water courses
#       - 41  # Water bodies
#       - 42  # Coastal lagoons
#       - 43  # Estuaries
#       - 44  # Estuaries

import numpy as np


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

    aspect_mask = aspect_diff <= 45
    slope_mask = slope_diff <= 20

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


def find_nearest_grid_hgt(
    lat_target, lon_target, lats, lons, location_pft, IVGTYP_vprm, hgt, hgt_site, radius
):
    """Find the nearest grid index for a given lat/lon with the same PFT and lowest height difference."""

    # Get valid lat/lon values
    valid_mask = IVGTYP_vprm == location_pft
    valid_lats = np.where(valid_mask, lats, np.nan)
    valid_lons = np.where(valid_mask, lons, np.nan)

    # Convert latitude and longitude differences to km
    lat_diff = (
        valid_lats - lat_target
    )  # * 111  # approximate conversion factor for degrees to km
    lon_diff = (
        valid_lons - lon_target  # * 111 * np.cos(np.radians(lat_target))
    )  # adjust for latitude
    dist = np.sqrt(lat_diff**2 + lon_diff**2)
    dist_km = dist * 111  # approximate conversion factor for degrees to km

    # Mask the distance to only consider points within the radius
    within_radius_mask = dist_km <= abs(radius)
    dist_within_radius = np.where(within_radius_mask, dist_km, np.nan)

    # Check if there are valid points within the radius
    if np.all(np.isnan(dist_within_radius)):
        min_index = np.unravel_index(np.nanargmin(dist), lats.shape)
        return (
            np.nanmin(dist_km),
            min_index,
        )  # No valid points found within the radius, closest index is given back without checking height

    # Calculate the height difference for points within the radius
    height_diff_within_radius = np.where(
        within_radius_mask, np.abs(hgt - hgt_site), np.nan
    )

    # Get the index of the minimum height difference within the radius
    min_height_diff_idx = np.unravel_index(
        np.nanargmin(height_diff_within_radius), lats.shape
    )

    # Get the value of the distance with the minimum height
    dist_idx = dist_within_radius[min_height_diff_idx[0], min_height_diff_idx[1]]

    # Return the minimum distance and the index of the minimum height difference
    return dist_idx, min_height_diff_idx


def get_int_var(lat_target, lon_target, lats, lons, WRF_var):
    interpolator = RegularGridInterpolator((lats[:, 0], lons[0, :]), WRF_var)
    interpolated_value = interpolator((lat_target, lon_target))
    return interpolated_value


############# INPUT ############
domain = "_d02"
dx = "_1km"
radius = 20  # should be set to dx*20 to fit to extract_timeseries.py
base_mz = (
    "/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out"
    + domain
    + "_2012"
    + dx
)

if dx == "_1km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d02"
elif dx == "_3km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d01"
elif dx == "_9km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km/wrfinput_d01"
elif dx == "_27km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km/wrfinput_d01"
elif dx == "_54km":
    wrfinput_path = "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km/wrfinput_d01"

t_file_fra = "VPRM_input_VEG_FRA" + domain + "_2012.nc"
# t_file_fra = 'VPRM_input_VEG_FRA_2022.nc'
locations_hgt = {
    "CH-Dav": 1639,
    # "CH-Oe2": 452,
    # "DE-Lkb": 1308,
    "IT-Lav": 1353,
    "IT-Ren": 1730,
    "AT-Neu": 970,
    "IT-MBo": 1550,
    # "IT-Tor": 2160,
    # "CH-Lae": 689,
}

sites = [
    # {"site": "CH-Oe2", "country": "Switzerland", "pft": "CRO", "pft_id": "6", "lat": 47.2863, "lon": 7.7343,"hgt_site": 452},
    # {"site": "IT-Isp", "country": "Italy", "pft": "DBF", "pft_id": "2", "lat": 45.8126, "lon": 8.6336},
    # {"site": "IT-PT1", "country": "Italy", "pft": "DBF", "pft_id": "2", "lat": 45.2009, "lon": 9.061},
    {
        "site": "CH-Dav",
        "country": "Switzerland",
        "pft": "ENF",
        "pft_id": "1",
        "lat": 46.8153,
        "lon": 9.8559,
        "hgt_site": 1639,
    },
    # {"site": "DE-Lkb", "country": "Germany", "pft": "ENF", "pft_id": "1", "lat": 49.0996, "lon": 13.3047,"hgt_site": 1308},
    # {"site": "IT-La2", "country": "Italy", "pft": "ENF", "pft_id": "1", "lat": 45.9542, "lon": 11.2853},
    {
        "site": "IT-Lav",
        "country": "Italy",
        "pft": "ENF",
        "pft_id": "1",
        "lat": 45.9562,
        "lon": 11.2813,
        "hgt_site": 1353,
    },
    {
        "site": "IT-Ren",
        "country": "Italy",
        "pft": "ENF",
        "pft_id": "1",
        "lat": 46.5869,
        "lon": 11.4337,
        "hgt_site": 1730,
    },
    {
        "site": "AT-Neu",
        "country": "Austria",
        "pft": "GRA",
        "pft_id": "7",
        "lat": 47.1167,
        "lon": 11.3175,
        "hgt_site": 970,
    },
    # {"site": "CH-Cha", "country": "Switzerland", "pft": "GRA", "pft_id": "7", "lat": 47.2102, "lon": 8.4104},
    # {"site": "CH-Fru", "country": "Switzerland", "pft": "GRA", "pft_id": "7", "lat": 47.1158, "lon": 8.5378},
    # {"site": "CH-Oe1", "country": "Switzerland", "pft": "GRA", "pft_id": "7", "lat": 47.2858, "lon": 7.7319},
    {
        "site": "IT-MBo",
        "country": "Italy",
        "pft": "GRA",
        "pft_id": "7",
        "lat": 46.0147,
        "lon": 11.0458,
        "hgt_site": 1550,
    },
    # {"site": "IT-Tor", "country": "Italy", "pft": "GRA", "pft_id": "7", "lat": 45.8444, "lon": 7.5781,"hgt_site": 2160},
    # {"site": "CH-Lae", "country": "Switzerland", "pft": "MF", "pft_id": "3", "lat": 47.4781, "lon": 8.3644,"hgt_site": 689},
    # {"site": "CZ-wet", "country": "Czech Republic", "pft": "WET", "pft_id": "8", "lat": 49.0247, "lon": 14.7704},
    # {"site": "DE-SfN", "country": "Germany", "pft": "WET", "pft_id": "8", "lat": 47.8064, "lon": 11.3275},
]

# CORINE vegetation type labels
vegetation_labels = [
    "Continuous Urban Fabric",
    "Discontinuous Urban Fabric",
    "Industrial or Commercial Units",
    "Road and Rail Networks",
    "Port Areas",
    "Airports",
    "Mineral Extraction Sites",
    "Dump Sites",
    "Construction Sites",
    "Green Urban Areas",
    "Sport and Leisure Facilities",
    "Non-irrigated Arable Land",
    "Permanently Irrigated Land",
    "Rice Fields",
    "Vineyards",
    "Fruit Trees and Berry Plantations",
    "Olive Groves",
    "Pastures",
    "Annual Crops Associated with Permanent Crops",
    "Complex Cultivation Patterns",
    "Land Principally Occupied by Agriculture",
    "Agro-forestry Areas",
    "Broad-leaved Forest",
    "Coniferous Forest",
    "Mixed Forest",
    "Natural Grasslands",
    "Moors and Heathland",
    "Sclerophyllous Vegetation",
    "Transitional Woodland-Shrub",
    "Beaches, Dunes, Sands",
    "Bare Rocks",
    "Sparsely Vegetated Areas",
    "Burnt Areas",
    "Glaciers and Perpetual Snow",
    "Inland Marshes",
    "Peat Bogs",
    "Salt Marshes",
    "Salines",
    "Intertidal Flats",
    "Water Courses",
    "Water Bodies",
    "Coastal Lagoons",
    "Estuaries",
]

# Define a color map for CORINE vegetation types
vegetation_colors = [
    "#FF0000",
    "#FF4500",
    "#B22222",
    "#808080",
    "#4682B4",
    "#87CEFA",
    "#8B4513",
    "#A0522D",
    "#D2691E",
    "#32CD32",
    "#7FFF00",
    "#FFFF00",
    "#FFD700",
    "#F0E68C",
    "#8B0000",
    "#B8860B",
    "#CD853F",
    "#ADFF2F",
    "#DDA0DD",
    "#F5DEB3",
    "#FFE4C4",
    "#DEB887",
    "#228B22",
    "#006400",
    "#8FBC8F",
    "#7CFC00",
    "#556B2F",
    "#6B8E23",
    "#8B7500",
    "#F4A460",
    "#A9A9A9",
    "#D3D3D3",
    "#FFFFFF",
    "#2E8B57",
    "#3CB371",
    "#20B2AA",
    "#5F9EA0",
    "#48D1CC",
    "#00CED1",
    "#1E90FF",
    "#0000CD",
    "#4682B4",
    "#708090",
]

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

# Define labels for the simplified vegetation types
labels_vprm_short = [
    "ENF",
    "DBF",
    "MF",
    "SHB",
    "SAV",
    "CRO",
    "GRA",
    "OTH",  # Include "Others"
]
# Load the NetCDF file
nc_fid = nc.Dataset(wrfinput_path, "r")
IVGTYP = nc_fid.variables["IVGTYP"][0, 10:-10, 10:-10]
IVGTYP.shape
# Retrieve the CORINE vegetation types (IVGTYP) and coordinates (XLAT, XLONG)
XLAT = nc_fid.variables["XLAT"][0, 10:-10, 10:-10]  # Assuming the first time slice
XLONG = nc_fid.variables["XLONG"][0, 10:-10, 10:-10]
HGT_M = nc_fid.variables["HGT"][0, 10:-10, 10:-10]
IVGTYP_vprm = np.vectorize(corine_to_vprm.get)(IVGTYP[:, :])
IVGTYP_vprm.shape

in_veg_frac = xr.open_dataset(os.path.join(base_mz, t_file_fra))
veg_frac_map = in_veg_frac["vegetation_fraction_map"][:, 10:-10, 10:-10]

veg_frac_class_8 = veg_frac_map.sel(vprm_classes=8)
mask_ivgtyp_trimmed = IVGTYP == 44
veg_frac_class_8 = veg_frac_class_8.where(~mask_ivgtyp_trimmed, 1)

veg_frac_map.loc[{"vprm_classes": 8}] = veg_frac_class_8

# Proceed from trimmed veg_frac_map only
VPRM_in_dom_vgtyp = veg_frac_map.argmax(dim="vprm_classes") + 1
df_VPRM_in_dom_vgtyp = VPRM_in_dom_vgtyp.to_dataframe(
    name="vegetation_type"
).reset_index()


# VEGTYPES:
# 1) evergreen, 2) deciduous, 3) mixed forest,4) shrubland, 5) savannas, 6) cropland, 7) grassland, and 8) others

# # Define the bins and normalize
# cmap = plt.cm.inferno_r
# cmaplist = [cmap(i) for i in range(cmap.N)]
# cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
# bounds = np.linspace(0, 1, 11)
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# # Loop through the vegetation classes
# for i in range(1, 9):
#     # Create a new figure for each vegetation class
#     fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

#     # Plot the vegetation fraction map for the current class
#     cb = in_veg_frac.sel({'vprm_classes': i})['vegetation_fraction_map'].plot.pcolormesh(
#         cmap=cmap, ax=ax, x='lon', y='lat', vmin=0, vmax=1.0, add_colorbar=False
#     )

#     ax.set_title(f'Vegetation Class {i}')

#     # Set limits
#     lats = in_veg_frac['lat'].values.flatten()
#     lons = in_veg_frac['lon'].values.flatten()
#     ax.set_xlim(np.min(lons), np.max(lons))
#     ax.set_ylim(np.min(lats), np.max(lats))

#     # Add gridlines
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                       linewidth=1.5, color='gray', alpha=0.5, linestyle='--')
#     gl.xlabels_top = False
#     gl.ylabels_right = False

#     # Add colorbar
#     axins = inset_axes(ax, width="5%", height="90%", loc='center right', borderpad=-5)
#     cbar = fig.colorbar(cb, cax=axins)
#     cbar.ax.get_yaxis().labelpad = 15
#     cbar.ax.set_ylabel('Vegetation Fraction', rotation=270)
#     ax.plot(lon_target, lat_target, marker='x', color='blue', markersize=20, transform=ccrs.PlateCarree(), label='FluxNet Tower')
#     ax.legend()
#     # Show the figure
#     plt.tight_layout()
#     plt.savefig(f"VGT_frac_VPRM_compare_{i}_{domain+dx}.png")
#     plt.show()


# find pft match in 10 dx radius with same height
# Append model PFT to each site
for site in sites:
    lat, lon = site["lat"], site["lon"]
    dist_km, grid_idx, target_slope,target_aspect = find_nearest_grid_hgt_sa(
        lat,
        lon,
        XLAT,
        XLONG,
        int(site["pft_id"]),
        VPRM_in_dom_vgtyp,
        HGT_M,
        site["hgt_site"],
        radius,
    )
    site["model_pft"] = IVGTYP[grid_idx[0], grid_idx[1]]
    site["model_hgt_NN"] = HGT_M[grid_idx[0], grid_idx[1]]
    site["model_lat"] = XLAT[grid_idx[0], grid_idx[1]]
    site["model_lon"] = XLONG[grid_idx[0], grid_idx[1]]
    site["dist_km"] = dist_km
    site["target_slope"] = target_slope
    site["target_aspect"] = target_aspect
    site["model_pft_CLC"] = vegetation_labels[site["model_pft"] - 1]
    site["model_pft_CLC_to_VPRM"] = labels_vprm_short[
        corine_to_vprm[site["model_pft"]] - 1
    ]
    # site["model_pft_VPRM_dom_id"] = int(VPRM_in_dom_vgtyp[grid_idx[0],grid_idx[1]])
    if site["pft"] == site["model_pft_CLC_to_VPRM"]:
        site["pft_match"] = True
    else:
        site["pft_match"] = False

# Convert to DataFrame for tabular output
df_sites_match = pd.DataFrame(sites)

# Print the table
print(df_sites_match)
df_sites_match.to_csv("pft_site_match_at" + dx + ".csv")

# Create color map and boundaries for  CORINE types
cmap_vegetation = mcolors.ListedColormap(vegetation_colors)
bounds_vegetation = list(
    range(1, 45)
)  # Vegetation types (1–44), extra bin to close the range
norm_vegetation = mcolors.BoundaryNorm(bounds_vegetation, cmap_vegetation.N)

# Plot the vegetation map using the CORINE  classification
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})

# Plot using pcolormesh with XLAT and XLONG as coordinates and shading='auto'
vegetation_plot = ax.pcolormesh(
    XLONG,
    XLAT,
    IVGTYP[:, :],  # Use first time slice of IVGTYP for 2D data
    cmap=cmap_vegetation,
    norm=norm_vegetation,
    transform=ccrs.PlateCarree(),
    shading="auto",
)

# Add colorbar with vegetation labels
cbar = fig.colorbar(
    vegetation_plot, ax=ax, ticks=np.arange(1, 44) + 0.5, orientation="vertical"
)
cbar.ax.set_yticklabels(vegetation_labels)
cbar.ax.set_ylabel("CORINE Noah Vegetation Type (IVGTYP)")
legend_entries = []

# Plot markers for each site
for index, site in df_sites_match.iterrows():
    label = f"{site['site']}-{site['pft']}"  # Create label as 'site-pft'

    # Adjust specific label positions if necessary
    if site["site"] == "IT-Lav":
        text_offset_x, text_offset_y = 0.05, -0.1
    else:
        text_offset_x, text_offset_y = 0.05, 0.1

    # Plot the marker
    ax.plot(
        site["lon"],
        site["lat"],
        marker="o",
        color="black",
        markersize=10,
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        site["model_lon"],
        site["model_lat"],
        marker="o",
        color="blue",
        markersize=10,
        linewidth=2,
        transform=ccrs.PlateCarree(),
    )
    # add blue dashed lines to connect the site and model point
    ax.plot(
        [site["lon"], site["model_lon"]],
        [site["lat"], site["model_lat"]],
        color="blue",
        linestyle="--",
        linewidth=3,
        transform=ccrs.PlateCarree(),
    )

    # Add the label
    ax.text(
        site["lon"] + text_offset_x,
        site["lat"] + text_offset_y,
        label,
        color="black",
        transform=ccrs.PlateCarree(),
        fontsize=20,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.3, boxstyle="round, pad=0.3"),
        ha="left",
        va="center",
    )

# Set title and gridlines
# ax.set_title("Vegetation Types from WRF CORINE")
gl = ax.gridlines(
    draw_labels=True, linewidth=1.5, color="gray", alpha=0.5, linestyle="--"
)
gl.xlabels_top = False
gl.ylabels_right = False
plt.tight_layout()
plt.show()
plt.savefig(f"VGT_CORINE{domain+dx}.png")
