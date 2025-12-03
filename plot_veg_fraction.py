import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Input ---
dx = "_54km"
domain = "_d01"
base_mz = f"/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out{domain}_2012{dx}"
t_file_fra = "VPRM_input_VEG_FRA_d01_2012.nc"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

# --- Load vegetation fraction map ---
ds = xr.open_dataset(os.path.join(base_mz, t_file_fra))
veg_frac_map = ds["vegetation_fraction_map"]
lat = ds["lat"].values
lon = ds["lon"].values

# --- Load d02 domain and restrict data to that extent ---
d2 = xr.open_dataset("/scratch/c7071034/WPS/geo_em.d02.nc")
lat2 = (
    d2["XLAT_M"]
    .isel(Time=0, south_north=slice(10, -10), west_east=slice(10, -10))
    .values
)
lon2 = (
    d2["XLONG_M"]
    .isel(Time=0, south_north=slice(10, -10), west_east=slice(10, -10))
    .values
)

lat2_min, lat2_max = lat2.min(), lat2.max()
lon2_min, lon2_max = lon2.min(), lon2.max()

mask = (lat >= lat2_min) & (lat <= lat2_max) & (lon >= lon2_min) & (lon <= lon2_max)

# Get indices where mask is True
y_idx, x_idx = np.where(mask)
if len(y_idx) == 0 or len(x_idx) == 0:
    raise ValueError("No overlapping data between domain and vegetation fraction map")

# Slice bounds
ymin, ymax = y_idx.min(), y_idx.max() + 1
xmin, xmax = x_idx.min() - 1, x_idx.max() + 1

# Crop datasets
veg_frac_map = veg_frac_map.isel(
    south_north=slice(ymin, ymax), west_east=slice(xmin, xmax)
)
lat = lat[ymin:ymax, xmin:xmax]
lon = lon[ymin:ymax, xmin:xmax]

# --- PFT settings ---
colors = [
    "#006400",
    "#228B22",
    "#8FBC8F",
    "#A0522D",
    "#FFD700",
    "#FFA07A",
    "#7CFC00",
    "#808080",
]
pft_labels = [
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Shrubland",
    "Savannas",
    "Cropland",
    "Grassland",
    "Others",
]
cmap = mcolors.ListedColormap(colors)
bounds = np.arange(1, 10)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Compute dominant PFT (1–8) ---
dominant_type = veg_frac_map.argmax(dim="vprm_classes") + 1

# --- Plot base map with cartopy ---
fig = plt.figure(figsize=(12, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(
    [float(lon2.min()), float(lon2.max()), float(lat2.min()), float(lat2.max())],
    crs=ccrs.PlateCarree(),
)

# Background: dominant vegetation type
im = ax.pcolormesh(
    lon,
    lat,
    dominant_type,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
)

# Add map features
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, edgecolor="black")
ax.add_feature(cfeature.LAKES, edgecolor="black")
ax.add_feature(cfeature.RIVERS)

# Add gridlines
gl = ax.gridlines(
    draw_labels=True, linewidth=1.5, color="black", alpha=0.3, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 18}
gl.ylabel_style = {"size": 18}

# --- Overlay: fractional PFT pies via inset_axes ---
step = 1
pie_size = 1.2

for i in range(1, lat.shape[0] - 1, step):
    for j in range(1, lat.shape[1] - 1, step):
        f = veg_frac_map[:, i, j].values
        if np.isnan(f).all() or np.sum(f) == 0:
            continue
        f = f / np.sum(f)
        x, y = lon[i, j], lat[i, j]

        ax_inset = inset_axes(
            ax,
            width=pie_size,
            height=pie_size,
            loc="center",
            bbox_to_anchor=(x, y),
            bbox_transform=ax.transData,
            borderpad=0,
        )
        ax_inset.pie(f, colors=colors, wedgeprops=dict(edgecolor="k", linewidth=0.2))
        ax_inset.set_aspect("equal")
        ax_inset.axis("off")

# --- Labels, legend, colorbar ---
# ax.set_title(
#    "Fractional PFT composition with dominant PFT background (cut to d02)", fontsize=16
# )

# patches = [Patch(color=colors[k], label=lab) for k, lab in enumerate(pft_labels)]
# ax.legend(handles=patches, loc="upper right", fontsize=16)

# cbar = plt.colorbar(
#     im, ax=ax, ticks=np.arange(1.5, 9.5), fraction=0.046, pad=0.04, shrink=0.3
# )
# cbar.ax.set_yticklabels(pft_labels)
# cbar.ax.tick_params(labelsize=14)

plt.tight_layout()
# plt.show()
plt.savefig(f"{outfolder}/domain_d02_PFTs_pie_per_cell.pdf", bbox_inches="tight")
