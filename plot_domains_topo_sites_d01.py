# wrf_overlay_domains.py (xarray-only version)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines

outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
# Load datasets
d1 = xr.open_dataset("/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km/wrfinput_d01")
d2 = xr.open_dataset("/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d02")

# Extract variables
var1 = d1["HGT"].isel(Time=0)
lat1 = d1["XLAT"].isel(Time=0)
lon1 = d1["XLONG"].isel(Time=0)

# Plot setup
# fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 6))
# ax.coastlines()
fig = plt.figure(figsize=(12, 15))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent(
    [float(lon1.min()), float(lon1.max()), float(lat1.min()), float(lat1.max())],
    crs=ccrs.PlateCarree(),
)

# Add map features
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, edgecolor="black")
ax.add_feature(cfeature.LAKES, edgecolor="black")
ax.add_feature(cfeature.RIVERS)

# Add gridlines
gl = ax.gridlines(
    draw_labels=True, linewidth=1.5, color="black", alpha=0.5, linestyle="--"
)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 20}
gl.ylabel_style = {"size": 20}

# Contour settings
levels = list(range(0, 3000, 100)) + [4000]
cf1 = ax.contourf(lon1, lat1, var1, levels=levels, cmap="terrain", extend="both")

# Add colorbar
# cb = plt.colorbar(cf1, ax=ax, orientation="vertical", pad=0.01, shrink=0.4)
# cb.set_label("[m]", fontsize=20)
# cb.ax.tick_params(labelsize=20)

# Domain labels
ax.text(
    float(lon1[0, 0]) + 0.4, float(lat1[0, 0]) + 0.2, "d01", fontsize=20, weight="bold"
)

# Site info (filtered)
site_names = [
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
    "IT-Lav",
    "IT-MBo",
    "IT-PT1",
    "IT-Ren",
    "IT-Tor",
]
site_lat = [
    47.1167,
    47.2102,
    46.8153,
    47.1158,
    47.4781,
    47.2858,
    47.2863,
    49.0996,
    45.8126,
    45.9542,
    45.9562,
    46.0147,
    45.2009,
    46.5869,
    45.8444,
]
site_lon = [
    11.3175,
    8.4104,
    9.8559,
    8.5378,
    8.3650,
    7.7319,
    7.7343,
    13.3047,
    8.6336,
    11.2853,
    11.2813,
    11.0458,
    9.0610,
    11.4337,
    7.5781,
]


site_types = [
    "GRA",
    "GRA",
    "ENF",
    "GRA",
    "MF",
    "GRA",
    "CRO",
    "ENF",
    "DBF",
    "ENF",
    "ENF",
    "GRA",
    "DBF",
    "ENF",
    "GRA",
]

colors = [
    "#006400",  # Evergreen
    "#228B22",  # Deciduous
    "#8FBC8F",  # Mixed Forest
    "#A0522D",  # Shrubland
    "#FFD700",  # Savannas
    "#FFA07A",  # Cropland
    "#7CFC00",  # Grassland
]
pft_labels = [
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Shrubland",
    "Savannas",
    "Cropland",
    "Grassland",
]
pft_codes = ["ENF", "DBF", "MF", "SHB", "SAV", "CRO", "GRA"]

# Map PFT codes to colors
pft_color_map = dict(zip(pft_codes, colors))

# Plot sites with PFT colors
for lon, lat, typ in zip(site_lon, site_lat, site_types):
    ax.plot(
        lon,
        lat,
        "o",
        color=pft_color_map.get(typ, "#808080"),
        markersize=12,
        transform=ccrs.PlateCarree(),
        markeredgecolor="black",
        markeredgewidth=1.5,
    )

# Build legend handles
handles = [
    mlines.Line2D(
        [], [], color=col, marker="o", linestyle="None", markersize=10, label=lbl
    )
    for col, lbl in zip(colors, pft_labels)
]

ax.legend(handles=handles, loc="upper left", fontsize=12, frameon=True)

plt.tight_layout()
plt.savefig(outfolder + "domains_topo_sites_d01.pdf", bbox_inches="tight")


# site_types = [
#     "GRA",
#     "GRA",
#     "ENF",
#     "GRA",
#     "MF",
#     "GRA",
#     "CRO",
#     "ENF",
#     "DBF",
#     "ENF",
#     "ENF",
#     "GRA",
#     "DBF",
#     "ENF",
#     "GRA",
# ]

# # Plot site markers
# for name, lat, lon, typ in zip(site_names, site_lat, site_lon, site_types):
#     ax.plot(lon, lat, "o", color="black", markersize=3, transform=ccrs.PlateCarree())
#     label = f"{name}-{typ}"
#     dx, dy = 0.025, 0.025
#     ax.text(lon + dx, lat + dy, label, fontsize=16, transform=ccrs.PlateCarree())

# # plt.title("Overlay Domains with Sites", fontsize=16)
# plt.tight_layout()
# plt.show()
# plt.savefig(f"{outfolder}/domains_topo_sites.pdf", bbox_inches="tight")


#     # dx, dy = 0.025, 0.025
#     # if name == "IT-Lav":
#     #     dx, dy = 0.1, -0.15
#     # elif name == "IT-La2":
#     #     dx, dy = 0.1, -0.4
#     # elif name == "CH-Oe1":
#     #     dx, dy = -1.5, 0.1
#     # elif name == "CH-Oe2":
#     #     dx, dy = -1.5, -0.3
#     # elif name == "CH-Fru":
#     #     dx, dy = 0, -0.2
#     # elif name == "CH-Dav":
#     #     dx, dy = -1.5, -0.35
#     # elif name == "IT-PT1":
#     #     dx, dy = 0, -0.35
#     # elif name == "IT-Tor":
#     #     dx, dy = -1.5, -0.35
