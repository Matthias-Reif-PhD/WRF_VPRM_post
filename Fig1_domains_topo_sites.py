"""
Plot WRF domains with topography and FLUXNET site locations.
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.lines as mlines

# ==================== Configuration ====================

OUTFOLDER = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

# Site metadata
SITES = {
    "names": [
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
    ],
    "lats": [
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
    ],
    "lons": [
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
    ],
    "types": [
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
    ],
}

# PFT configuration
PFT_CODES = ["ENF", "DBF", "MF", "SHB", "SAV", "CRO", "GRA"]
PFT_LABELS = [
    "Evergreen forest",
    "Deciduous forest",
    "Mixed forest",
    "Shrubland",
    "Savannas",
    "Cropland",
    "Grassland",
]
PFT_COLORS = [
    "#006400",
    "#228B22",
    "#8FBC8F",
    "#A0522D",
    "#FFD700",
    "#FFA07A",
    "#7CFC00",
]

PFT_COLOR_MAP = dict(zip(PFT_CODES, PFT_COLORS))

# ==================== Functions ====================


def load_wrf_data(filepath):
    """Load WRF dataset and extract coordinates and topography."""
    ds = xr.open_dataset(filepath)
    hgt = ds["HGT"].isel(Time=0)
    lat = ds["XLAT"].isel(Time=0)
    lon = ds["XLONG"].isel(Time=0)
    return hgt, lat, lon


def setup_map(ax, lon_min, lon_max, lat_min, lat_max):
    """Configure map axes with features and gridlines."""
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

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


def plot_topography(ax, lon, lat, hgt, add_colorbar=False):
    """Plot topographic contours."""
    levels = list(range(0, 3000, 100)) + [4000]
    cf = ax.contourf(lon, lat, hgt, levels=levels, cmap="terrain", extend="both")

    if add_colorbar:
        cb = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.01, shrink=0.4)
        cb.set_label("[m]", fontsize=20)
        cb.ax.tick_params(labelsize=20)

    return cf


def plot_sites(ax, site_lons, site_lats, site_types, add_legend=True):
    """Plot FLUXNET sites with PFT colors."""
    for lon, lat, pft_type in zip(site_lons, site_lats, site_types):
        ax.plot(
            lon,
            lat,
            "o",
            color=PFT_COLOR_MAP.get(pft_type, "#808080"),
            markersize=12,
            transform=ccrs.PlateCarree(),
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

    if add_legend:
        handles = [
            mlines.Line2D(
                [],
                [],
                color=col,
                marker="o",
                linestyle="None",
                markersize=10,
                label=lbl,
            )
            for col, lbl in zip(PFT_COLORS, PFT_LABELS)
        ]
        ax.legend(handles=handles, loc="upper left", fontsize=12, frameon=True)


def plot_nested_domain_boundary(ax, lon2, lat2):
    """Plot the boundary of a nested domain."""
    ny2, nx2 = lon2.shape
    xbox = [
        float(lon2[0, 0]),
        float(lon2[0, nx2 - 1]),
        float(lon2[ny2 - 1, nx2 - 1]),
        float(lon2[ny2 - 1, 0]),
        float(lon2[0, 0]),
    ]
    ybox = [
        float(lat2[0, 0]),
        float(lat2[0, nx2 - 1]),
        float(lat2[ny2 - 1, nx2 - 1]),
        float(lat2[ny2 - 1, 0]),
        float(lat2[0, 0]),
    ]
    ax.plot(xbox, ybox, color="black", linewidth=1.5, transform=ccrs.PlateCarree())


def add_domain_labels(ax, lon, lat, domain_labels):
    """Add text labels for domains."""
    for label, lo, la in domain_labels:
        ax.text(float(lo) + 0.4, float(la) + 0.2, label, fontsize=20, weight="bold")


def plot_single_domain(hgt1, lat1, lon1, domain_label, filename):
    """Plot a single domain with topography and sites."""
    fig = plt.figure(figsize=(12, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())

    setup_map(
        ax, float(lon1.min()), float(lon1.max()), float(lat1.min()), float(lat1.max())
    )
    plot_topography(ax, lon1, lat1, hgt1, add_colorbar=False)
    plot_sites(ax, SITES["lons"], SITES["lats"], SITES["types"], add_legend=True)
    add_domain_labels(ax, lon1, lat1, [(domain_label, lon1[0, 0], lat1[0, 0])])

    plt.tight_layout()
    plt.savefig(OUTFOLDER + filename, bbox_inches="tight")
    plt.close()


def plot_nested_domains(hgt1, lat1, lon1, hgt2, lat2, lon2, domain_labels, filename):
    """Plot nested domains with topography and sites."""
    fig = plt.figure(figsize=(12, 15))
    ax = plt.axes(projection=ccrs.PlateCarree())

    setup_map(
        ax, float(lon1.min()), float(lon1.max()), float(lat1.min()), float(lat1.max())
    )
    plot_topography(ax, lon1, lat1, hgt1, add_colorbar=True)
    plot_topography(ax, lon2, lat2, hgt2, add_colorbar=False)
    plot_nested_domain_boundary(ax, lon2, lat2)
    plot_sites(ax, SITES["lons"], SITES["lats"], SITES["types"], add_legend=False)
    add_domain_labels(ax, lon1, lat1, domain_labels)

    plt.tight_layout()
    plt.savefig(OUTFOLDER + filename, bbox_inches="tight")
    plt.close()


# ==================== Main ====================


def main():
    """Generate domain plots."""
    # Plot 1: Single domain (54km)
    hgt_d01, lat_d01, lon_d01 = load_wrf_data(
        "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km/wrfinput_d01"
    )
    plot_single_domain(hgt_d01, lat_d01, lon_d01, "d01", "domains_topo_sites_d01.pdf")

    # Plot 2: Nested domains (9km parent, 3km nested)
    hgt_d02, lat_d02, lon_d02 = load_wrf_data(
        "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km/wrfinput_d01"
    )
    hgt_d03, lat_d03, lon_d03 = load_wrf_data(
        "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d02"
    )

    domain_labels = [
        ("d02", lon_d02[0, 0], lat_d02[0, 0]),
        ("d03", lon_d03[0, 0], lat_d03[0, 0]),
    ]
    plot_nested_domains(
        hgt_d02,
        lat_d02,
        lon_d02,
        hgt_d03,
        lat_d03,
        lon_d03,
        domain_labels,
        "domains_topo_sites.pdf",
    )


if __name__ == "__main__":
    main()


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
