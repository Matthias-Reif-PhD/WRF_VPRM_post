"""
Fig2_PFTs_d033_d01.py

Script to produce Fig.2 PFT maps (dominant + fractional) by reusing
functions from `plot_veg_fraction.py`.

This script generates two plots per domain/dx pair:
 - site-matched vegetation map with hatched low-std areas
 - dominant PFT map derived from VPRM vegetation fractions

Edit the `DOMAINS` list below to change which domain/resolution pairs
are produced.
"""

import os
import sys
from pathlib import Path

# Try to import helper functions from the refactored module
try:
    from plot_veg_fraction import (
        load_wrf_data,
        load_veg_fraction,
        match_sites_to_grid,
        plot_veg_map_with_sites,
        plot_veg_fractions_with_pies,
        SITES_1KM,
        SITES_COARSE,
        OUTFOLDER,
    )
except Exception as e:
    print("Could not import from plot_veg_fraction.py:", e)
    print("Make sure this script lives in the same folder as plot_veg_fraction.py")
    raise

# Make sure output folder exists
Path(OUTFOLDER).mkdir(parents=True, exist_ok=True)

# Domains to process: list of (domain, dx)
# Adjust to your needs. The examples below produce plots for a 3km (d03-like)
# and a 54km (d01-like) configuration.
DOMAINS = [
    {"domain": "_d03", "dx": "_3km"},
    {"domain": "_d01", "dx": "_54km"},
]


# Map dx to a sensible wrfinput path - update if your repository uses different paths
def wrfinput_for_dx(dx):
    if dx == "_1km":
        return "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d02"
    if dx == "_3km":
        return "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_3km/wrfinput_d01"
    if dx == "_9km":
        return "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_9km/wrfinput_d01"
    if dx == "_27km":
        return "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_27km/wrfinput_d01"
    if dx == "_54km":
        return "/scratch/c7071034/DATA/WRFOUT/WRFOUT_ALPS_54km/wrfinput_d01"
    # fallback
    return None


# For each domain/dx, generate both plots
def run_for_pair(domain, dx):
    print(f"\n=== Processing domain={domain} dx={dx} ===")

    base_mz = f"/scratch/c7071034/DATA/pyVPRM/pyVPRM_examples/wrf_preprocessor/out{domain}_2012{dx}"

    # choose WRF input and VPRM input file paths
    wrfinput = wrfinput_for_dx(dx)

    # Attempt to find a VPRM VEG FRA file in base_mz
    vprm_candidate = os.path.join(base_mz, f"VPRM_input_VEG_FRA{domain}_2012.nc")
    if not os.path.exists(vprm_candidate):
        # try alternative filename used in script
        vprm_candidate = os.path.join(base_mz, f"VPRM_input_VEG_FRA{domain}_2012.nc")

    if not os.path.exists(vprm_candidate):
        print(f"Warning: VPRM fraction file not found at {vprm_candidate}")

    # Load WRF & VPRM inputs
    if wrfinput is None or not os.path.exists(wrfinput):
        print(f"Warning: wrfinput not found for dx={dx}: {wrfinput}")
        print("Skipping site-matched plot for this pair.")
    else:
        print("Loading WRF data from:", wrfinput)
        ivgtyp, xlat, xlong, hgt_m, stdvar, ivgtyp_vprm = load_wrf_data(wrfinput)

        # choose sites list
        sites = SITES_1KM if dx == "_1km" else SITES_COARSE

        # load veg fractions (some workflows use a separate VPRM input for plotting)
        # If the dedicated VPRM input exists, try that first; otherwise call load_veg_fraction
        try:
            veg_frac_map = load_veg_fraction(vprm_candidate)
        except Exception as e:
            print(f"Warning: could not load veg fraction from {vprm_candidate}: {e}")
            veg_frac_map = None

        # Match sites and produce site-matched plot
        try:
            print("Matching sites and creating site-matched map...")
            sites_matched = match_sites_to_grid(
                sites, xlat, xlong, ivgtyp, hgt_m, veg_frac_map, radius=30
            )
            import pandas as pd

            df_sites = pd.DataFrame(sites_matched)

            veg_type = ivgtyp_vprm if ivgtyp_vprm.shape == hgt_m.shape else None
            if veg_type is None:
                # fallback: map CORINE to VPRM
                import numpy as np

                veg_type = np.vectorize(lambda v: CORINE_TO_VPRM.get(v, 8))(ivgtyp)

            mask_low_std = stdvar < 200
            plot_veg_map_with_sites(
                xlat,
                xlong,
                veg_type,
                mask_low_std,
                df_sites,
                f"{domain}{dx}",
                std_threshold=200,
            )
        except Exception as e:
            print("Error creating site-matched map:", e)

    # Always produce the dominant-PFT map from veg fractions (this replaced the pie-chart approach)
    try:
        print("Creating dominant-PFT map from VPRM veg fractions...")
        plot_veg_fractions_with_pies(
            base_mz, domain, dx, extent_ds_path="/scratch/c7071034/WPS/geo_em.d02.nc"
        )
    except Exception as e:
        print("Error creating dominant-PFT map with extent crop:", e)
        print("Retrying without extent crop...")
        try:
            plot_veg_fractions_with_pies(base_mz, domain, dx, extent_ds_path=None)
        except Exception as e2:
            print("Error creating dominant-PFT map:", e2)


if __name__ == "__main__":
    # Run for each domain/dx pair
    for pair in DOMAINS:
        run_for_pair(pair["domain"], pair["dx"])
