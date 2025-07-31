import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configs ---
csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

start_date = "2012-01-01 00:00:00"
end_date = "2012-12-30 00:00:00"
ref_tag = "_54km"
output_file = os.path.join(
    csv_folder, f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}.csv"
)
STD_TOPO = 50

# --- Load Data ---
df = pd.read_csv(output_file, index_col="datetime", parse_dates=True)

# --- Settings ---
resolutions = ["54km", "9km"]
resolution_colors = {
    "54km": "blue",
    "27km": "green",
    "9km": "red",
    "3km": "purple",
}

variable_groups = {
    "dT": "dT (K)",
    "dGPP": "dGPP (gC/m²/day)",
    "dRECO": "dRECO (gC/m²/day)",
}


# --- Plot Function ---
def plot_per_resolution(df, variable_groups):
    for res in resolutions:
        for var_prefix, ylabel in variable_groups.items():
            columns = [
                col
                for col in df.columns
                if col.startswith(var_prefix + "_") and col.endswith("_" + res)
            ]
            plt.figure(figsize=(15, 5))
            label_shown = set()

            for column in columns:
                data_series = df[column].dropna()
                grouped = data_series.groupby(data_series.index.hour)
                hours = np.array(sorted(grouped.groups.keys()))
                avg_values = [grouped.get_group(h).mean() for h in hours]

                type_key = None
                if "calc" in column:
                    type_key = f"{var_prefix} - lapse rate"
                elif "model" in column:
                    type_key = f"{var_prefix} - T2 diff."
                elif "real" in column:
                    type_key = f"{var_prefix} - model"

                type_colors = {
                    f"{var_prefix} - lapse rate": "red",
                    f"{var_prefix} - T2 diff.": "green",
                    f"{var_prefix} - model": "black",
                }

                color = type_colors.get(type_key, "gray")
                show_label = type_key not in label_shown

                plt.plot(
                    hours,
                    avg_values,
                    label=type_key if show_label else None,
                    color=color,
                )
                label_shown.add(type_key)

            plt.xticks(hours)
            plt.title(f"Hourly Avg {var_prefix} for {res} Resolution")
            plt.xlabel("Hour of Day")
            plt.ylabel(ylabel)
            plt.grid(True)
            if var_prefix == "dT":
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
            else:
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
            plt.tight_layout()
            plotname = f"{outfolder}hourly_avg_{var_prefix}_{res}_std_topo_{STD_TOPO}_{start_date}_{end_date}{ref_tag}.pdf"
            print(f"saved plot: {plotname}")
            plt.savefig(plotname, bbox_inches="tight")
            plt.close()


# --- Generate Plots ---
plot_per_resolution(df, variable_groups)
