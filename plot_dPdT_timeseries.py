import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# --- Configs ---
csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

start_date = "2012-06-20 00:00:00"
end_date = "2012-06-30 00:00:00"
ref_tag = "_54km"
output_file = os.path.join(
    csv_folder, f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}.csv"
)
STD_TOPO = 50

# --- Load Data ---
df = pd.read_csv(output_file, index_col="datetime", parse_dates=True)

# --- Plot Settings ---
plt.rcParams.update({"figure.figsize": (12, 6), "lines.linewidth": 1.5})
resolution_colors = {
    "54km": "blue",
    "27km": "green",
    "9km": "red",
    "3km": "purple",
}

ref_sim = ""
subdaily = ""

# --- Identify Resolutions ---
resolutions = ["54km", "9km"]

# --- Variable Groups ---
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
            gap_size = 1
            xticks = []
            xticklabels = []

            plt.figure(figsize=(16, 3))
            label_shown = set()

            for column in columns:
                data_series = df[column].dropna()
                grouped = data_series.groupby(data_series.index.date)
                valid_days = [
                    (date, group)
                    for date, group in grouped
                    if not group.dropna().eq(0).all()
                ]

                current_x = 0
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

                all_y = []
                for i, (date, group) in enumerate(valid_days):
                    y = group.values
                    x = np.arange(len(y)) + current_x
                    plt.plot(
                        x,
                        y,
                        label=type_key if show_label and i == 0 else None,
                        color=color,
                    )
                    all_y.extend(y)

                    # Plot daily average as a horizontal line for this day
                    if len(y) > 0:
                        daily_avg = np.nanmean(y)
                        plt.hlines(
                            daily_avg,
                            x[0],
                            x[-1],
                            color=color,
                            linestyle="--",
                            linewidth=1.0,
                            alpha=0.7,
                        )

                    xticks.append(current_x)
                    xticklabels.append(str(date))

                    if i < len(valid_days) - 1:
                        gap_start = current_x + len(y)
                        gap_end = gap_start + gap_size
                        plt.axvspan(gap_start, gap_end, color="lightgray", alpha=0.5)

                    current_x += len(y) + gap_size

                label_shown.add(type_key)

            plt.xticks(xticks, xticklabels, ha="left")
            plt.title(f"{var_prefix} for {res} Resolution")
            plt.xlabel("date")
            plt.ylabel(ylabel)
            plt.grid(True)
            if var_prefix == "dT":
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
            else:
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
            plt.tight_layout()
            # plt.show()
            plotname = f"{outfolder}timeseries_{var_prefix}_{res}_grouped_std_topo_{STD_TOPO}_{start_date}_{end_date}{ref_tag}.pdf"
            print(f"saved plot: {plotname}")
            plt.savefig(plotname, bbox_inches="tight")
            plt.close()


# --- Generate Plots ---
plot_per_resolution(df, variable_groups)
