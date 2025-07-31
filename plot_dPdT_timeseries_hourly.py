import matplotlib.colors as mcolors
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

########### Settings ###########
csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
start_date = "2012-01-01 00:00:00"
end_date = "2012-12-30 00:00:00"
ref_tag = ""
STD_TOPO = 50
###############################4

input_file = os.path.join(
    csv_folder, f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}.csv"
)
# TODO "_REF" simulations should be compared to optimized data
ref_sim = "_REF"
input_file_REF = os.path.join(
    csv_folder, f"dPdT_timeseries_{start_date}_{end_date}{ref_tag}{ref_sim}.csv"
)
# end TODO

df = pd.read_csv(input_file, index_col="datetime", parse_dates=True)

resolutions = ["54km", "9km"]


variable_groups = {
    "dT": "ΔT [°C]",
    "dGPP": "ΔGPP [gC/m²/day]",
    "dRECO": "ΔRECO [gC/m²/day]",
}

# Read REF data
df_ref = pd.read_csv(input_file_REF, index_col="datetime", parse_dates=True)

resolution_colors = {
    "54km": "red",
    "9km": "blue",
}
resolution_colors_light = {
    "54km": mcolors.to_rgba("red", alpha=0.4),
    "9km": mcolors.to_rgba("blue", alpha=0.4),
}


# --- Refactored Plot Function ---
def plot_combined(df, df_ref, variable_groups):
    for var_prefix, ylabel in variable_groups.items():
        plt.figure(figsize=(10, 6))
        label_shown = set()

        for res in resolutions:
            columns = [
                col
                for col in df.columns
                if col.startswith(var_prefix + "_") and col.endswith("_" + res)
            ]
            ref_columns = [
                col
                for col in df_ref.columns
                if col.startswith(var_prefix + "_") and col.endswith("_" + res)
            ]

            for column in columns[::-1]:
                data_series = df[column].dropna()
                grouped = data_series.groupby(data_series.index.hour)
                hours = np.array(sorted(grouped.groups.keys()))
                avg_values = [grouped.get_group(h).mean() for h in hours]

                if "model" in column:
                    type_key = f"{var_prefix} ΔTcalc."
                    color = resolution_colors_light.get(res, "gray")
                    if var_prefix == "dT":
                        type_key = f"{var_prefix}"
                        color = resolution_colors.get(res, "gray")
                elif "real" in column:
                    type_key = f"{var_prefix}"
                    color = resolution_colors.get(res, "gray")
                else:
                    continue

                label = f"{type_key} ({res})"
                show_label = label not in label_shown

                plt.plot(
                    hours,
                    avg_values,
                    label=label if show_label else None,
                    color=color,
                )
                label_shown.add(label)

            for column in ref_columns[::-1]:
                data_series = df_ref[column].dropna()
                grouped = data_series.groupby(data_series.index.hour)
                hours = np.array(sorted(grouped.groups.keys()))
                avg_values = [grouped.get_group(h).mean() for h in hours]

                if "model" in column:
                    type_key = f"{var_prefix} ΔTcalc."
                    color = resolution_colors_light.get(res, "gray")
                    if var_prefix == "dT":
                        type_key = f"{var_prefix}"
                        color = resolution_colors.get(res, "gray")
                elif "real" in column:
                    type_key = f"{var_prefix}"
                    color = resolution_colors.get(res, "gray")
                else:
                    continue

                label = f"{type_key} ({res}, REF)"
                show_label = label not in label_shown

                plt.plot(
                    hours,
                    avg_values,
                    linestyle="dotted",
                    label=label if show_label else None,
                    color=color,
                )
                label_shown.add(label)

        plt.xticks(hours)
        plt.xlabel("UTC [h]")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plotname = f"{outfolder}hourly_avg_{var_prefix}_combined_std_topo_{STD_TOPO}_{start_date}_{end_date}.pdf"
        print(f"saved plot: {plotname}")
        # plt.show()
        plt.savefig(plotname, bbox_inches="tight")
        plt.close()


# --- Generate Plots ---
plot_combined(df, df_ref, variable_groups)
