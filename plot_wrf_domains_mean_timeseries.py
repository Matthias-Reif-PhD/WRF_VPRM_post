import matplotlib.pyplot as plt
import pandas as pd

csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"

start_date = "2012-07-02 00:00:00"
end_date = "2012-07-03 00:00:00"
STD_TOPO = 50
plot_lt = False
ref_sim = "" # "_REF" to use REF simulation or "" for tuned values
run_Pmodel = False
if run_Pmodel:
    columns = ["RECO_migli", "GPP_pmodel", "NEE_PM", "GPP", "RECO", "NEE", "T2"]
else:
    columns = ["GPP", "RECO", "NEE", "T2"]
subdaily = ""  # "_subdailyC3" or ""
units = [
    " [µmol m² s⁻¹]",
    " [µmol m² s⁻¹]",
    " [µmol m² s⁻¹]",
    " [µmol m² s⁻¹]",
    " [µmol m² s⁻¹]",
    " [µmol m² s⁻¹]",
    " [°C]",
]
# Save to CSV
merged_df_gt = pd.read_csv(
    f"{csv_folder}timeseries_domain_averaged{ref_sim}{subdaily}_std_topo_gt_{STD_TOPO}_{start_date}_{end_date}.csv"
)
if plot_lt:
    merged_df_lt = pd.read_csv(
        f"{csv_folder}timeseries_domain_averaged{ref_sim}{subdaily}_std_topo_lt_{STD_TOPO}_{start_date}_{end_date}.csv"
    )

# Variables to plot
resolutions = ["1km","3km", "9km", "27km", "54km", "CAMS"]

for res in resolutions:
    # Extract the series
    merged_df_gt[f"NEE_{res}"] = (
        -merged_df_gt[f"GPP_{res}"] + merged_df_gt[f"RECO_{res}"]
    )
    if plot_lt:
        merged_df_lt[f"NEE_{res}"] = (
            -merged_df_lt[f"GPP_{res}"] + merged_df_lt[f"RECO_{res}"]
        )
if run_Pmodel:
    for res in resolutions[:-1]:
        merged_df_gt[f"NEE_PM_{res}"] = (
            -merged_df_gt[f"GPP_pmodel_{res}"] + merged_df_gt[f"RECO_migli_{res}"]
        )
        if plot_lt:
            merged_df_lt[f"NEE_PM_{res}"] = (
                -merged_df_lt[f"GPP_pmodel_{res}"] + merged_df_lt[f"RECO_migli_{res}"]
            )

# resolutions_diff = ["3km", "9km", "27km"]
# for column in columns:
#     for res in resolutions_diff:
#         merged_df_gt[f"diff_{column}_54km-{res}"] = (
#             merged_df_gt[f"{column}_54km"] - merged_df_gt[f"{column}_{res}"]
#         )
#         merged_df_lt[f"diff_{column}_54km-{res}"] = (
#             merged_df_lt[f"{column}_54km"] - merged_df_lt[f"{column}_{res}"]
#         )


merged_df_gt["datetime"] = pd.to_datetime(
    merged_df_gt["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S"
)
merged_df_gt.set_index("datetime", inplace=True)
merged_df_gt["hour"] = merged_df_gt.index.hour
numeric_columns = merged_df_gt.select_dtypes(include=["number"]).columns
hourly_avg = merged_df_gt[numeric_columns].groupby("hour").mean()

if plot_lt:
    merged_df_lt["datetime"] = pd.to_datetime(
        merged_df_lt["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S"
    )
    merged_df_lt.set_index("datetime", inplace=True)
    merged_df_lt["hour"] = merged_df_lt.index.hour
    numeric_columns = merged_df_lt.select_dtypes(include=["number"]).columns
    hourly_avg_lt = merged_df_lt[numeric_columns].groupby("hour").mean()


resolution_colors = {
    "1km": "black",
    "3km": "blue",
    "9km": "purple",
    "27km": "red",
    "54km": "green",
    "CAMS": "orange",
}

# Create separate plots for each variable
for column, unit in zip(columns, units):
    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions:
        # Extract the series
        if (
            f"{column}_{res}" == "GPP_pmodel_CAMS"
            or f"{column}_{res}" == "RECO_migli_CAMS"
            or f"{column}_{res}" == "NEE_PM_CAMS"
        ):
            cams_column = column.split("_")[0] + "_CAMS"
            data_series = merged_df_gt[cams_column]
            if plot_lt:
                data_series_lt = merged_df_lt[cams_column]
        else:
            data_series = merged_df_gt[f"{column}_{res}"]
            if plot_lt:
                data_series_lt = merged_df_lt[f"{column}_{res}"]
        # Skip NaN values for CAMS data during plotting
        if res == "CAMS":
            data_series = data_series.dropna()
            if plot_lt:
                data_series_lt = data_series_lt.dropna()
        if column != "T2":
            # hourly_avg = merged_df_gt[numeric_columns].groupby("hour").mean()
            gC_per_day = data_series.mean() * 60 * 60 * 24 * 1e-6 * 12  # gC m^-2 d^-1
            label_i = f"{column} {res} > std {STD_TOPO} mean={gC_per_day:.2f} gC/m²/day"
            if plot_lt:
                gC_per_day_lt = (
                    data_series_lt.mean() * 60 * 60 * 24 * 1e-6 * 12
                )  # gC m^-2 d^-1
                label_i_lt = (
                    f"{column} {res} < std {STD_TOPO} mean={gC_per_day_lt:.2f} gC/m²/day"
                )
        else:
            sum_over_time = data_series.mean()
            label_i = f"{column}{ref_sim} {res} > std {STD_TOPO}; mean={sum_over_time:.2f}"
            if plot_lt:
                sum_over_time_lt = data_series_lt.mean()
                label_i_lt = f"{column}{ref_sim} {res} < std {STD_TOPO}; mean={sum_over_time_lt:.2f}"
        # Plot the data
        plt.plot(
            data_series.index,
            data_series,
            label=label_i,
            linestyle="-",
            color=resolution_colors[res],
        )
        if plot_lt:
            plt.plot(
                data_series_lt.index,
                data_series_lt,
                label=label_i_lt,
                linestyle=":",
                color=resolution_colors[res],
            )
    # Customize the plot
    plt.title(f"Comparison of {column} Across resolutions")
    plt.xlabel("Time")
    plt.ylabel(column + " " + unit)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plotname = f"{outfolder}timeseries_{column}_domain_averaged{ref_sim}{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    print(f"saved plot: {plotname}")
    plt.savefig(plotname)
    plt.close()

    resolutions_diff = ["54km", "27km", "9km", "3km"]

    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions_diff:
        diff_gt = merged_df_gt[f"{column}_{res}"] - merged_df_gt[f"{column}_1km"]
        if plot_lt:
            diff_lt = merged_df_lt[f"{column}_{res}"] - merged_df_lt[f"{column}_1km"]
        if column != "T2":
            gC_per_day = diff_gt.mean() * 60 * 60 * 24 * 1e-6 * 12  # gC m^-2 d^-1
            label_gt = (
                f"{column} {res}-1km > std {STD_TOPO} mean={gC_per_day:.2f} gC/m²/day"
            )
            if plot_lt:
                gC_per_day_lt = diff_lt.mean() * 60 * 60 * 24 * 1e-6 * 12  # gC m^-2 d^-1
                label_lt = f"{column} {res}-1km < std {STD_TOPO} mean={gC_per_day_lt:.2f} gC/m²/day"
        else:
            gC_per_day = diff_gt.mean()
            label_gt = f"{column}{ref_sim} {res}-1km > std {STD_TOPO}; mean={gC_per_day:.2f}"
            if plot_lt:
                gC_per_day_lt = diff_lt.mean()
                label_lt = f"{column}{ref_sim} {res}-1km < std {STD_TOPO}; mean={gC_per_day_lt:.2f}"
        plt.plot(
            diff_gt.index,
            diff_gt,
            label=label_gt,
            linestyle="-",
            color=resolution_colors[res],
        )
        if plot_lt:
            plt.plot(
                diff_lt.index,
                diff_lt,
                label=label_lt,
                linestyle=":",
                color=resolution_colors[res],
            )

    # Customize the plot
    plt.title(f"Differences of coarse(dx)-1km of {column} across resolutions")
    plt.xlabel("Time")
    plt.ylabel(column + " " + unit)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plotname = f"{outfolder}timeseries_diff_of_dx-1km_{column}_domain_averaged{ref_sim}{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    print(f"saved plot: {plotname}")
    plt.savefig(plotname)
    plt.close()

    # Create separate plots for each variable
    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions:
        # Skip NaN values for CAMS data during plotting
        # Extract the series
        if (
            f"{column}_{res}" == "GPP_pmodel_CAMS"
            or f"{column}_{res}" == "RECO_migli_CAMS"
            or f"{column}_{res}" == "NEE_PM_CAMS"
        ):
            cams_column = column.split("_")[0] + "_CAMS"
            data_series = hourly_avg[cams_column]
            if plot_lt:
                data_series_lt = hourly_avg_lt[cams_column]
        else:
            data_series = hourly_avg[f"{column}_{res}"]
            if plot_lt:
                data_series_lt = hourly_avg_lt[f"{column}_{res}"]

        # Skip NaN values for CAMS data during plotting
        if res == "CAMS":
            data_series = data_series.dropna()
            if plot_lt:
                data_series_lt = data_series_lt.dropna()
        if column != "T2":  # Daily mean
            gC_per_day = data_series.mean() * 60 * 60 * 24 * 1e-6 * 12  # gC m^-2 d^-1
            label_i = f"{column} {res} > std {STD_TOPO} with {gC_per_day:.2f} gC/m²/day"
            if plot_lt:
                gC_per_day_lt = (
                    data_series_lt.mean() * 60 * 60 * 24 * 1e-6 * 12
                )  # gC m^-2 d^-1
                label_i_lt = (
                    f"{column} {res} < std {STD_TOPO} with {gC_per_day_lt:.2f} gC/m²/day"
                )
        else:
            sum_over_time = data_series.mean()
            label_i = f"{column}{ref_sim} {res} > std {STD_TOPO}; mean: {sum_over_time:.2f}"
            if plot_lt:
                sum_over_time_lt = data_series_lt.mean()
                label_i_lt = (
                    f"{column}{ref_sim} {res} < std {STD_TOPO}; mean: {sum_over_time_lt:.2f}"
                )
        # Plot the data
        plt.plot(
            data_series.index,
            data_series,
            label=label_i,
            linestyle="-",
            color=resolution_colors[res],
        )
        if plot_lt:
            plt.plot(
                data_series_lt.index,
                data_series_lt,
                label=label_i_lt,
                linestyle=":",
                color=resolution_colors[res],
            )

    # Customize the plot
    plt.title(f"Comparison hourly averages of {column} Across resolutions")
    plt.xlabel("Time")
    plt.ylabel(column + " " + unit)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plotname = f"{outfolder}timeseries_hourly_{column}_domain_averaged{ref_sim}{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    print(f"saved plot: {plotname}")
    plt.savefig(plotname)
    plt.close()

    resolutions_diff = ["54km", "27km", "9km", "3km"]

    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions_diff:
        diff_gt = hourly_avg[f"{column}_{res}"] - hourly_avg[f"{column}_1km"]
        if plot_lt:
            diff_lt = hourly_avg_lt[f"{column}_{res}"] - hourly_avg_lt[f"{column}_1km"]
        if column != "T2":
            gC_per_day = diff_gt.mean() * 60 * 60 * 24 * 1e-6 * 12  # gC m^-2 d^-1
            label_gt = (
                f"{column} {res}-1km > std {STD_TOPO}; mean={gC_per_day:.2f} gC/m²/day"
            )
            if plot_lt:
                gC_per_day_lt = diff_lt.mean() * 60 * 60 * 24 * 1e-6 * 12  # gC m^-2 d^-1
                label_lt = f"{column} {res}-1km < std {STD_TOPO}; mean={gC_per_day_lt:.2f} gC/m²/day"
        else:
            sum_over_time_gt = diff_gt.mean()
            label_gt = (
                f"{column}{ref_sim} {res}-1km > std {STD_TOPO}; mean={sum_over_time_gt:.2f}"
            )
            if plot_lt:
                sum_over_time_lt = diff_lt.mean()
                label_lt = (
                    f"{column}{ref_sim} {res}-1km < std {STD_TOPO}; mean={sum_over_time_lt:.2f}"
                )
        plt.plot(
            diff_gt.index,
            diff_gt,
            label=label_gt,
            linestyle="-",
            color=resolution_colors[res],
        )
        if plot_lt:
            plt.plot(
                diff_lt.index,
                diff_lt,
                label=label_lt,
                linestyle=":",
                color=resolution_colors[res],
            )

    # Customize the plot
    plt.title(f"Differences of coarse(dx)-1km of {column} across resolutions")
    plt.xlabel("Time")
    plt.ylabel(column + " " + unit)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plotname = f"{outfolder}timeseries_hourly_diff_of_54km_{column}_domain_averaged{ref_sim}{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    print(f"saved plot: {plotname}")
    plt.savefig(plotname)
    plt.close()


resolutions = ["1km","3km", "9km", "27km", "54km"]
if run_Pmodel:
    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions:
        data_series = hourly_avg[f"GPP_{res}"] - hourly_avg[f"GPP_pmodel_{res}"]
        if plot_lt:
            data_series_lt = hourly_avg_lt[f"GPP_{res}"] - hourly_avg_lt[f"GPP_pmodel_{res}"]
        gC_per_day = data_series.mean() * 60 * 60 * 24 * 1e-6 * 12
        label_i = f"{res}{ref_sim} > std {STD_TOPO}; mean={gC_per_day:.2f}"
        if plot_lt:

            gC_per_day_lt = data_series_lt.mean() * 60 * 60 * 24 * 1e-6 * 12
            label_i_lt = f"{res}{ref_sim} < std {STD_TOPO}; mean={gC_per_day_lt:.2f}"

        plt.plot(
            hourly_avg.index,
            hourly_avg[f"GPP_{res}"] - hourly_avg[f"GPP_pmodel_{res}"],
            label=label_i,
            linestyle="-",
            color=resolution_colors[res],
        )
        if plot_lt:
            plt.plot(
                hourly_avg_lt.index,
                hourly_avg_lt[f"GPP_{res}"] - hourly_avg_lt[f"GPP_pmodel_{res}"],
                label=label_i_lt,
                linestyle=":",
                color=resolution_colors[res],
            )

    # Customize the plot
    plt.title(f"Difference of GPP_WRF - GPP_pmodel within resolutions")
    plt.xlabel("Time")
    plt.ylabel("GPP")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plotname = f"{outfolder}timeseries_hourly_diff_of GPP_WRF_vs_pmodel_domain_averaged{ref_sim}{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    print(f"saved plot: {plotname}")
    plt.savefig(plotname)
    plt.close()


    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions:
        data_series = hourly_avg[f"RECO_{res}"] - hourly_avg[f"RECO_migli_{res}"]
        if plot_lt:
            data_series_lt = hourly_avg_lt[f"RECO_{res}"] - hourly_avg_lt[f"RECO_migli_{res}"]
        gC_per_day = data_series.mean() * 60 * 60 * 24 * 1e-6 * 12
        label_i = f"{res}{ref_sim} > std {STD_TOPO}; mean={gC_per_day:.2f} gC/m²/day"
        if plot_lt:
            gC_per_day_lt = data_series_lt.mean() * 60 * 60 * 24 * 1e-6 * 12
            label_i_lt = f"{res}{ref_sim} < std {STD_TOPO}; mean={gC_per_day_lt:.2f} gC/m²/day"
        plt.plot(
            hourly_avg.index,
            hourly_avg[f"RECO_{res}"] - hourly_avg[f"RECO_migli_{res}"],
            label=f"{res} > std {STD_TOPO}",
            linestyle="-",
            color=resolution_colors[res],
        )
        if plot_lt:
            plt.plot(
                hourly_avg_lt.index,
                hourly_avg_lt[f"RECO_{res}"] - hourly_avg_lt[f"RECO_migli_{res}"],
                label=f"{res} < std {STD_TOPO}",
                linestyle=":",
                color=resolution_colors[res],
            )

    # Customize the plot
    plt.title(f"Difference of RECO_WRF - RECO_migli within resolutions")
    plt.xlabel("Time")
    plt.ylabel("GPP")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plotname = f"{outfolder}timeseries_hourly_diff_of_RECO_WRF_vs_Migli_domain_averaged{ref_sim}{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    print(f"saved plot: {plotname}")
    plt.savefig(plotname)
    plt.close()

print("finished plotting")
