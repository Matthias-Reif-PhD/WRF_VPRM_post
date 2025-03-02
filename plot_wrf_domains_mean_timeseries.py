import matplotlib.pyplot as plt
import pandas as pd

csv_folder = "/scratch/c7071034/DATA/WRFOUT/csv/"
outfolder = "/home/c707/c7071034/Github/WRF_VPRM_post/plots/"
start_date = "2012-07-15 00:00:00"
end_date = "2012-07-30 00:00:00"
STD_TOPO = 50
columns = ["RECO_migli", "GPP_pmodel", "NEE_PM", "GPP", "RECO", "NEE", "T2"]
subdaily = "_subdailyC3"  # "_subdailyC3" or ""

# Save to CSV
merged_df_gt = pd.read_csv(
    f"{csv_folder}timeseries_domain_averaged{subdaily}_std_topo_gt_{STD_TOPO}_{start_date}_{end_date}.csv"
)
merged_df_lt = pd.read_csv(
    f"{csv_folder}timeseries_domain_averaged{subdaily}_std_topo_lt_{STD_TOPO}_{start_date}_{end_date}.csv"
)

# Variables to plot
resolutions = ["3km", "9km", "27km", "54km", "CAMS"]

for res in resolutions:
    # Extract the series
    merged_df_gt[f"NEE_{res}"] = (
        -merged_df_gt[f"GPP_{res}"] + merged_df_gt[f"RECO_{res}"]
    )
    merged_df_lt[f"NEE_{res}"] = (
        -merged_df_lt[f"GPP_{res}"] + merged_df_lt[f"RECO_{res}"]
    )
for res in resolutions[:-1]:
    merged_df_gt[f"NEE_PM_{res}"] = (
        -merged_df_gt[f"GPP_pmodel_{res}"] + merged_df_gt[f"RECO_migli_{res}"]
    )
    merged_df_lt[f"NEE_PM_{res}"] = (
        -merged_df_lt[f"GPP_pmodel_{res}"] + merged_df_lt[f"RECO_migli_{res}"]
    )

resolutions_diff = ["3km", "9km", "27km"]
for column in columns:
    for res in resolutions_diff:
        merged_df_gt[f"diff_{column}_54km-{res}"] = (
            merged_df_gt[f"{column}_54km"] - merged_df_gt[f"{column}_{res}"]
        )
        merged_df_lt[f"diff_{column}_54km-{res}"] = (
            merged_df_lt[f"{column}_54km"] - merged_df_lt[f"{column}_{res}"]
        )


merged_df_gt["datetime"] = pd.to_datetime(
    merged_df_gt["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S"
)
merged_df_gt.set_index("datetime", inplace=True)
merged_df_gt["hour"] = merged_df_gt.index.hour
numeric_columns = merged_df_gt.select_dtypes(include=["number"]).columns
hourly_avg = merged_df_gt[numeric_columns].groupby("hour").mean()

merged_df_lt["datetime"] = pd.to_datetime(
    merged_df_lt["Unnamed: 0"], format="%Y-%m-%d %H:%M:%S"
)
merged_df_lt.set_index("datetime", inplace=True)
merged_df_lt["hour"] = merged_df_lt.index.hour
numeric_columns = merged_df_lt.select_dtypes(include=["number"]).columns
hourly_avg_lt = merged_df_lt[numeric_columns].groupby("hour").mean()


resolution_colors = {
    "3km": "blue",
    "9km": "purple",
    "27km": "red",
    "54km": "green",
    "CAMS": "orange",
}
print("start plotting")
# Create separate plots for each variable
for column in columns:
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
            data_series_lt = merged_df_lt[cams_column]
        else:
            data_series = merged_df_gt[f"{column}_{res}"]
            data_series_lt = merged_df_lt[f"{column}_{res}"]
        # Skip NaN values for CAMS data during plotting
        if res == "CAMS":
            data_series = data_series.dropna()
            data_series_lt = data_series_lt.dropna()

        # Plot the data
        plt.plot(
            data_series.index,
            data_series,
            label=f"{column} {res} > std {STD_TOPO}",
            linestyle="-",
            color=resolution_colors[res],
        )
        plt.plot(
            data_series_lt.index,
            data_series_lt,
            label=f"{column} {res} < std {STD_TOPO}",
            linestyle=":",
            color=resolution_colors[res],
        )
    # Customize the plot
    plt.title(f"Comparison of {column} Across Resolutions")
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"{outfolder}timeseries_{column}_domain_averaged{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    )
    plt.close()

resolutions_diff = ["3km", "9km", "27km"]

for column in columns:
    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions_diff:
        plt.plot(
            merged_df_gt.index,
            merged_df_gt[f"{column}_54km"] - merged_df_gt[f"{column}_{res}"],
            label=f"{column} 54km-{res} > std {STD_TOPO}",
            linestyle="-",
            color=resolution_colors[res],
        )
        plt.plot(
            merged_df_lt.index,
            merged_df_lt[f"{column}_54km"] - merged_df_lt[f"{column}_{res}"],
            label=f"{column} 54km-{res} < std {STD_TOPO}",
            linestyle=":",
            color=resolution_colors[res],
        )

    # Customize the plot
    plt.title(f"Differences to dx=54km of {column} Across Resolutions")
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"{outfolder}timeseries_diff_of_54km_{column}_domain_averaged{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    )
    plt.close()

print("plotting hourly averages")
# Create separate plots for each variable
for column in columns:
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
            data_series_lt = hourly_avg_lt[cams_column]
        else:
            data_series = hourly_avg[f"{column}_{res}"]
            data_series_lt = hourly_avg_lt[f"{column}_{res}"]

        # Skip NaN values for CAMS data during plotting
        if res == "CAMS":
            data_series = data_series.dropna()
            data_series_lt = data_series_lt.dropna()

        # Plot the data
        plt.plot(
            data_series.index,
            data_series,
            label=f"{column} {res} > std {STD_TOPO}",
            linestyle="-",
            color=resolution_colors[res],
        )
        plt.plot(
            data_series_lt.index,
            data_series_lt,
            label=f"{column} {res} < std {STD_TOPO}",
            linestyle=":",
            color=resolution_colors[res],
        )

    # Customize the plot
    plt.title(f"Comparison hourly averages of {column} Across Resolutions")
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"{outfolder}timeseries_hourly_{column}_domain_averaged{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    )
    plt.close()

resolutions_diff = ["3km", "9km", "27km"]

for column in columns:
    plt.figure(figsize=(10, 6))
    # Extract data for the current variable across all resolutions
    for res in resolutions_diff:
        plt.plot(
            hourly_avg.index,
            hourly_avg[f"{column}_54km"] - hourly_avg[f"{column}_{res}"],
            label=f"{column} 54km-{res} > std {STD_TOPO}",
            linestyle="-",
            color=resolution_colors[res],
        )
        plt.plot(
            hourly_avg_lt.index,
            hourly_avg_lt[f"{column}_54km"] - hourly_avg_lt[f"{column}_{res}"],
            label=f"{column} 54km-{res} < std {STD_TOPO}",
            linestyle=":",
            color=resolution_colors[res],
        )

    # Customize the plot
    plt.title(
        f"Differences to dx=54km of hourly averages of {column} Across Resolutions"
    )
    plt.xlabel("Time")
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"{outfolder}timeseries_hourly_diff_of_54km_{column}_domain_averaged{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
    )
    plt.close()


resolutions = ["3km", "9km", "27km", "54km"]

plt.figure(figsize=(10, 6))
# Extract data for the current variable across all resolutions
for res in resolutions:
    plt.plot(
        hourly_avg.index,
        hourly_avg[f"GPP_{res}"] - hourly_avg[f"GPP_pmodel_{res}"],
        label=f"{res} > std {STD_TOPO}",
        linestyle="-",
        color=resolution_colors[res],
    )
    plt.plot(
        hourly_avg_lt.index,
        hourly_avg_lt[f"GPP_{res}"] - hourly_avg_lt[f"GPP_pmodel_{res}"],
        label=f"{res} < std {STD_TOPO}",
        linestyle=":",
        color=resolution_colors[res],
    )

# Customize the plot
plt.title(f"Difference of GPP_WRF - GPP_pmodel Across Resolutions")
plt.xlabel("Time")
plt.ylabel("GPP")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig(
    f"{outfolder}timeseries_hourly_diff_of GPP_WRF_vs_pmodel_domain_averaged{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
)
plt.close()

plt.figure(figsize=(10, 6))
# Extract data for the current variable across all resolutions
for res in resolutions:
    plt.plot(
        hourly_avg.index,
        hourly_avg[f"RECO_{res}"] - hourly_avg[f"RECO_migli_{res}"],
        label=f"{res} > std {STD_TOPO}",
        linestyle="-",
        color=resolution_colors[res],
    )
    plt.plot(
        hourly_avg_lt.index,
        hourly_avg_lt[f"RECO_{res}"] - hourly_avg_lt[f"RECO_migli_{res}"],
        label=f"{res} < std {STD_TOPO}",
        linestyle=":",
        color=resolution_colors[res],
    )

# Customize the plot
plt.title(f"Difference of RECO_WRF - RECO_migli Across Resolutions")
plt.xlabel("Time")
plt.ylabel("GPP")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig(
    f"{outfolder}timeseries_hourly_diff_of_RECO_WRF_vs_Migli_domain_averaged{subdaily}_std_topo_{STD_TOPO}_{start_date}_{end_date}.png"
)
plt.close()

print("finished plotting")
