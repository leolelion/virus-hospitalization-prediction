import requests
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

REGION_COORDS = {
    "ile_de_france": (48.8566, 2.3522),   # Paris
    "auvergne_rhone_alpes": (45.7640, 4.8357),  # Lyon
    # Add more if needed
}

POLLUTANTS = [
    "pm10",
    "pm2_5",
    "nitrogen_dioxide",
    "ozone"
]

BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_air_quality_openmeteo(start="2022-01-01", end=None, aggregate_weekly=True):
    print("=== Fetching Open-Meteo Air Quality ===")

    if end is None:
        end = date.today().isoformat()

    all_frames = []

    for region, (lat, lon) in REGION_COORDS.items():
        print(f"\n--> Region: {region} ({lat},{lon})")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(POLLUTANTS)
        }

        try:
            resp = requests.get(BASE_URL, params=params, timeout=30)
            print("Status:", resp.status_code)
            resp.raise_for_status()
        except Exception as e:
            print("Request failed:", e)
            continue

        js = resp.json()

        if "hourly" not in js or "time" not in js["hourly"]:
            print("No hourly data available.")
            continue

        df = pd.DataFrame(js["hourly"])
        df["region"] = region
        df["latitude"] = lat
        df["longitude"] = lon

        # Convert time to datetime for aggregation
        df["time"] = pd.to_datetime(df["time"])
        
        if aggregate_weekly:
            print(f"Aggregating {len(df)} hourly records to weekly averages...")
            
            # Set time as index for resampling
            df_indexed = df.set_index("time")
            
            # Group by week (Monday to Sunday) and calculate weekly averages for pollutants
            pollutant_cols = [col for col in POLLUTANTS if col in df_indexed.columns]
            
            # Resample to weekly periods starting on Monday ('W-MON')
            weekly_df = df_indexed.groupby(['region', 'latitude', 'longitude']).resample('W-MON').agg({
                **{col: 'mean' for col in pollutant_cols},  # Average pollutant values
            }).reset_index()
            
            # Rename time column for consistency
            weekly_df = weekly_df.rename(columns={'time': 'week_start'})
            
            print(f"Aggregated to {len(weekly_df)} weekly records.")
            all_frames.append(weekly_df)
        else:
            # Keep hourly data
            all_frames.append(df)
            print(f"Fetched {len(df)} hourly rows.")

    if not all_frames:
        print("No data fetched at all.")
        return pd.DataFrame()

    final_df = pd.concat(all_frames, ignore_index=True)
    
    if aggregate_weekly:
        print(f"\n=== Done. Total weekly records: {len(final_df)} ===")
        print("Columns:", final_df.columns.tolist())
        print(f"Date range: {final_df['week_start'].min().date()} to {final_df['week_start'].max().date()}")
    else:
        print(f"\n=== Done. Total hourly rows: {len(final_df)} ===")

    return final_df


def plot_pollutant_trends(df, region_filter="ile_de_france"):
    """Plot time series of pollutant data for a specific region"""
    
    # Filter data for the specified region
    region_data = df[df['region'] == region_filter].copy()
    
    if region_data.empty:
        print(f"No data found for region: {region_filter}")
        return
    
    # Sort by date
    region_data = region_data.sort_values('week_start')
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots for each pollutant
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Weekly Air Quality Trends - {region_filter.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    # Plot each pollutant
    pollutants = ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone']
    pollutant_labels = ['PM10 (μg/m³)', 'PM2.5 (μg/m³)', 'NO₂ (μg/m³)', 'O₃ (μg/m³)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (pollutant, label, color) in enumerate(zip(pollutants, pollutant_labels, colors)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Plot the time series
        ax.plot(region_data['week_start'], region_data[pollutant], 
               color=color, linewidth=2, alpha=0.8, label=label)
        
        # Add trend line
        x_numeric = pd.to_numeric(region_data['week_start'])
        z = np.polyfit(x_numeric, region_data[pollutant], 1)
        p = np.poly1d(z)
        ax.plot(region_data['week_start'], p(x_numeric), 
               '--', color='gray', alpha=0.7, linewidth=1)
        
        # Formatting
        ax.set_title(label, fontweight='bold', fontsize=12)
        ax.set_ylabel('Concentration (μg/m³)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Add statistics text box
        mean_val = region_data[pollutant].mean()
        max_val = region_data[pollutant].max()
        min_val = region_data[pollutant].min()
        
        stats_text = f'Mean: {mean_val:.1f}\nMax: {max_val:.1f}\nMin: {min_val:.1f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Pollutant Statistics for {region_filter.replace('_', ' ').title()} ===")
    print(f"Date range: {region_data['week_start'].min().date()} to {region_data['week_start'].max().date()}")
    print(f"Total weeks: {len(region_data)}")
    
    for pollutant, label in zip(pollutants, pollutant_labels):
        mean_val = region_data[pollutant].mean()
        std_val = region_data[pollutant].std()
        print(f"{label}: {mean_val:.2f} ± {std_val:.2f} μg/m³")


def plot_seasonal_patterns(df, region_filter="ile_de_france"):
    """Plot seasonal patterns of pollutants"""
    
    region_data = df[df['region'] == region_filter].copy()
    
    if region_data.empty:
        print(f"No data found for region: {region_filter}")
        return
    
    # Add month column for seasonal analysis
    region_data['month'] = region_data['week_start'].dt.month
    
    # Create seasonal plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    pollutants = ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone']
    pollutant_labels = ['PM10', 'PM2.5', 'NO₂', 'O₃']
    
    for pollutant, label in zip(pollutants, pollutant_labels):
        monthly_avg = region_data.groupby('month')[pollutant].mean()
        ax.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, label=label)
    
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Concentration (μg/m³)')
    ax.set_title(f'Seasonal Patterns - {region_filter.replace("_", " ").title()}', 
                fontweight='bold', fontsize=14)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Fetch weekly aggregated data
    df = fetch_air_quality_openmeteo(aggregate_weekly=True)
    
    if not df.empty:
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nLast 5 rows:")
        print(df.tail())
        print("\nData info:")
        print(df.info())
        print("\nColumn names:", df.columns.tolist())
        
        # Show weekly averages for recent dates
        if 'week_start' in df.columns:
            print(f"\nSample of recent weekly averages:")
            recent = df[df['week_start'] >= pd.Timestamp.now() - pd.Timedelta(weeks=4)]
            print(recent[['week_start', 'region', 'pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone']].head(10))
            
            # Show weekly distribution
            print(f"\nWeekly records per region:")
            print(df.groupby('region')['week_start'].count())
            
        # Create plots for Île-de-France
        print("\n" + "="*50)
        print("GENERATING POLLUTANT PLOTS FOR ÎLE-DE-FRANCE")
        print("="*50)
        
        # Time series plot
        plot_pollutant_trends(df, region_filter="ile_de_france")
        
        # Seasonal patterns plot
        plot_seasonal_patterns(df, region_filter="ile_de_france")