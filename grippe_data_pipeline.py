import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from joblib import Memory
import os
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import time

# Setup caching
cache_dir = os.path.join(os.getcwd(), "cache")
memory = Memory(cache_dir, verbose=0)

load_dotenv()

# Constants
START_DATE = pd.Timestamp("2020-01-01").strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")
DISEASE = "grippe"
AIRPARIF_API_KEY = os.getenv("AIRPARIF_API_KEY")

# French regions with their coordinates and health data region names
FRENCH_REGIONS = {
    "ile_de_france": {
        "name": "Île-de-France",
        "health_region": "Île-de-France",
        "coordinates": (48.8566, 2.3522),  # Paris coordinates
        "description": "Paris metropolitan area"
    },
    "auvergne_rhone_alpes": {
        "name": "Auvergne-Rhône-Alpes", 
        "health_region": "Auvergne-Rhône-Alpes",
        "coordinates": (45.7640, 4.8357),  # Lyon coordinates
        "description": "Lyon metropolitan area"
    },
    "provence_alpes_cote_azur": {
        "name": "Provence-Alpes-Côte d'Azur",
        "health_region": "Provence-Alpes-Côte d'Azur",
        "coordinates": (43.2965, 5.3698),  # Marseille coordinates
        "description": "Marseille metropolitan area"
    },
    "nouvelle_aquitaine": {
        "name": "Nouvelle-Aquitaine",
        "health_region": "Nouvelle-Aquitaine", 
        "coordinates": (44.8378, -0.5792),  # Bordeaux coordinates
        "description": "Bordeaux metropolitan area"
    },
    "occitanie": {
        "name": "Occitanie",
        "health_region": "Occitanie",
        "coordinates": (43.6047, 1.4442),  # Toulouse coordinates
        "description": "Toulouse metropolitan area"
    },
    "hauts_de_france": {
        "name": "Hauts-de-France",
        "health_region": "Hauts-de-France",
        "coordinates": (50.6292, 3.0573),  # Lille coordinates
        "description": "Lille metropolitan area"
    }
}

# Step 1: Weekly time index
def create_weekly_index(start=START_DATE, end=END_DATE):
    return pd.DataFrame({"date_complet": pd.date_range(start, end, freq="W-MON")})

# Step 2: Hospitalization data
@memory.cache
def fetch_hospitalization_data(region_key, batch_size=100, disease=DISEASE):
    """Fetch hospitalization data for a specific region"""
    region_info = FRENCH_REGIONS[region_key]
    health_region = region_info["health_region"]
    
    print(f"Fetching hospitalization data for {health_region}...")
    
    try:
        url = "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/grippe-passages-urgences-et-actes-sos-medecin_reg/records"
        all_records = []
        offset = 0
        
        while True:
            params = {
                "refine": [f"reglib:{health_region}", "sursaud_cl_age_gene:Tous âges"],
                "limit": batch_size,
                "offset": offset,
                "sort": "semaine",
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                break
                
            all_records.extend(results)
            print(f"  Fetched {len(results)} records (total {len(all_records)})")
            offset += batch_size
            
            # Safety limit to prevent infinite loops
            if offset > 10000:
                print(f"  ⚠️  Reached safety limit, stopping at {len(all_records)} records")
                break
        
        if not all_records:
            print(f"  ⚠️  No hospitalization data found for {health_region}")
            return create_empty_disease_dataframe()
            
        return clean_disease_data(pd.DataFrame(all_records), START_YEAR=2020, disease=disease)
        
    except Exception as e:
        print(f"  ❌ Failed to fetch hospitalization data for {health_region}: {e}")
        print(f"  Using empty DataFrame for {health_region}")
        return create_empty_disease_dataframe()

def clean_disease_data(df, START_YEAR, disease):
    """Clean disease data with better error handling"""
    if df.empty:
        print(f"  ⚠️  No data found - creating empty DataFrame")
        return create_empty_disease_dataframe()
    
    print(f"  Raw data columns: {list(df.columns)}")
    
    # Keep relevant columns (based on your keys)
    expected_cols = [
        "date_complet",
        "semaine", 
        f"taux_passages_{disease}_sau",
        f"taux_hospit_{disease}_sau",
        f"taux_actes_{disease}_sos"
    ]
    
    # Check which columns actually exist
    available_cols = [col for col in expected_cols if col in df.columns]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        print(f"  ⚠️  Missing columns: {missing_cols}")
    
    if not available_cols:
        print(f"  ❌ No expected columns found - creating empty DataFrame")
        return create_empty_disease_dataframe()
    
    # Use only available columns
    df = df[available_cols].copy()

    # Convert to appropriate types
    if "date_complet" in df.columns:
        df["date_complet"] = pd.to_datetime(df["date_complet"], errors="coerce")
    
    for col in available_cols[2:]:  # Skip date_complet and semaine
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort chronologically if date column exists
    if "date_complet" in df.columns:
        df = df.sort_values("date_complet")
        df = df[df["date_complet"] >= pd.Timestamp(f"{START_YEAR}-01-01")]
    
    print(f"  ✅ Cleaned data: {len(df)} rows, columns: {list(df.columns)}")
    return df

def create_empty_disease_dataframe():
    """Create empty DataFrame with expected disease data structure"""
    dates = pd.date_range(START_DATE, END_DATE, freq="W-MON")
    return pd.DataFrame({
        "date_complet": dates,
        "semaine": ["2020-W01"] * len(dates),  # Placeholder
        "taux_passages_grippe_sau": [np.nan] * len(dates),
        "taux_hospit_grippe_sau": [np.nan] * len(dates),
        "taux_actes_grippe_sos": [np.nan] * len(dates)
    })

# Step 3: Weather data
@memory.cache
def fetch_weather_data(region_key):
    """Fetch weather data for a specific region"""
    region_info = FRENCH_REGIONS[region_key]
    latitude, longitude = region_info["coordinates"]
    region_name = region_info["name"]
    
    print(f"Fetching weather data for {region_name} ({latitude}, {longitude})...")
    
    temp_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ["temperature_2m_mean", "relative_humidity_2m_mean"],
        "timezone": "Europe/Paris"
    }

    response = requests.get(temp_url, params=params)
    response.raise_for_status()
    temp_data = response.json()

    temp_df = pd.DataFrame({
        "date_complet": pd.to_datetime(temp_data["daily"]["time"]),
        "temp_mean": temp_data["daily"]["temperature_2m_mean"],
        "humidity_mean": temp_data["daily"]["relative_humidity_2m_mean"]
    })

    temp_df = temp_df.resample("W-MON", on="date_complet").mean().reset_index()
    print(f"  Fetched {len(temp_df)} weeks of weather data")
    return temp_df

# Step 4: Air quality data
@memory.cache
def fetch_air_quality_data(region_key):
    """Fetch weekly air quality data for a specific region using Open-Meteo API"""
    import requests
    
    region_info = FRENCH_REGIONS[region_key]
    latitude, longitude = region_info["coordinates"]
    region_name = region_info["name"]
    
    print(f"Fetching air quality data for {region_name} ({latitude}, {longitude})...")
    
    pollutants = ["pm10", "pm2_5", "nitrogen_dioxide", "ozone"]
    base_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(pollutants)
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data or "time" not in data["hourly"]:
            print("  No air quality data available, using mock data")
            return create_mock_air_quality_data(region_key)
            
        # Create DataFrame from hourly data
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        
        # Aggregate to weekly data (Monday to Sunday)
        df_indexed = df.set_index("time")
        weekly_df = df_indexed.resample('W-MON').agg({
            col: 'mean' for col in pollutants if col in df_indexed.columns
        }).reset_index()
        
        # Rename columns to match pipeline expectations and time column
        weekly_df = weekly_df.rename(columns={
            'time': 'date_complet',
            'pm2_5': 'pm25',  # Rename to match your pipeline
            'nitrogen_dioxide': 'no2',  # Rename to match your pipeline  
            'ozone': 'o3'  # Rename to match your pipeline
        })
        
        # Add AQI calculation (simplified European AQI)
        weekly_df['aqi'] = calculate_simplified_aqi(weekly_df)
        
        print(f"  Fetched {len(weekly_df)} weekly air quality records")
        return weekly_df
        
    except Exception as e:
        print(f"  Air quality API failed: {e}, using mock data")
        return create_mock_air_quality_data(region_key)

def create_mock_air_quality_data(region_key):
    """Create mock air quality data with seasonal patterns for a specific region"""
    dates = pd.date_range(START_DATE, END_DATE, freq="W-MON")
    region_info = FRENCH_REGIONS[region_key]
    
    # Use region as seed for different patterns per region
    region_seed = hash(region_key) % 1000
    np.random.seed(region_seed)
    
    # Seasonal patterns for air quality
    weeks = len(dates)
    seasonal_factor = np.sin(2 * np.pi * np.arange(weeks) / 52)
    
    # Regional variations (coastal vs inland, industrial vs residential)
    regional_multiplier = {
        "ile_de_france": 1.2,      # Higher pollution in Paris
        "auvergne_rhone_alpes": 0.9,  # Mountain region, cleaner
        "provence_alpes_cote_azur": 1.0,  # Mediterranean 
        "nouvelle_aquitaine": 0.8,  # Atlantic coast, cleaner
        "occitanie": 0.9,          # Mixed inland/coastal
        "hauts_de_france": 1.1     # Industrial region
    }.get(region_key, 1.0)
    
    mock_data = pd.DataFrame({
        "date_complet": dates,
        "pm25": (15 + 8 * seasonal_factor) * regional_multiplier + np.random.normal(0, 3, weeks),
        "pm10": (25 + 12 * seasonal_factor) * regional_multiplier + np.random.normal(0, 5, weeks), 
        "no2": (30 + 15 * seasonal_factor) * regional_multiplier + np.random.normal(0, 8, weeks),
        "o3": (60 - 20 * seasonal_factor) + np.random.normal(0, 10, weeks),
    })
    
    # Ensure positive values
    for col in ["pm25", "pm10", "no2", "o3"]:
        mock_data[col] = np.maximum(mock_data[col], 1)
    
    # Add AQI calculation
    mock_data['aqi'] = calculate_simplified_aqi(mock_data)
    
    region_name = region_info["name"]
    print(f"  Created {len(mock_data)} weeks of mock air quality data for {region_name}")
    return mock_data

def calculate_simplified_aqi(df):
    """Calculate simplified European AQI based on pollutant levels"""
    # Simplified European AQI calculation (0-100+ scale)
    pm25_aqi = df['pm25'] / 25 * 50  # PM2.5: 25 μg/m³ = 50 AQI
    pm10_aqi = df['pm10'] / 50 * 50  # PM10: 50 μg/m³ = 50 AQI  
    no2_aqi = df['no2'] / 40 * 50    # NO2: 40 μg/m³ = 50 AQI
    o3_aqi = df['o3'] / 120 * 50     # O3: 120 μg/m³ = 50 AQI
    
    # Take maximum of all pollutant AQIs
    return pd.DataFrame([pm25_aqi, pm10_aqi, no2_aqi, o3_aqi]).max()

# Step 5: School calendar
def fetch_school_calendar(region_key):
    """Fetch real French school calendar data for a specific region"""
    from vacation.school_calendar import fetch_all_vacations, expand_vacations_to_weeks, get_region_weekly_vacation
    
    region_info = FRENCH_REGIONS[region_key]
    region_name = region_info["name"]
    
    # Map region keys to school calendar region names (must match vacation/school_calendar.py)
    region_mapping = {
        "ile_de_france": "ile_de_france",
        "auvergne_rhone_alpes": "auvergne_rhone_alpes", 
        "provence_alpes_cote_azur": "provence_alpes_cote_d_azur",  # Note: 'd' not 'de'
        "nouvelle_aquitaine": "nouvelle_aquitaine",
        "occitanie": "occitanie",
        "hauts_de_france": "haut_de_france"  # Note: 'haut' not 'hauts'
    }
    
    school_region = region_mapping.get(region_key)
    
    if not school_region:
        print(f"  ⚠️  No school calendar mapping for {region_name}, using mock data")
        return create_mock_school_calendar()
    
    try:
        print(f"Fetching school calendar for {region_name}...")
        
        # Fetch all vacation data (cached by imports)
        vacations_df = fetch_all_vacations()
        
        # Expand to weekly format
        weekly_vac_df = expand_vacations_to_weeks(vacations_df, START_DATE, END_DATE)
        
        # Get region-specific data
        region_calendar = get_region_weekly_vacation(school_region, weekly_vac_df)
        
        # Rename columns to match pipeline expectations
        region_calendar = region_calendar.rename(columns={'week_start': 'date_complet'})
        
        # Fill missing weeks with False values
        all_weeks = pd.DataFrame({"date_complet": pd.date_range(START_DATE, END_DATE, freq="W-MON")})
        
        # Merge and fill missing values
        full_calendar = all_weeks.merge(region_calendar[['date_complet', 'is_vacation_week', 'is_back_to_school']], 
                                      on='date_complet', how='left')
        full_calendar['is_vacation_week'] = full_calendar['is_vacation_week'].fillna(False)
        full_calendar['is_back_to_school'] = full_calendar['is_back_to_school'].fillna(False)
        
        print(f"  Fetched {len(full_calendar)} weeks of school calendar data")
        print(f"  Vacation weeks: {full_calendar['is_vacation_week'].sum()}")
        print(f"  Back-to-school weeks: {full_calendar['is_back_to_school'].sum()}")
        
        return full_calendar
        
    except Exception as e:
        print(f"  ❌ Failed to fetch school calendar for {region_name}: {e}")
        print(f"  Using mock school calendar data")
        return create_mock_school_calendar()

def create_mock_school_calendar():
    """Create mock school calendar data with realistic patterns"""
    dates = pd.date_range(START_DATE, END_DATE, freq="W-MON")
    
    # Create realistic vacation pattern (summer, winter, spring breaks)
    vacation_weeks = []
    back_to_school_weeks = []
    
    for year in range(2020, 2027):
        # Summer vacation (July-August, ~8 weeks)
        summer_start = pd.Timestamp(f"{year}-07-01")
        summer_end = pd.Timestamp(f"{year}-08-31")
        summer_weeks = pd.date_range(summer_start, summer_end, freq="W-MON")
        vacation_weeks.extend(summer_weeks)
        if summer_end + pd.Timedelta(days=7) <= dates.max():
            back_to_school_weeks.append(summer_end + pd.Timedelta(days=7) - pd.to_timedelta(summer_end.weekday(), unit='d'))
        
        # Winter break (2 weeks around Christmas/New Year)
        winter_start = pd.Timestamp(f"{year}-12-23")
        winter_end = pd.Timestamp(f"{year+1}-01-06")
        winter_weeks = pd.date_range(winter_start, winter_end, freq="W-MON")
        vacation_weeks.extend(winter_weeks)
        if winter_end + pd.Timedelta(days=7) <= dates.max():
            back_to_school_weeks.append(winter_end + pd.Timedelta(days=7) - pd.to_timedelta(winter_end.weekday(), unit='d'))
        
        # Spring break (2 weeks in April)
        spring_start = pd.Timestamp(f"{year}-04-15")
        spring_end = pd.Timestamp(f"{year}-04-29")
        spring_weeks = pd.date_range(spring_start, spring_end, freq="W-MON")
        vacation_weeks.extend(spring_weeks)
        if spring_end + pd.Timedelta(days=7) <= dates.max():
            back_to_school_weeks.append(spring_end + pd.Timedelta(days=7) - pd.to_timedelta(spring_end.weekday(), unit='d'))
    
    # Create DataFrame
    df = pd.DataFrame({"date_complet": dates})
    df["is_vacation_week"] = df["date_complet"].isin(vacation_weeks)
    df["is_back_to_school"] = df["date_complet"].isin(back_to_school_weeks)
    
    return df

# Step 6: Temporal features
def add_temporal_features(df):
    df["week_of_year"] = df["date_complet"].dt.isocalendar().week
    df["month"] = df["date_complet"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

# Step 7: Lagged features
def add_lagged_features(df, target_col):
    for lag in range(1, 5):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df

# Step 8: Build dataset for a single region
def build_region_dataset(region_key):
    """Build dataset for a specific region"""
    region_info = FRENCH_REGIONS[region_key]
    region_name = region_info["name"]
    
    print(f"\n{'='*60}")
    print(f"BUILDING DATASET FOR {region_name.upper()}")
    print(f"{'='*60}")
    
    base = create_weekly_index()
    grippe = fetch_hospitalization_data(region_key, disease=DISEASE)
    weather = fetch_weather_data(region_key)
    air = fetch_air_quality_data(region_key)
    school = fetch_school_calendar(region_key)

    # Merge all data sources
    df = base.merge(grippe, on="date_complet", how="left")
    df = df.merge(weather, on="date_complet", how="left")
    df = df.merge(air, on="date_complet", how="left")
    df = df.merge(school, on="date_complet", how="left")

    # Add features
    df = add_temporal_features(df)
    df = add_lagged_features(df, "taux_hospit_grippe_sau")
    
    # Add region identifier
    df['region'] = region_key
    df['region_name'] = region_name
    
    # Clean up columns
    columns_to_drop = ["semaine", "week_of_year", "month"]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    print(f"✅ Dataset completed for {region_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Step 9: Build datasets for multiple regions
def build_multi_region_datasets(region_keys=None):
    """Build datasets for multiple regions"""
    if region_keys is None:
        region_keys = list(FRENCH_REGIONS.keys())
    
    print(f"Building datasets for {len(region_keys)} regions: {', '.join([FRENCH_REGIONS[k]['name'] for k in region_keys])}")
    
    region_datasets = {}
    
    for region_key in region_keys:
        try:
            df = build_region_dataset(region_key)
            region_datasets[region_key] = df
        except Exception as e:
            print(f"❌ Failed to build dataset for {FRENCH_REGIONS[region_key]['name']}: {e}")
            continue
    
    return region_datasets

# Legacy function for backward compatibility
def build_dataset(region_key="ile_de_france"):
    """Build dataset for a single region (default: Île-de-France)"""
    return build_region_dataset(region_key)

def plot_correlation_heatmap(final_df):
    # Select relevant columns
    cols = [
        "taux_passages_grippe_sau",
        "taux_hospit_grippe_sau",
        "taux_actes_grippe_sos",
        "temp_mean",
        "humidity_mean",
        "pm25", "pm10", "no2", "o3", "aqi",
        "is_vacation_week", "is_back_to_school",
        "taux_hospit_grippe_sau_lag_1",
        "taux_hospit_grippe_sau_lag_2",
        "taux_hospit_grippe_sau_lag_3",
        "taux_hospit_grippe_sau_lag_4"
    ]

    heatmap_df = final_df[cols]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        heatmap_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Correlation Heatmap of Epidemic Forecasting Features")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    
    # Option 1: Build dataset for a single region (fast demo)
    print("=" * 80)
    print("OPTION 1: SINGLE REGION DEMO (ÎLE-DE-FRANCE)")  
    print("=" * 80)
    
    single_region_df = build_dataset("ile_de_france")
    
    print(f"\nDataset shape: {single_region_df.shape}")
    print(f"Date range: {single_region_df['date_complet'].min()} to {single_region_df['date_complet'].max()}")
    print(f"Total weeks covered: {len(single_region_df)}")
    
    print("\nMissing values per column:")
    missing = single_region_df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
    
    print("\nTarget variable statistics (taux_hospit_grippe_sau):")
    print(single_region_df['taux_hospit_grippe_sau'].describe())
    
    print(f"\nFirst 3 rows:")
    print(single_region_df.tailn(20))
    
    # Option 2: Build datasets for multiple regions (enable for multi-region demo)
    
    print("\n" + "=" * 80)
    print("OPTION 2: MULTI-REGION DATASETS")  
    print("=" * 80)
    
    # Select a few regions for demo (add more as needed)
    demo_regions = ["ile_de_france", "auvergne_rhone_alpes", "provence_alpes_cote_azur"]
    
    region_datasets = build_multi_region_datasets(demo_regions)
    
    print(f"\n📊 MULTI-REGION SUMMARY:")
    print("-" * 50)
    for region_key, df in region_datasets.items():
        region_name = FRENCH_REGIONS[region_key]['name']
        missing_target = df['taux_hospit_grippe_sau'].isnull().sum()
        print(f"{region_name:25} | {df.shape[0]:3} weeks | {missing_target:2} missing targets")
    
    # Example: Compare regions
    # print(f"\n🔍 REGIONAL COMPARISON:")
    # print("-" * 50)
    # for region_key, df in region_datasets.items():
    #     region_name = FRENCH_REGIONS[region_key]['name']
    #     if not df['taux_hospit_grippe_sau'].isnull().all():
    #         avg_hosp = df['taux_hospit_grippe_sau'].mean()
    #         avg_temp = df['temp_mean'].mean()
    #         avg_pm25 = df['pm25'].mean()
    #         print(f"{region_name:25} | Hosp: {avg_hosp:6.1f} | Temp: {avg_temp:5.1f}°C | PM2.5: {avg_pm25:5.1f}")
    
    # print("\n" + "=" * 80)
    # print("AVAILABLE REGIONS")
    # print("=" * 80)
    # print("You can build datasets for any of these regions:")
    # for key, info in FRENCH_REGIONS.items():
    #     lat, lon = info['coordinates']
    #     print(f"• {key:20} | {info['name']:25} | ({lat:6.2f}, {lon:6.2f}) | {info['description']}")
    
    print(f"\n💡 USAGE EXAMPLES:")
    print("# Single region:")
    print("df = build_dataset('auvergne_rhone_alpes')")
    print("\n# Multiple regions:")
    print("datasets = build_multi_region_datasets(['ile_de_france', 'auvergne_rhone_alpes'])")
    print("\n# All regions:")
    print("all_datasets = build_multi_region_datasets()")
    
    print("\n✅ Pipeline setup completed successfully!")
