import pandas as pd
import numpy as np
import requests
from datetime import datetime
from joblib import Memory
import os
import seaborn as sns
import matplotlib.pyplot as plt


# Setup caching
cache_dir = os.path.join(os.getcwd(), "cache")
memory = Memory(cache_dir, verbose=0)

# Constants
START_DATE = "2022-01-01"
START_DATE = pd.Timestamp("2022-01-01").strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")
REGION = "Île-de-France"
DISEASE = "grippe"

# Step 1: Weekly time index
def create_weekly_index(start=START_DATE, end=END_DATE):
    return pd.DataFrame({"date_complet": pd.date_range(start, end, freq="W-MON")})

# Step 2: Hospitalization data
@memory.cache
def fetch_hospitalization_data(region=REGION, batch_size=100, disease=DISEASE):
    url = "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/grippe-passages-urgences-et-actes-sos-medecin_reg/records"
    all_records = []
    offset = 0
    while True:
        params = {
            "refine": [f"reglib:{region}", "sursaud_cl_age_gene:Tous âges"],
            "limit": batch_size,
            "offset": offset,
            "sort": "semaine",
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            break
        all_records.extend(results)
        print(f"Fetched {len(results)} records (total {len(all_records)})")
        offset += batch_size
    return clean_disease_data(pd.DataFrame(all_records), START_YEAR=2022, disease=disease)

def clean_disease_data(df, START_YEAR, disease):
    # Keep relevant columns (based on your keys)
    cols = [
        "date_complet",
        "semaine",
        f"taux_passages_{disease}_sau",
        f"taux_hospit_{disease}_sau",
        f"taux_actes_{disease}_sos"
    ]
    df = df[cols]

    # Convert to appropriate types
    df["date_complet"] = pd.to_datetime(df["date_complet"], errors="coerce")
    for c in cols[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort chronologically
    df = df.sort_values("date_complet")

    df = df[df["date_complet"] >= pd.Timestamp(f"{START_YEAR}-01-01")]
    return df



# Step 3: Weather data
@memory.cache
def fetch_weather_data(latitude=48.8566, longitude=2.3522):
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
    return temp_df

@memory.cache
def fetch_air_quality_data():
    weeks = pd.date_range(START_DATE, END_DATE, freq="W-MON")
    np.random.seed(42)  # for reproducibility

    return pd.DataFrame({
        "date_complet": weeks,
        "pm25": np.random.normal(loc=15, scale=5, size=len(weeks)).clip(0),
        "pm10": np.random.normal(loc=25, scale=7, size=len(weeks)).clip(0),
        "no2": np.random.normal(loc=20, scale=6, size=len(weeks)).clip(0),
        "o3": np.random.normal(loc=30, scale=8, size=len(weeks)).clip(0),
        "aqi": np.random.normal(loc=50, scale=15, size=len(weeks)).clip(0)
    })


# Step 4: Air quality data
# @memory.cache
# def fetch_air_quality_data(api_key, latitude=48.8566, longitude=2.3522):
#     air_url = "https://api.airparif.asso.fr/airquality/history"

#     params = {
#         "apikey": api_key,
#         "start_date": START_DATE,
#         "end_date": END_DATE,
#         "lat": latitude,
#         "lon": longitude,
#         "pollutants": ["PM2.5", "PM10", "NO2", "O3", "AQI"],
#         "format": "json"
#     }

#     response = requests.get(air_url, params=params)
#     response.raise_for_status()
#     air_data = response.json()

#     df = pd.DataFrame({
#         "date_complet": pd.to_datetime([entry["date"] for entry in air_data]),
#         "pm25": [entry["PM2.5"] for entry in air_data],
#         "pm10": [entry["PM10"] for entry in air_data],
#         "no2": [entry["NO2"] for entry in air_data],
#         "o3": [entry["O3"] for entry in air_data],
#         "aqi": [entry["AQI"] for entry in air_data]
#     })

#     df = df.resample("W-MON", on="date_complet").mean().reset_index()
#     return df


# Step 5: School calendar
def fetch_school_calendar():
    # Placeholder: Replace with data.gouv.fr school calendar logic
    df = pd.DataFrame({"date_complet": pd.date_range(START_DATE, END_DATE, freq="W-MON")})
    df["is_vacation_week"] = np.random.choice([0, 1], size=len(df))
    df["is_back_to_school"] = np.random.choice([0, 1], size=len(df))
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


# Step 8: Merge all features
def build_dataset():
    base = create_weekly_index()
    grippe = fetch_hospitalization_data(disease=DISEASE)
    #flu = fetch_hospitalization_data(disease="Influenza")
    weather = fetch_weather_data()
    #air = fetch_air_quality_data(api_key=AIRPARIF_API_KEY)
    air = fetch_air_quality_data()
    school = fetch_school_calendar()

    df = base.merge(grippe, on="date_complet", how="left")
    #df = df.merge(flu, on="date_complet", how="left")
    df = df.merge(weather, on="date_complet", how="left")
    df = df.merge(air, on="date_complet", how="left")
    df = df.merge(school, on="date_complet", how="left")

    df = add_temporal_features(df)
    df = add_lagged_features(df,"taux_hospit_grippe_sau")
    #df = add_lagged_features(df, "influenza_hospitalizations")
    df.drop(columns=["semaine", "week_of_year", "month"], inplace=True)

    return df

# Run pipeline
final_df = build_dataset()
pd.set_option("display.max_columns", None)

print(len(final_df.columns.to_list()))
print(final_df.shape)
#print(final_df.isnull().sum())  # Count of missing values per column

print(final_df.head(3))




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


plot_correlation_heatmap(final_df)
