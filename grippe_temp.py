import requests
import pandas as pd
import matplotlib.pyplot as plt
import requests
from joblib import Memory
import os


# API base
url = "https://odisse.santepubliquefrance.fr/api/explore/v2.1/catalog/datasets/grippe-passages-urgences-et-actes-sos-medecin_reg/records"
region = "Île-de-France"

# Set up cache directory
cache_dir = os.path.join(os.getcwd(), "cache")
memory = Memory(cache_dir, verbose=0)

@memory.cache
def fetch_grippe_records(region=region, batch_size=100):
    all_records = []
    offset = 0
    while True:
        params = {
            "refine": f"reglib:{region}",
            "limit": batch_size,
            "offset": offset,
            "sort": "semaine"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            break
        all_records.extend(results)
        print(f"Fetched {len(results)} records (total {len(all_records)})")
        offset += batch_size
    return all_records


# ---- Fetch all data ----
data = fetch_grippe_records(region)
df = pd.DataFrame(data)

def clean_disease_data(df, start_year=2022):
    # Keep relevant columns (based on your keys)
    cols = [
        "date_complet",
        "semaine",
        "taux_passages_grippe_sau",
        "taux_hospit_grippe_sau",
        "taux_actes_grippe_sos"
    ]
    df = df[cols]

    # Convert to appropriate types
    df["date_complet"] = pd.to_datetime(df["date_complet"], errors="coerce")
    for c in cols[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort chronologically
    df = df.sort_values("date_complet")

    # Filter by start year
    df = df[df["date_complet"] >= pd.Timestamp(f"{start_year}-01-01")]
    
    return df


def fetch_temperature_data(start_date, end_date):
    temp_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "daily": "temperature_2m_mean",
        "timezone": "Europe/Paris"
    }
    temp_data = requests.get(temp_url, params=params).json()
    temp_df = pd.DataFrame({
        "date_complet": pd.to_datetime(temp_data["daily"]["time"]),
        "temp_mean": temp_data["daily"]["temperature_2m_mean"]
    })
    temp_df = temp_df.resample("W-MON", on="date_complet").mean().reset_index()
    return temp_df

def merge_and_plot(df, temp_df):
    merged = pd.merge_asof(
        df.sort_values("date_complet"),
        temp_df.sort_values("date_complet"),
        on="date_complet"
    )

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(merged["date_complet"], merged["taux_passages_grippe_sau"], label="ER visits (grippe rate)")
    ax1.plot(merged["date_complet"], merged["taux_hospit_grippe_sau"], color="orange", label="Hospitalizations (rate)")
    ax1.plot(merged["date_complet"], merged["taux_actes_grippe_sos"], label="SOS Médecins (rate)")
    ax1.set_ylabel("Rate (%)")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(merged["date_complet"], merged["temp_mean"], color="blue", alpha=0.5, label="Weekly avg. temperature (°C)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.legend(loc="upper right")
    plt.grid(True)
    plt.title("Grippe indicators vs. Weekly Temperature – Île-de-France")
    plt.show()


print(df.columns)

# Clean the data
df_cleaned = clean_disease_data(df)

# Get date range for temperature data
start_date = df_cleaned["date_complet"].min()
end_date = df_cleaned["date_complet"].max()

print(f"Fetching temperature data from {start_date.date()} to {end_date.date()}")

# Fetch temperature data
temp_df = fetch_temperature_data(start_date, end_date)

# Merge and plot the data
merge_and_plot(df_cleaned, temp_df)
