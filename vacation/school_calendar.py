# school_calendar.py
import requests
from ics import Calendar
import pandas as pd
from datetime import timedelta

# Mapping of French regions to school zones
REGION_TO_ZONE = {
    "ile_de_france": "C",
    "auvergne_rhone_alpes": "A",
    "bretagne": "B",
    "bourgogne_franche_comte": "B",
    "centre_val_de_loire": "B",
    "corse": "C",
    "grand_est": "A",
    "haut_de_france": "B",
    "normandie": "B",
    "nouvelle_aquitaine": "B",
    "occitanie": "A",
    "pays_de_la_loire": "B",
    "provence_alpes_cote_d_azur": "C"
}

# ICS URLs per zone
ZONE_ICS_URLS = {
    "A": "https://fr.ftp.opendatasoft.com/openscol/fr-en-calendrier-scolaire/Zone-A.ics",
    "B": "https://fr.ftp.opendatasoft.com/openscol/fr-en-calendrier-scolaire/Zone-B.ics",
    "C": "https://fr.ftp.opendatasoft.com/openscol/fr-en-calendrier-scolaire/Zone-C.ics"
}

def fetch_zone_ics(zone: str) -> str:
    """Fetch ICS content for a given zone."""
    url = ZONE_ICS_URLS[zone]
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text

def parse_vacations_from_ics(ics_content: str, zone: str) -> pd.DataFrame:
    """Parse ICS content into vacation periods."""
    cal = Calendar(ics_content)
    rows = []
    for ev in cal.events:
        start = ev.begin.date()
        end = ev.end.date() - timedelta(days=1)  # ICS end dates are exclusive
        rows.append({
            "zone": zone,
            "vacation": ev.name,
            "date_debut": pd.to_datetime(start),
            "date_fin": pd.to_datetime(end)
        })
    return pd.DataFrame(rows)

def fetch_all_vacations() -> pd.DataFrame:
    """Fetch and parse vacations for all zones."""
    all_vac = []
    for zone in ["A", "B", "C"]:
        print(f"Fetching Zone {zone} ICS...")
        ics_content = fetch_zone_ics(zone)
        vac_df = parse_vacations_from_ics(ics_content, zone)
        all_vac.append(vac_df)
    all_vacations_df = pd.concat(all_vac, ignore_index=True)
    return all_vacations_df

def expand_vacations_to_weeks(vacations_df: pd.DataFrame, start_date="2022-01-01", end_date="2026-12-31") -> pd.DataFrame:
    """
    Expand vacation periods into weekly calendar (week_start = Monday) with boolean flags.
    - A week is considered a vacation week if at least 5 days overlap with vacation.
    - A week is considered 'back to school' if it is the first week immediately after a vacation.
    """
    weekly_rows = []

    for _, row in vacations_df.iterrows():
        vacation_days = pd.date_range(row['date_debut'], row['date_fin'], freq='D')

        # Group days by their week (Monday)
        week_groups = {}
        for day in vacation_days:
            week_start = day - pd.to_timedelta(day.weekday(), unit='d')
            week_groups.setdefault(week_start, []).append(day)

        # Add vacation weeks
        vacation_weeks = sorted(week_groups.keys())
        for i, week_start in enumerate(vacation_weeks):
            if len(week_groups[week_start]) >= 5 and pd.Timestamp(start_date) <= week_start <= pd.Timestamp(end_date):
                weekly_rows.append({
                    "zone": row['zone'],
                    "week_start": week_start,
                    "is_vacation_week": True,
                    "is_back_to_school": False  # default
                })

        # Add 'back to school' week (first week after vacation ends)
        last_vacation_day = row['date_fin']
        back_to_school_week = last_vacation_day + pd.Timedelta(days=1)
        back_to_school_monday = back_to_school_week - pd.to_timedelta(back_to_school_week.weekday(), unit='d')
        if pd.Timestamp(start_date) <= back_to_school_monday <= pd.Timestamp(end_date):
            weekly_rows.append({
                "zone": row['zone'],
                "week_start": back_to_school_monday,
                "is_vacation_week": False,
                "is_back_to_school": True
            })

    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df = weekly_df.sort_values(["zone", "week_start"]).reset_index(drop=True)
    return weekly_df

def get_region_weekly_vacation(region_name: str, weekly_vac_df: pd.DataFrame) -> pd.DataFrame:
    """Return weekly vacation DataFrame for a specific region."""
    zone = REGION_TO_ZONE.get(region_name)
    if zone is None:
        raise ValueError(f"Region {region_name} not mapped to a zone")
    return weekly_vac_df[weekly_vac_df['zone'] == zone].copy()

if __name__ == "__main__":
    print("Fetching all school vacations (Zones A/B/C)...")
    vacations_df = fetch_all_vacations()
    print(f"Total vacation periods fetched: {len(vacations_df)}")

    print("Expanding to weekly calendar...")
    weekly_vac_df = expand_vacations_to_weeks(vacations_df)
    print(f"Total weekly entries: {len(weekly_vac_df)}")

    # Example: fetch weekly vacation for Ile-de-France
    region_name = "ile_de_france"
    region_weekly = get_region_weekly_vacation(region_name, weekly_vac_df)
    print(f"Weekly vacation for {region_name}:")
    print(region_weekly.tail(50))
