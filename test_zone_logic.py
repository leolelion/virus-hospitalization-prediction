#!/usr/bin/env python3
"""
Test script to verify that different regions get different school zones and vacation schedules
"""

from grippe_data_pipeline import build_dataset, FRENCH_REGIONS
from vacation.school_calendar import REGION_TO_ZONE
import pandas as pd

def test_zone_based_vacations():
    print("=" * 70)
    print("TESTING ZONE-BASED SCHOOL VACATION LOGIC")
    print("=" * 70)
    
    # Test regions from different zones
    test_regions = {
        "ile_de_france": "C",        # Zone C - Paris region
        "auvergne_rhone_alpes": "A", # Zone A - Lyon region  
        "nouvelle_aquitaine": "B"    # Zone B - Bordeaux region
    }
    
    print(f"\n📍 Expected Zone Mapping:")
    for region_key, expected_zone in test_regions.items():
        region_name = FRENCH_REGIONS[region_key]["name"]
        actual_zone = REGION_TO_ZONE.get(region_key, "Unknown")
        status = "✅" if actual_zone == expected_zone else "❌"
        print(f"  {status} {region_name:25} → Zone {actual_zone} (expected: {expected_zone})")
    
    print(f"\n🏖️ Testing Vacation Schedules by Zone:")
    print("-" * 70)
    
    vacation_data = {}
    
    for region_key, expected_zone in test_regions.items():
        region_name = FRENCH_REGIONS[region_key]["name"]
        print(f"\n🔍 Testing {region_name} (Zone {expected_zone})...")
        
        try:
            # Build dataset for this region
            df = build_dataset(region_key)
            
            # Get vacation weeks for 2025 
            vacation_2025 = df[
                (df['is_vacation_week'] == True) & 
                (df['date_complet'].dt.year == 2025)
            ]['date_complet'].tolist()
            
            vacation_data[region_key] = {
                'zone': expected_zone,
                'region_name': region_name,
                'vacation_weeks_2025': vacation_2025
            }
            
            print(f"  ✅ Found {len(vacation_2025)} vacation weeks in 2025")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Compare vacation schedules between zones
    print(f"\n📊 VACATION SCHEDULE COMPARISON (2025):")
    print("-" * 70)
    
    # Show first few vacation weeks for each region
    for region_key, data in vacation_data.items():
        print(f"\n🏫 {data['region_name']} (Zone {data['zone']}):")
        if data['vacation_weeks_2025']:
            # Show first 8 vacation weeks
            for i, week in enumerate(data['vacation_weeks_2025'][:8]):
                week_str = week.strftime('%Y-%m-%d')
                print(f"   {i+1:2d}. {week_str}")
            if len(data['vacation_weeks_2025']) > 8:
                print(f"   ... and {len(data['vacation_weeks_2025']) - 8} more weeks")
        else:
            print("   No vacation weeks found")
    
    # Check if different zones have different vacation schedules
    print(f"\n🔍 ZONE DIFFERENCE VERIFICATION:")
    print("-" * 70)
    
    if len(vacation_data) >= 2:
        regions = list(vacation_data.keys())
        region1, region2 = regions[0], regions[1]
        
        vac1 = set(vacation_data[region1]['vacation_weeks_2025'])
        vac2 = set(vacation_data[region2]['vacation_weeks_2025'])
        
        common = vac1 & vac2
        different1 = vac1 - vac2
        different2 = vac2 - vac1
        
        name1 = vacation_data[region1]['region_name']
        name2 = vacation_data[region2]['region_name']
        zone1 = vacation_data[region1]['zone']
        zone2 = vacation_data[region2]['zone']
        
        print(f"Comparing {name1} (Zone {zone1}) vs {name2} (Zone {zone2}):")
        print(f"  Common vacation weeks: {len(common)}")
        print(f"  {name1} unique weeks: {len(different1)}")
        print(f"  {name2} unique weeks: {len(different2)}")
        
        if different1 or different2:
            print(f"  ✅ Zones have different vacation schedules (as expected)")
        else:
            print(f"  ⚠️  Zones have identical vacation schedules (unexpected)")
            
        # Show some different weeks
        if different1:
            print(f"\n  📅 Some weeks unique to {name1}:")
            for week in sorted(list(different1))[:3]:
                print(f"     {week.strftime('%Y-%m-%d')}")
        
        if different2:
            print(f"\n  📅 Some weeks unique to {name2}:")
            for week in sorted(list(different2))[:3]:
                print(f"     {week.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    test_zone_based_vacations()
    print(f"\n✅ Zone-based vacation logic test completed!")
