#!/usr/bin/env python3
"""
Test script to verify zone-based school calendar differences in the data pipeline
"""

from grippe_data_pipeline import build_multi_region_datasets, FRENCH_REGIONS
import pandas as pd

def test_pipeline_zone_differences():
    """Test that different regions have different vacation patterns due to zones"""
    print("🧪 TESTING ZONE-BASED LOGIC IN DATA PIPELINE")
    print("=" * 60)
    
    # Test 3 regions from different zones
    test_regions = {
        "ile_de_france": "Zone C",  
        "auvergne_rhone_alpes": "Zone A",
        "nouvelle_aquitaine": "Zone B"
    }
    
    print(f"Testing regions from different zones:")
    for region, zone in test_regions.items():
        region_name = FRENCH_REGIONS[region]['name']
        print(f"  • {region_name} ({zone})")
    
    print("\nBuilding datasets...")
    
    # Build datasets for test regions
    datasets = build_multi_region_datasets(list(test_regions.keys()))
    
    print(f"\n📊 VACATION PATTERN ANALYSIS:")
    print("-" * 60)
    
    # Analyze vacation patterns for each region
    vacation_analysis = {}
    
    for region_key, df in datasets.items():
        region_name = FRENCH_REGIONS[region_key]['name']
        zone = test_regions[region_key]
        
        # Get vacation weeks
        vacation_weeks = df[df['is_vacation_week'] == True]['date_complet'].dt.strftime('%Y-%m-%d').tolist()
        total_vacation_weeks = len(vacation_weeks)
        
        vacation_analysis[region_key] = {
            'name': region_name,
            'zone': zone,
            'vacation_weeks': vacation_weeks,
            'total_vacation_weeks': total_vacation_weeks
        }
        
        print(f"{region_name:25} ({zone}) | {total_vacation_weeks:2} vacation weeks")
    
    # Compare spring break periods (should be different for each zone)
    print(f"\n🌸 SPRING BREAK COMPARISON (April 2024):")
    print("-" * 60)
    
    april_2024_weeks = [
        '2024-04-01', '2024-04-08', '2024-04-15', '2024-04-22', '2024-04-29'
    ]
    
    for region_key, analysis in vacation_analysis.items():
        april_vacations = [week for week in analysis['vacation_weeks'] if week.startswith('2024-04')]
        print(f"{analysis['name']:25} ({analysis['zone']}) | April 2024 vacations: {april_vacations}")
    
    # Check for differences
    print(f"\n🔍 ZONE DIFFERENCE VERIFICATION:")
    print("-" * 60)
    
    all_vacation_sets = [set(analysis['vacation_weeks']) for analysis in vacation_analysis.values()]
    
    if len(set(frozenset(s) for s in all_vacation_sets)) > 1:
        print("✅ SUCCESS: Different regions have different vacation schedules!")
        
        # Find common and unique weeks
        common_weeks = set.intersection(*all_vacation_sets)
        print(f"   • Common vacation weeks across all zones: {len(common_weeks)}")
        
        for region_key, analysis in vacation_analysis.items():
            unique_weeks = set(analysis['vacation_weeks']) - common_weeks
            print(f"   • {analysis['name']} ({analysis['zone']}) unique weeks: {len(unique_weeks)}")
            
    else:
        print("❌ ISSUE: All regions have identical vacation schedules!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_pipeline_zone_differences()
    
    if success:
        print(f"\n🎉 PIPELINE ZONE VALIDATION: PASSED")
        print("The data pipeline correctly implements zone-based school calendars!")
    else:
        print(f"\n❌ PIPELINE ZONE VALIDATION: FAILED")
        print("Zone-based logic may not be working properly in the pipeline.")
