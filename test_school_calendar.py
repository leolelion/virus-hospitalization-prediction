#!/usr/bin/env python3
"""
Test script to verify the school calendar integration in the pipeline
"""

from grippe_data_pipeline import build_dataset
import pandas as pd

def test_school_calendar_integration():
    print("=" * 60)
    print("TESTING SCHOOL CALENDAR INTEGRATION")
    print("=" * 60)
    
    # Build dataset for Île-de-France (Zone C)
    df = build_dataset('ile_de_france')
    
    # Check school calendar columns
    print(f"\n📊 School Calendar Statistics:")
    print(f"Total weeks: {len(df)}")
    print(f"Vacation weeks: {df['is_vacation_week'].sum()}")
    print(f"Back-to-school weeks: {df['is_back_to_school'].sum()}")
    print(f"Regular school weeks: {len(df) - df['is_vacation_week'].sum() - df['is_back_to_school'].sum()}")
    
    # Show recent school calendar data
    print(f"\n📅 Recent School Calendar (last 10 weeks):")
    recent_calendar = df[['date_complet', 'is_vacation_week', 'is_back_to_school']].tail(10)
    recent_calendar['week_type'] = recent_calendar.apply(lambda row: 
        'Vacation' if row['is_vacation_week'] else 
        'Back-to-School' if row['is_back_to_school'] else 
        'Regular School', axis=1)
    print(recent_calendar[['date_complet', 'week_type']])
    
    # Check for upcoming vacation periods
    print(f"\n🏖️ Vacation Periods in 2025:")
    vacation_weeks = df[df['is_vacation_week'] & (df['date_complet'].dt.year == 2025)]
    if not vacation_weeks.empty:
        # Group consecutive vacation weeks
        vacation_weeks = vacation_weeks.copy()
        vacation_weeks['date_str'] = vacation_weeks['date_complet'].dt.strftime('%Y-%m-%d')
        
        # Find vacation periods (consecutive weeks)
        vacation_periods = []
        current_period = []
        
        for date in vacation_weeks['date_complet']:
            if not current_period or (date - current_period[-1]).days <= 7:
                current_period.append(date)
            else:
                if current_period:
                    vacation_periods.append((current_period[0], current_period[-1]))
                current_period = [date]
        
        if current_period:
            vacation_periods.append((current_period[0], current_period[-1]))
        
        for start, end in vacation_periods:
            weeks = len(pd.date_range(start, end, freq='W-MON'))
            print(f"  {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({weeks} weeks)")
    
    # Verify correlation potential with hospitalization data
    print(f"\n🔍 School Calendar vs Hospitalization Analysis:")
    
    # Filter out missing hospitalization data
    complete_data = df.dropna(subset=['taux_hospit_grippe_sau'])
    
    if not complete_data.empty:
        avg_hosp_vacation = complete_data[complete_data['is_vacation_week']]['taux_hospit_grippe_sau'].mean()
        avg_hosp_school = complete_data[~complete_data['is_vacation_week'] & ~complete_data['is_back_to_school']]['taux_hospit_grippe_sau'].mean()
        avg_hosp_back_to_school = complete_data[complete_data['is_back_to_school']]['taux_hospit_grippe_sau'].mean()
        
        print(f"  Average hospitalization during vacation weeks: {avg_hosp_vacation:.1f}")
        print(f"  Average hospitalization during regular school weeks: {avg_hosp_school:.1f}")
        print(f"  Average hospitalization during back-to-school weeks: {avg_hosp_back_to_school:.1f}")
        
        # Calculate differences
        vacation_diff = ((avg_hosp_vacation - avg_hosp_school) / avg_hosp_school * 100) if avg_hosp_school > 0 else 0
        back_to_school_diff = ((avg_hosp_back_to_school - avg_hosp_school) / avg_hosp_school * 100) if avg_hosp_school > 0 else 0
        
        print(f"\n📈 Relative Differences:")
        print(f"  Vacation weeks vs regular school: {vacation_diff:+.1f}%")
        print(f"  Back-to-school weeks vs regular school: {back_to_school_diff:+.1f}%")

if __name__ == "__main__":
    test_school_calendar_integration()
    print(f"\n✅ School calendar integration test completed!")
