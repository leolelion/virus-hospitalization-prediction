#!/usr/bin/env python3
"""
Demo script showing how to use the multi-region virus hospitalization prediction pipeline
"""

from grippe_data_pipeline import (
    build_dataset, 
    build_multi_region_datasets, 
    FRENCH_REGIONS,
    plot_correlation_heatmap
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def demo_single_region():
    """Demo: Build dataset for a single region"""
    print("=" * 60)
    print("DEMO 1: SINGLE REGION (AUVERGNE-RHÔNE-ALPES)")
    print("=" * 60)
    
    # Build dataset for Lyon region
    df_lyon = build_dataset('auvergne_rhone_alpes')
    
    print(f"\n📊 Dataset Summary:")
    print(f"Shape: {df_lyon.shape}")
    print(f"Region: {df_lyon['region_name'].iloc[0]}")
    print(f"Date range: {df_lyon['date_complet'].min().date()} to {df_lyon['date_complet'].max().date()}")
    
    # Show some statistics
    if not df_lyon['taux_hospit_grippe_sau'].isnull().all():
        print(f"Average hospitalization rate: {df_lyon['taux_hospit_grippe_sau'].mean():.1f}")
        print(f"Average temperature: {df_lyon['temp_mean'].mean():.1f}°C")
        print(f"Average PM2.5: {df_lyon['pm25'].mean():.1f} μg/m³")


def demo_multi_region():
    """Demo: Build datasets for multiple regions"""
    print("\n" + "=" * 60)
    print("DEMO 2: MULTIPLE REGIONS")
    print("=" * 60)
    
    # Select 3 regions for comparison
    regions_to_compare = ['ile_de_france', 'auvergne_rhone_alpes', 'provence_alpes_cote_azur']
    
    # Build datasets for multiple regions
    datasets = build_multi_region_datasets(regions_to_compare)
    
    print(f"\n📊 Multi-Region Summary:")
    print(f"Built datasets for {len(datasets)} regions")
    
    # Compare regions
    comparison_data = []
    for region_key, df in datasets.items():
        region_name = FRENCH_REGIONS[region_key]['name']
        
        # Calculate statistics if data is available
        if not df['taux_hospit_grippe_sau'].isnull().all():
            stats = {
                'Region': region_name,
                'Weeks': len(df),
                'Avg_Hospitalization': df['taux_hospit_grippe_sau'].mean(),
                'Avg_Temperature': df['temp_mean'].mean(),
                'Avg_PM25': df['pm25'].mean(),
                'Avg_NO2': df['no2'].mean(),
                'Missing_Targets': df['taux_hospit_grippe_sau'].isnull().sum()
            }
            comparison_data.append(stats)
    
    # Create comparison DataFrame
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n🔍 Regional Comparison:")
        print(comparison_df.round(1))


def demo_plotting():
    """Demo: Create plots comparing regions"""
    print("\n" + "=" * 60)
    print("DEMO 3: REGIONAL COMPARISON PLOTS")
    print("=" * 60)
    
    # Build datasets for comparison
    regions_to_plot = ['ile_de_france', 'auvergne_rhone_alpes']
    datasets = build_multi_region_datasets(regions_to_plot)
    
    # Combine data for plotting
    combined_data = []
    for region_key, df in datasets.items():
        df_copy = df.copy()
        df_copy['region_key'] = region_key
        combined_data.append(df_copy)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Regional Comparison: Health and Environmental Data', fontsize=16)
        
        # Plot 1: Hospitalization rates over time
        ax1 = axes[0, 0]
        for region_key in regions_to_plot:
            region_data = combined_df[combined_df['region_key'] == region_key]
            region_name = FRENCH_REGIONS[region_key]['name']
            ax1.plot(region_data['date_complet'], region_data['taux_hospit_grippe_sau'], 
                    label=region_name, alpha=0.7, linewidth=2)
        ax1.set_title('Hospitalization Rates Over Time')
        ax1.set_ylabel('Hospitalization Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Air quality comparison
        ax2 = axes[0, 1] 
        sns.boxplot(data=combined_df, x='region_name', y='pm25', ax=ax2)
        ax2.set_title('PM2.5 Distribution by Region')
        ax2.set_ylabel('PM2.5 (μg/m³)')
        
        # Plot 3: Temperature comparison
        ax3 = axes[1, 0]
        sns.boxplot(data=combined_df, x='region_name', y='temp_mean', ax=ax3)
        ax3.set_title('Temperature Distribution by Region')
        ax3.set_ylabel('Temperature (°C)')
        
        # Plot 4: Correlation between temp and hospitalizations
        ax4 = axes[1, 1]
        for region_key in regions_to_plot:
            region_data = combined_df[combined_df['region_key'] == region_key]
            region_name = FRENCH_REGIONS[region_key]['name']
            ax4.scatter(region_data['temp_mean'], region_data['taux_hospit_grippe_sau'], 
                       label=region_name, alpha=0.6)
        ax4.set_title('Temperature vs Hospitalization')
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Hospitalization Rate')
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("📈 Regional comparison plots generated!")


if __name__ == "__main__":
    print("🚀 MULTI-REGION PIPELINE DEMO")
    print("=" * 80)
    
    # Run demos
    demo_single_region()
    demo_multi_region() 
    
    # Uncomment to generate plots
    # demo_plotting()
    
    print("\n" + "=" * 80)
    print("✅ Demo completed!")
    print("\n💡 Tips:")
    print("- Use build_dataset('region_key') for single region")
    print("- Use build_multi_region_datasets(['region1', 'region2']) for multiple regions")
    print("- All data is cached for faster subsequent runs")
    print("- Each region has its own coordinates for weather and air quality data")
    print("- Health data is automatically matched to the correct administrative region")
