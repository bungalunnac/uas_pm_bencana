import pandas as pd
import os

def preprocess_data(raw_data_path, processed_dir):
    """
    Load raw EMDAT data, aggregate it by Country, Disaster Type, and Year,
    and save processed datasets.
    """
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_data_path}")
        
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load raw dataset
    df = pd.read_csv(raw_data_path, sep=';')
    
    # 1. General aggregation for time series modeling
    agg = df.groupby(['Country', 'Disaster Type', 'Year']).size().reset_index(name='Disaster_Count')
    
    # 2. Specific aggregation for Indonesia Flood analysis
    df_indo_flood = df[(df['Country'] == 'Indonesia') & (df['Disaster Type'] == 'Flood')].copy()
    df_indo_flood['Decade'] = (df_indo_flood['Year'] // 10) * 10
    
    freq_per_decade = df_indo_flood.groupby('Decade').size().reset_index(name='Disaster_Count')
    
    # Normalize per year because 2020s decade is incomplete (only up to 2023 in the dataset)
    decade_years = {
        1950: 1, 1960: 10, 1970: 10, 1980: 10,
        1990: 10, 2000: 10, 2010: 10, 2020: 4
    }
    freq_per_decade['Years_in_Decade'] = freq_per_decade['Decade'].map(decade_years)
    freq_per_decade['Annual_Average'] = (
        freq_per_decade['Disaster_Count'] / freq_per_decade['Years_in_Decade']
    ).round(2)
    
    # Save outputs
    agg_path = os.path.join(processed_dir, 'processed_agg.csv')
    indo_flood_path = os.path.join(processed_dir, 'processed_indo_flood.csv')
    
    agg.to_csv(agg_path, index=False)
    freq_per_decade.to_csv(indo_flood_path, index=False)
    
    print("[SUCCESS] Preprocessing completed successfully!")
    print(f"   - Saved: {agg_path}")
    print(f"   - Saved: {indo_flood_path}")

if __name__ == '__main__':
    # Support Colab: set env var PROJECT_ROOT before calling this script
    project_root = os.environ.get('PROJECT_ROOT', os.path.abspath('.'))

    raw_path      = os.path.join(project_root, 'data', 'raw',
                        '_EmergencyEventsDatabase-CountryProfiles_emdat-country-profiles_2023_04_06.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    preprocess_data(raw_path, processed_dir)
