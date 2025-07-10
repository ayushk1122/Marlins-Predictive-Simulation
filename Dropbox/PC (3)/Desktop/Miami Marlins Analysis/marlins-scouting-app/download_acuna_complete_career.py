import pybaseball as pb
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def download_acuna_complete_career():
    """
    Download Ronald Acuña Jr.'s complete career Statcast data with all columns
    """
    print("🔍 Downloading Ronald Acuña Jr.'s complete career data...")
    
    # Ronald Acuña Jr.'s MLBAM ID (from previous searches)
    acuna_id = 660271
    
    # Get his career data from 2018 to present
    start_year = 2018
    end_year = datetime.now().year
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        print(f"📅 Downloading {year} season...")
        
        try:
            # Download Statcast data for Acuña for this year
            year_data = pb.statcast_batter(
                f"{year}-01-01", 
                f"{year}-12-31", 
                acuna_id
            )
            
            if not year_data.empty:
                print(f"✅ {year}: {len(year_data)} pitches downloaded")
                all_data.append(year_data)
            else:
                print(f"⚠️  {year}: No data found")
                
        except Exception as e:
            print(f"❌ Error downloading {year} data: {e}")
    
    if all_data:
        # Combine all years
        complete_career = pd.concat(all_data, ignore_index=True)
        
        # Define the specific columns you want
        desired_columns = [
            'game_date', 'at_bat_number', 'game_pk', 'pitch_type', 'pitch_name',
            'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
            'release_pos_x', 'release_pos_y', 'release_pos_z', 'vx0', 'vy0', 'vz0',
            'ax', 'ay', 'az', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone',
            'sz_top', 'sz_bot', 'api_break_z_with_gravity', 'api_break_x_arm',
            'api_break_x_batter_in', 'p_throws', 'if_fielding_alignment',
            'of_fielding_alignment', 'balls', 'strikes', 'description', 'events',
            'launch_speed', 'launch_angle', 'hit_distance_sc', 'bb_type',
            'estimated_ba_using_speedangle', 'estimated_slg_using_speedangle',
            'estimated_woba_using_speedangle', 'swing_length', 'bat_speed',
            'attack_angle', 'swing_path_tilt', 'arm_angle', 'spin_dir',
            'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated',
            'stand', 'effective_speed', 'babip_value', 'iso_value',
            'launch_speed_angle', 'hyper_speed', 'age_pit', 'age_bat',
            'age_pit_legacy', 'age_bat_legacy', 'attack_direction', 'home_team', 'stadium'
        ]
        
        # Filter to only include columns that exist in the dataset
        available_columns = [col for col in desired_columns if col in complete_career.columns]
        missing_columns = [col for col in desired_columns if col not in complete_career.columns]
        
        print(f"\n📋 Requested columns found: {len(available_columns)}")
        print(f"❌ Missing columns: {len(missing_columns)}")
        if missing_columns:
            print(f"Missing: {missing_columns}")
        
        # Select only the desired columns
        filtered_career = complete_career[available_columns]
        
        print(f"\n📊 Filtered career data shape: {filtered_career.shape}")
        print(f"📅 Date range: {filtered_career['game_date'].min()} to {filtered_career['game_date'].max()}")
        
        # Save the complete dataset with only desired columns
        filename = 'ronald_acuna_jr_complete_career_statcast.csv'
        filtered_career.to_csv(filename, index=False)
        print(f"💾 Complete career data saved to: {filename}")
        
        # Show available columns
        print(f"\n📋 Available columns ({len(complete_career.columns)} total):")
        for i, col in enumerate(complete_career.columns, 1):
            print(f"{i:2d}. {col}")
        
        # Show data quality info
        print(f"\n📈 Data Quality:")
        print(f"- Total pitches: {len(complete_career)}")
        print(f"- Unique games: {complete_career['game_pk'].nunique()}")
        print(f"- Unique at-bats: {complete_career['at_bat_number'].nunique()}")
        
        # Convert game_date to datetime if it's not already
        if complete_career['game_date'].dtype == 'object':
            complete_career['game_date'] = pd.to_datetime(complete_career['game_date'])
        
        print(f"- Years covered: {complete_career['game_date'].dt.year.nunique()}")
        
        # Show sample of the data
        print(f"\n🔍 Sample data (first 3 rows):")
        print(complete_career.head(3).to_string())
        
        return complete_career
    else:
        print("❌ No data was downloaded")
        return None

if __name__ == "__main__":
    print("🚀 RONALD ACUÑA JR. COMPLETE CAREER DOWNLOADER")
    print("=" * 60)
    
    data = download_acuna_complete_career()
    
    if data is not None:
        print("\n✅ Download complete!")
        print("📁 File saved as: ronald_acuna_jr_complete_career_statcast.csv")
    else:
        print("❌ Download failed") 