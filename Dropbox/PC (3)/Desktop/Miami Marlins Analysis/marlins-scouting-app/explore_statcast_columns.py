import pybaseball as pb
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def explore_statcast_columns():
    """
    Explore all available columns in pybaseball's Statcast data
    """
    print("🔍 Exploring Statcast columns...")
    
    # Get current date and go back 30 days to get a good sample
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"📅 Downloading Statcast data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # Download a sample of Statcast data
        statcast_data = pb.statcast(
            start_dt=start_date.strftime('%Y-%m-%d'),
            end_dt=end_date.strftime('%Y-%m-%d'),
            player_type='batter'
        )
        
        print(f"✅ Downloaded {len(statcast_data)} records")
        print(f"📊 Dataset shape: {statcast_data.shape}")
        
        # Display all columns
        print("\n📋 ALL AVAILABLE COLUMNS:")
        print("=" * 80)
        for i, col in enumerate(statcast_data.columns, 1):
            print(f"{i:2d}. {col}")
        
        print(f"\n📈 Total columns: {len(statcast_data.columns)}")
        
        # Show data types
        print("\n📊 DATA TYPES:")
        print("=" * 80)
        dtype_counts = statcast_data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} columns")
        
        # Show sample of non-null values for each column
        print("\n🔍 SAMPLE VALUES (first 5 non-null values):")
        print("=" * 80)
        for col in statcast_data.columns:
            non_null_values = statcast_data[col].dropna().head(5)
            if len(non_null_values) > 0:
                print(f"\n{col}:")
                for i, val in enumerate(non_null_values):
                    print(f"  {i+1}. {val}")
            else:
                print(f"\n{col}: (all null)")
        
        # Show missing data percentage
        print("\n📉 MISSING DATA ANALYSIS:")
        print("=" * 80)
        missing_data = statcast_data.isnull().sum()
        missing_percentage = (missing_data / len(statcast_data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percentage': missing_percentage.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df.to_string(index=False))
        
        # Save the sample data for further exploration
        statcast_data.to_csv('statcast_sample_data.csv', index=False)
        print(f"\n💾 Sample data saved to 'statcast_sample_data.csv'")
        
        return statcast_data
        
    except Exception as e:
        print(f"❌ Error downloading Statcast data: {e}")
        return None

def explore_specific_columns():
    """
    Explore specific important columns in detail
    """
    print("\n🎯 EXPLORING SPECIFIC IMPORTANT COLUMNS:")
    print("=" * 80)
    
    try:
        # Load the sample data
        data = pd.read_csv('statcast_sample_data.csv')
        
        # Important column categories
        important_columns = {
            'Pitch Information': ['pitch_type', 'pitch_name', 'release_speed', 'release_spin_rate', 'spin_axis'],
            'Release Position': ['release_pos_x', 'release_pos_y', 'release_pos_z', 'release_extension'],
            'Velocity': ['vx0', 'vy0', 'vz0', 'ax', 'ay', 'az'],
            'Movement': ['pfx_x', 'pfx_z', 'api_break_z_with_gravity', 'api_break_x_batter_in'],
            'Plate Location': ['plate_x', 'plate_z', 'zone', 'sz_top', 'sz_bot'],
            'Game Context': ['balls', 'strikes', 'outs_when_up', 'inning', 'inning_topbot'],
            'Batter Info': ['batter', 'batter_name', 'bat_speed', 'attack_angle', 'swing_path_tilt'],
            'Pitcher Info': ['pitcher', 'pitcher_name', 'p_throws', 'arm_angle'],
            'Outcomes': ['description', 'events', 'bb_type', 'launch_speed', 'launch_angle'],
            'Expected Stats': ['estimated_ba_using_speedangle', 'estimated_slg_using_speedangle', 'estimated_woba_using_speedangle'],
            'Hit Data': ['hc_x', 'hc_y', 'hit_distance_sc', 'hit_location_1', 'hit_location_2'],
            'Fielding': ['if_fielding_alignment', 'of_fielding_alignment'],
            'Game Info': ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']
        }
        
        for category, columns in important_columns.items():
            print(f"\n📋 {category}:")
            print("-" * 40)
            available_cols = [col for col in columns if col in data.columns]
            if available_cols:
                for col in available_cols:
                    unique_vals = data[col].nunique()
                    null_pct = (data[col].isnull().sum() / len(data)) * 100
                    print(f"  {col}: {unique_vals} unique values, {null_pct:.1f}% null")
                    
                    # Show sample values for categorical columns
                    if data[col].dtype == 'object' and unique_vals < 20:
                        sample_vals = data[col].dropna().unique()[:5]
                        print(f"    Sample values: {sample_vals}")
            else:
                print(f"  No columns from this category found in dataset")
        
        return data
        
    except Exception as e:
        print(f"❌ Error exploring specific columns: {e}")
        return None

if __name__ == "__main__":
    print("🚀 STATCAST COLUMN EXPLORER")
    print("=" * 80)
    
    # Explore all columns
    data = explore_statcast_columns()
    
    if data is not None:
        # Explore specific important columns
        explore_specific_columns()
        
        print("\n✅ Exploration complete!")
        print("\n💡 TIPS:")
        print("- Use 'statcast_sample_data.csv' for further analysis")
        print("- Check the missing data percentages to understand data quality")
        print("- Focus on columns with low missing percentages for modeling")
        print("- Consider the data types when preprocessing")
    else:
        print("❌ Failed to download Statcast data") 