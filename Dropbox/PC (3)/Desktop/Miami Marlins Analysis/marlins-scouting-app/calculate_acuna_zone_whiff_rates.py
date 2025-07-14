import pandas as pd
import numpy as np
from collections import defaultdict

def load_acuna_data():
    """Load Acuna Jr.'s complete career Statcast data"""
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"✓ Loaded Acuna Jr. data with {len(df)} pitches")
        return df
    except FileNotFoundError:
        print("✗ Error: ronald_acuna_jr_complete_career_statcast.csv not found")
        return None

def calculate_zone_whiff_rates(df):
    """Calculate whiff rates by pitch type and zone for Acuna Jr."""
    
    # Filter for swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    
    # Create swing column
    df['is_swing'] = df['description'].isin(swing_events).astype(int)
    
    # Create whiff column (swinging strikes)
    whiff_events = ['swinging_strike', 'swinging_strike_blocked']
    df['is_whiff'] = df['description'].isin(whiff_events).astype(int)
    
    # Filter for swings only
    swing_df = df[df['is_swing'] == 1].copy()
    
    print(f"Total swings: {len(swing_df)}")
    print(f"Total whiffs: {swing_df['is_whiff'].sum()}")
    print(f"Overall whiff rate: {swing_df['is_whiff'].mean():.3f}")
    
    # Group by pitch type and zone
    results = []
    
    for (pitch_type, zone), group in swing_df.groupby(['pitch_type', 'zone']):
        total_swings = len(group)
        total_whiffs = group['is_whiff'].sum()
        whiff_rate = total_whiffs / total_swings if total_swings > 0 else 0
        
        results.append({
            'Pitch Type': pitch_type,
            'Zone': zone,
            'Total Swings': total_swings,
            'Total Whiffs': total_whiffs,
            'Whiff Rate': whiff_rate
        })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['Pitch Type', 'Zone'])
    
    return results_df

def analyze_whiff_patterns(df):
    """Analyze whiff patterns by different factors"""
    
    # Filter for swings only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    swing_df = df[df['description'].isin(swing_events)].copy()
    
    whiff_events = ['swinging_strike', 'swinging_strike_blocked']
    swing_df['is_whiff'] = swing_df['description'].isin(whiff_events).astype(int)
    
    print("\n=== WHIFF PATTERN ANALYSIS ===")
    
    # By pitch type
    print("\nWhiff rates by pitch type:")
    pitch_whiff = swing_df.groupby('pitch_type')['is_whiff'].agg(['count', 'sum', 'mean']).round(3)
    pitch_whiff.columns = ['Total Swings', 'Total Whiffs', 'Whiff Rate']
    print(pitch_whiff)
    
    # By zone
    print("\nWhiff rates by zone:")
    zone_whiff = swing_df.groupby('zone')['is_whiff'].agg(['count', 'sum', 'mean']).round(3)
    zone_whiff.columns = ['Total Swings', 'Total Whiffs', 'Whiff Rate']
    print(zone_whiff)
    
    # By count
    print("\nWhiff rates by count:")
    swing_df['count'] = swing_df['balls'].astype(str) + '-' + swing_df['strikes'].astype(str)
    count_whiff = swing_df.groupby('count')['is_whiff'].agg(['count', 'sum', 'mean']).round(3)
    count_whiff.columns = ['Total Swings', 'Total Whiffs', 'Whiff Rate']
    print(count_whiff)
    
    # By velocity ranges
    print("\nWhiff rates by velocity range:")
    swing_df['velocity_range'] = pd.cut(swing_df['release_speed'], 
                                       bins=[0, 85, 90, 95, 100, 105], 
                                       labels=['<85', '85-90', '90-95', '95-100', '100+'])
    vel_whiff = swing_df.groupby('velocity_range')['is_whiff'].agg(['count', 'sum', 'mean']).round(3)
    vel_whiff.columns = ['Total Swings', 'Total Whiffs', 'Whiff Rate']
    print(vel_whiff)
    
    # By movement ranges
    print("\nWhiff rates by movement range:")
    swing_df['movement_range'] = pd.cut(swing_df['movement_magnitude'], 
                                       bins=[0, 2, 4, 6, 8, 20], 
                                       labels=['<2', '2-4', '4-6', '6-8', '8+'])
    mov_whiff = swing_df.groupby('movement_range')['is_whiff'].agg(['count', 'sum', 'mean']).round(3)
    mov_whiff.columns = ['Total Swings', 'Total Whiffs', 'Whiff Rate']
    print(mov_whiff)

def main():
    """Main function to calculate and save zone-specific whiff rates"""
    
    print("=== ACUNA JR. ZONE-SPECIFIC WHIFF RATE CALCULATOR ===")
    
    # Load data
    df = load_acuna_data()
    if df is None:
        return
    
    # Calculate zone-specific whiff rates
    print("\nCalculating zone-specific whiff rates...")
    whiff_rates = calculate_zone_whiff_rates(df)
    
    # Save to CSV
    output_file = 'acuna_zone_whiff_rates.csv'
    whiff_rates.to_csv(output_file, index=False)
    print(f"\n✓ Saved whiff rates to {output_file}")
    print(f"✓ Calculated whiff rates for {len(whiff_rates)} pitch type x zone combinations")
    
    # Display summary
    print("\n=== SUMMARY ===")
    print(f"Total combinations: {len(whiff_rates)}")
    print(f"Pitch types: {whiff_rates['Pitch Type'].nunique()}")
    print(f"Zones: {whiff_rates['Zone'].nunique()}")
    
    # Show highest and lowest whiff rates
    print("\nHighest whiff rates:")
    high_whiff = whiff_rates[whiff_rates['Total Swings'] >= 5].nlargest(10, 'Whiff Rate')
    for _, row in high_whiff.iterrows():
        print(f"  {row['Pitch Type']} Zone {row['Zone']}: {row['Whiff Rate']:.3f} ({row['Total Whiffs']}/{row['Total Swings']})")
    
    print("\nLowest whiff rates (min 5 swings):")
    low_whiff = whiff_rates[whiff_rates['Total Swings'] >= 5].nsmallest(10, 'Whiff Rate')
    for _, row in low_whiff.iterrows():
        print(f"  {row['Pitch Type']} Zone {row['Zone']}: {row['Whiff Rate']:.3f} ({row['Total Whiffs']}/{row['Total Swings']})")
    
    # Analyze patterns
    analyze_whiff_patterns(df)
    
    print(f"\n✓ Zone-specific whiff rates saved to {output_file}")
    print("Use this data in your training scripts to improve whiff prediction!")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 