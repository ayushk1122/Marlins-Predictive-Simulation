import pandas as pd
import numpy as np

def load_career_data():
    """Load Acuna Jr.'s complete career data"""
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"Loaded {len(df)} career pitches")
        return df
    except Exception as e:
        print(f"Error loading career data: {e}")
        return None

def identify_hit_safely(row):
    """Identify if a pitch resulted in a hit (single, double, triple, home run)"""
    # Must be hit into play
    if row['description'] != 'hit_into_play':
        return False
    
    # Check events field for hit indicators
    if pd.isna(row['events']):
        return False
    
    events_str = str(row['events']).lower()
    
    # Check for hit indicators
    hit_indicators = [
        'single', 'double', 'triple', 'home run',
        '1b', '2b', '3b', 'hr'
    ]
    
    return any(indicator in events_str for indicator in hit_indicators)

def identify_field_out(row):
    """Identify if a pitch resulted in a field out"""
    # Must be hit into play
    if row['description'] != 'hit_into_play':
        return False
    
    # Check events field for field out
    if pd.isna(row['events']):
        return False
    
    events_str = str(row['events']).lower()
    
    return 'field_out' in events_str

def calculate_pitch_type_zone_batting_averages(df):
    """Calculate batting averages and whiff rates for each pitch type and zone combination"""
    print("Calculating batting averages and whiff rates by pitch type and zone...")
    
    # Clean and filter data
    df = df.copy()
    df['zone'] = df['zone'].fillna(0).astype(int)
    df['pitch_type'] = df['pitch_type'].fillna('Unknown')
    
    # Only include valid zones
    df = df[df['zone'] > 0]
    
    # Identify hit_safely and field_out pitches
    df['hit_safely'] = df.apply(identify_hit_safely, axis=1)
    df['field_out'] = df.apply(identify_field_out, axis=1)
    
    # Create swing and whiff columns for better whiff rate calculation
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt', 'single', 'double', 'triple', 'home run']
    whiff_events = ['swinging_strike', 'swinging_strike_blocked']
    
    df['is_swing'] = df['description'].isin(swing_events).astype(int)
    df['is_whiff'] = df['description'].isin(whiff_events).astype(int)
    
    # Group by pitch type and zone
    results = []
    
    pitch_types = df['pitch_type'].unique()
    zones = range(1, 15)  # Zones 1-14
    
    for pitch_type in pitch_types:
        if pitch_type == 'Unknown':
            continue
            
        for zone in zones:
            # Filter for this specific combination
            mask = (
                (df['pitch_type'] == pitch_type) & 
                (df['zone'] == zone)
            )
            
            zone_pitch_data = df[mask]
            
            if len(zone_pitch_data) >= 3:  # Minimum sample size
                total_pitches = len(zone_pitch_data)
                
                # Count balls in play (hit_into_play)
                balls_in_play = len(zone_pitch_data[zone_pitch_data['description'] == 'hit_into_play'])
                
                # Count hits and field outs
                hit_safely_pitches = zone_pitch_data['hit_safely'].sum()
                field_out_pitches = zone_pitch_data['field_out'].sum()
                
                # Calculate batting average on balls in play
                batting_average = hit_safely_pitches / balls_in_play if balls_in_play > 0 else 0.0
                
                # Calculate whiff rate (swinging strikes / total swings)
                total_swings = zone_pitch_data['is_swing'].sum()
                total_whiffs = zone_pitch_data['is_whiff'].sum()
                whiff_rate = total_whiffs / total_swings if total_swings > 0 else 0.0
                
                # Field out rate on balls in play
                field_out_rate = field_out_pitches / balls_in_play if balls_in_play > 0 else 0.0
                
                results.append({
                    'pitch_type': pitch_type,
                    'zone': zone,
                    'total_pitches': total_pitches,
                    'balls_in_play': balls_in_play,
                    'hit_safely_pitches': hit_safely_pitches,
                    'field_out_pitches': field_out_pitches,
                    'batting_average_bip': batting_average,
                    'total_swings': total_swings,
                    'total_whiffs': total_whiffs,
                    'whiff_rate': whiff_rate,
                    'field_out_rate_bip': field_out_rate
                })
    
    return results

def main():
    # Load data
    df = load_career_data()
    
    if df is None:
        return
    
    # Calculate batting averages
    results = calculate_pitch_type_zone_batting_averages(df)
    
    if not results:
        print("No valid data found for analysis")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by pitch type and zone for better readability
    results_df = results_df.sort_values(['pitch_type', 'zone'])
    
    # Format the CSV output with nice spacing
    print(f"\nCalculated batting averages for {len(results_df)} pitch type x zone combinations")
    
    # Save to CSV with nice formatting
    csv_content = "Pitch Type,Zone,Total Pitches,Balls in Play,Hit Safely Pitches,Field Out Pitches,Batting Average (BIP),Total Swings,Total Whiffs,Whiff Rate,Field Out Rate (BIP)\n"
    
    for _, row in results_df.iterrows():
        csv_content += f"{row['pitch_type']},{row['zone']},{row['total_pitches']},{row['balls_in_play']},{row['hit_safely_pitches']},{row['field_out_pitches']},{row['batting_average_bip']:.3f},{row['total_swings']},{row['total_whiffs']},{row['whiff_rate']:.3f},{row['field_out_rate_bip']:.3f}\n"
    
    # Write to file
    with open('pitch_type_zone_batting_averages.csv', 'w') as f:
        f.write(csv_content)
    
    print(f"Results saved to pitch_type_zone_batting_averages.csv")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total combinations analyzed: {len(results_df)}")
    print(f"Average batting average (BIP): {results_df['batting_average_bip'].mean():.3f}")
    print(f"Average whiff rate: {results_df['whiff_rate'].mean():.3f}")
    print(f"Average field out rate (BIP): {results_df['field_out_rate_bip'].mean():.3f}")
    
    # Show top performing zones
    print(f"\nTop 5 zones by batting average (BIP):")
    top_batting = results_df.nlargest(5, 'batting_average_bip')
    for _, row in top_batting.iterrows():
        print(f"  {row['pitch_type']} Zone {row['zone']}: {row['batting_average_bip']:.3f} ({row['hit_safely_pitches']}/{row['balls_in_play']} BIP)")
    
    # Show highest whiff zones
    print(f"\nTop 5 zones by whiff rate:")
    top_whiff = results_df[results_df['total_swings'] >= 5].nlargest(5, 'whiff_rate')
    for _, row in top_whiff.iterrows():
        print(f"  {row['pitch_type']} Zone {row['zone']}: {row['whiff_rate']:.3f} ({row['total_whiffs']}/{row['total_swings']} swings)")
    
    # Show zones with most balls in play
    print(f"\nTop 5 zones by balls in play:")
    top_bip = results_df.nlargest(5, 'balls_in_play')
    for _, row in top_bip.iterrows():
        print(f"  {row['pitch_type']} Zone {row['zone']}: {row['balls_in_play']} BIP (BA: {row['batting_average_bip']:.3f})")
    
    # Show breakdown by pitch type
    print(f"\nAverage BABIP by pitch type:")
    pitch_type_avg = results_df.groupby('pitch_type')['batting_average_bip'].mean().sort_values(ascending=False)
    for pitch_type, avg_ba in pitch_type_avg.items():
        print(f"  {pitch_type}: {avg_ba:.3f}")
    
    print(f"\nAverage whiff rate by pitch type:")
    pitch_type_whiff = results_df.groupby('pitch_type')['whiff_rate'].mean().sort_values(ascending=False)
    for pitch_type, avg_whiff in pitch_type_whiff.items():
        print(f"  {pitch_type}: {avg_whiff:.3f}")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 