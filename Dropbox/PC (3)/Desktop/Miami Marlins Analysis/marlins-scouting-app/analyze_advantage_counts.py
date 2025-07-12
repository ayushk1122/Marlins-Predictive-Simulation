import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_analyze_advantage_counts():
    """Analyze Acuna's swing rates in different advantage counts"""
    
    # Load career data
    df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
    print(f"Loaded {len(df)} pitches from Acuna's career")
    
    # Define swing events
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
                   'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
                   'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
                   'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
                   'sac_bunt_double_play', 'grounded_into_double_play', 'batter_interference',
                   'catcher_interference', 'fan_interference', 'strikeout', 'strikeout_double_play', 
                   'strikeout_triple_play', 'walk', 'intent_walk', 'hit_by_pitch',
                   'sacrifice_bunt_double_play', 'sacrifice_bunt_triple_play', 'umpire_interference']
    
    # Add swing indicator
    df['is_swing'] = df['events'].isin(swing_events)
    
    # Create count combinations
    df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    
    # Define advantage counts with weights
    advantage_counts = {
        '2-0': {'weight': 0.9, 'description': 'Best advantage - 2 balls, 0 strikes'},
        '3-1': {'weight': 0.85, 'description': 'Strong advantage - 3 balls, 1 strike'},
        '2-1': {'weight': 0.7, 'description': 'Good advantage - 2 balls, 1 strike'},
        '1-0': {'weight': 0.6, 'description': 'Minor advantage - 1 ball, 0 strikes'},
        '3-0': {'weight': 0.3, 'description': 'Rare swing - 3 balls, 0 strikes'}
    }
    
    # Analyze swing rates by count and pitch type
    results = {}
    
    for count, info in advantage_counts.items():
        count_data = df[df['count'] == count]
        if len(count_data) == 0:
            continue
            
        print(f"\n=== {count} Count Analysis ===")
        print(f"Total pitches: {len(count_data)}")
        print(f"Weight: {info['weight']}")
        print(f"Description: {info['description']}")
        
        # Overall swing rate for this count
        overall_swing_rate = count_data['is_swing'].mean()
        print(f"Overall swing rate: {overall_swing_rate:.3f}")
        
        # Swing rates by pitch type
        pitch_type_swings = {}
        for pitch_type in count_data['pitch_type'].unique():
            if pd.notna(pitch_type):
                pitch_data = count_data[count_data['pitch_type'] == pitch_type]
                if len(pitch_data) >= 5:  # Only include if we have enough data
                    swing_rate = pitch_data['is_swing'].mean()
                    pitch_type_swings[pitch_type] = {
                        'swing_rate': swing_rate,
                        'count': len(pitch_data)
                    }
                    print(f"  {pitch_type}: {swing_rate:.3f} ({len(pitch_data)} pitches)")
        
        results[count] = {
            'overall_swing_rate': overall_swing_rate,
            'pitch_type_swings': pitch_type_swings,
            'weight': info['weight'],
            'total_pitches': len(count_data)
        }
    
    # Calculate weighted advantage swing rates
    print("\n=== Weighted Advantage Swing Analysis ===")
    
    # Group pitch types
    fastball_types = ['FF', 'SI', 'FC', 'FT']
    breaking_types = ['SL', 'CU', 'KC', 'SV']
    offspeed_types = ['CH', 'FS', 'FO']
    
    for pitch_group, pitch_types in [('Fastball', fastball_types), ('Breaking', breaking_types), ('Offspeed', offspeed_types)]:
        print(f"\n{pitch_group} Pitches:")
        
        for count in advantage_counts.keys():
            if count in results:
                pitch_swings = results[count]['pitch_type_swings']
                group_swings = []
                
                for pitch_type in pitch_types:
                    if pitch_type in pitch_swings:
                        group_swings.append(pitch_swings[pitch_type]['swing_rate'])
                
                if group_swings:
                    avg_swing_rate = np.mean(group_swings)
                    weighted_rate = avg_swing_rate * results[count]['weight']
                    print(f"  {count}: {avg_swing_rate:.3f} (weighted: {weighted_rate:.3f})")
    
    # Create features for the classifier
    print("\n=== Advantage Count Features for Classifier ===")
    
    advantage_features = {}
    
    for count in advantage_counts.keys():
        if count in results:
            # Overall advantage swing rate
            advantage_features[f'acuna_{count}_advantage_swing_rate'] = results[count]['overall_swing_rate']
            
            # Weighted advantage swing rate
            advantage_features[f'acuna_{count}_weighted_advantage_swing_rate'] = (
                results[count]['overall_swing_rate'] * results[count]['weight']
            )
            
            # Pitch type specific rates
            for pitch_type in ['FF', 'SL', 'CH']:  # Focus on main pitch types
                if pitch_type in results[count]['pitch_type_swings']:
                    swing_rate = results[count]['pitch_type_swings'][pitch_type]['swing_rate']
                    advantage_features[f'acuna_{count}_{pitch_type}_advantage_swing_rate'] = swing_rate
                    advantage_features[f'acuna_{count}_{pitch_type}_weighted_advantage_swing_rate'] = (
                        swing_rate * results[count]['weight']
                    )
    
    # Print all features
    for feature, value in advantage_features.items():
        print(f"{feature}: {value:.3f}")
    
    return advantage_features, results

def create_advantage_count_visualization(results):
    """Create visualization of swing rates in advantage counts"""
    
    # Prepare data for plotting
    counts = list(results.keys())
    swing_rates = [results[count]['overall_swing_rate'] for count in counts]
    weights = [results[count]['weight'] for count in counts]
    weighted_rates = [results[count]['overall_swing_rate'] * results[count]['weight'] for count in counts]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Raw swing rates
    bars1 = ax1.bar(counts, swing_rates, color='skyblue', alpha=0.7)
    ax1.set_title('Acuna Swing Rates by Advantage Count')
    ax1.set_ylabel('Swing Rate')
    ax1.set_xlabel('Count')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars1, swing_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    # Plot 2: Weighted swing rates
    bars2 = ax2.bar(counts, weighted_rates, color='lightcoral', alpha=0.7)
    ax2.set_title('Acuna Weighted Swing Rates by Advantage Count')
    ax2.set_ylabel('Weighted Swing Rate')
    ax2.set_xlabel('Count')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars2, weighted_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('advantage_count_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'advantage_count_analysis.png'")

if __name__ == "__main__":
    advantage_features, results = load_and_analyze_advantage_counts()
    create_advantage_count_visualization(results)
    
    print(f"\nGenerated {len(advantage_features)} advantage count features for the classifier") 