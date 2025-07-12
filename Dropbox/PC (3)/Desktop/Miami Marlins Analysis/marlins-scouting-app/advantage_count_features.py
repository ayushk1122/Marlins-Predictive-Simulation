import pandas as pd
import numpy as np

def calculate_advantage_count_features(df):
    """
    Calculate advantage count features dynamically from the dataset.
    
    Args:
        df: DataFrame with 'balls', 'strikes', 'pitch_type', and 'events' columns
        
    Returns:
        dict: Dictionary of advantage count features
    """
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
            
        # Overall swing rate for this count
        overall_swing_rate = count_data['is_swing'].mean()
        
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
        
        results[count] = {
            'overall_swing_rate': overall_swing_rate,
            'pitch_type_swings': pitch_type_swings,
            'weight': info['weight'],
            'total_pitches': len(count_data)
        }
    
    # Create features for the classifier
    advantage_features = {}
    
    for count in advantage_counts.keys():
        if count in results:
            # Overall advantage swing rate
            advantage_features[f'{count}_advantage_swing_rate'] = results[count]['overall_swing_rate']
            
            # Weighted advantage swing rate
            advantage_features[f'{count}_weighted_advantage_swing_rate'] = (
                results[count]['overall_swing_rate'] * results[count]['weight']
            )
            
            # Pitch type specific rates
            for pitch_type in ['FF', 'SL', 'CH']:  # Focus on main pitch types
                if pitch_type in results[count]['pitch_type_swings']:
                    swing_rate = results[count]['pitch_type_swings'][pitch_type]['swing_rate']
                    advantage_features[f'{count}_{pitch_type}_advantage_swing_rate'] = swing_rate
                    advantage_features[f'{count}_{pitch_type}_weighted_advantage_swing_rate'] = (
                        swing_rate * results[count]['weight']
                    )
    
    return advantage_features, results

def get_current_advantage_features(balls, strikes, pitch_type, advantage_features):
    """
    Get advantage features for a specific count and pitch type.
    
    Args:
        balls: Number of balls
        strikes: Number of strikes
        pitch_type: Type of pitch
        advantage_features: Dictionary of calculated advantage features
        
    Returns:
        dict: Features for the current count and pitch type
    """
    current_count = f"{balls}-{strikes}"
    
    # Define advantage count weights
    advantage_weights = {
        '2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3
    }
    
    features = {}
    
    # Check if this is an advantage count
    if current_count in advantage_weights:
        weight = advantage_weights[current_count]
        overall_rate = advantage_features.get(f'{current_count}_advantage_swing_rate', 0.5)
        pitch_specific_rate = advantage_features.get(f'{current_count}_{pitch_type}_advantage_swing_rate', overall_rate)
        
        features[f'current_advantage_swing_rate'] = overall_rate
        features[f'current_advantage_weighted_swing_rate'] = overall_rate * weight
        features[f'current_advantage_{pitch_type}_swing_rate'] = pitch_specific_rate
        features[f'current_advantage_{pitch_type}_weighted_swing_rate'] = pitch_specific_rate * weight
        features[f'advantage_count_weight'] = weight
    else:
        # Default values for non-advantage counts
        features[f'current_advantage_swing_rate'] = 0.5
        features[f'current_advantage_weighted_swing_rate'] = 0.25
        features[f'current_advantage_{pitch_type}_swing_rate'] = 0.5
        features[f'current_advantage_{pitch_type}_weighted_swing_rate'] = 0.25
        features[f'advantage_count_weight'] = 0.5
    
    return features 