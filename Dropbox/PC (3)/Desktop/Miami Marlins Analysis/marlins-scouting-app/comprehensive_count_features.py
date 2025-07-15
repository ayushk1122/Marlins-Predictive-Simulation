import pandas as pd
import numpy as np

def calculate_comprehensive_count_features(df):
    """
    Calculate comprehensive count-specific swing rates for all pitch types.
    
    Args:
        df: DataFrame with 'balls', 'strikes', 'pitch_type', and 'events' columns
        
    Returns:
        dict: Dictionary of comprehensive count features
    """
    # Make a copy to avoid fragmentation
    df = df.copy()
    
    # Define swing events - be more specific to avoid false positives
    # Only include events that clearly indicate a swing
    swing_events = [
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
        'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
        'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
        'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
        'sac_bunt_double_play', 'grounded_into_double_play'
    ]
    
    # Define non-swing events explicitly
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'walk', 'intent_walk', 'hit_by_pitch', 'strikeout', 'strikeout_double_play', 
        'strikeout_triple_play', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 
        'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home',
        'catcher_pickoff_1b', 'catcher_pickoff_2b', 'catcher_pickoff_3b', 
        'catcher_pickoff_caught_stealing_2b', 'catcher_pickoff_caught_stealing_3b', 
        'catcher_pickoff_caught_stealing_home', 'runner_double_play', 'batter_interference', 
        'catcher_interference', 'fan_interference', 'umpire_interference'
    ]
    
    # Define non-swing events to be more explicit
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_caught_stealing_2b',
        'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home', 'catcher_pickoff_1b',
        'catcher_pickoff_2b', 'catcher_pickoff_3b', 'catcher_pickoff_caught_stealing_2b',
        'catcher_pickoff_caught_stealing_3b', 'catcher_pickoff_caught_stealing_home',
        'runner_double_play', 'batter_interference', 'catcher_interference', 'umpire_interference'
    ]
    
    # Create swing indicator - be more careful about what constitutes a swing
    # First, check if events are in swing events
    df['is_swing'] = df['events'].isin(swing_events)
    
    # Then, explicitly mark non-swing events as False
    df.loc[df['events'].isin(non_swing_events), 'is_swing'] = False
    
    # Debug: Check what events we have and their swing classification
    unique_events = df['events'].unique()
    print(f"Found {len(unique_events)} unique events in dataset")
    print("Sample events:", unique_events[:10])
    
    # Show swing rate breakdown
    swing_count = df['is_swing'].sum()
    total_count = len(df)
    print(f"Total pitches: {total_count}")
    print(f"Swing pitches: {swing_count}")
    print(f"Overall swing rate: {swing_count/total_count:.3f}")
    
    # Show breakdown by event type
    event_swing_rates = df.groupby('events')['is_swing'].agg(['count', 'sum', 'mean']).sort_values('count', ascending=False)
    print("\nTop 10 events by frequency:")
    print(event_swing_rates.head(10))
    
    # Create count combinations
    df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    
    # Get all unique counts and pitch types
    all_counts = df['count'].unique()
    all_pitch_types = df['pitch_type'].unique()
    
    # Define advantage count weights for special handling
    advantage_weights = {
        '2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3
    }
    
    # Calculate comprehensive count features
    count_features = {}
    count_results = {}
    
    print("Calculating comprehensive count features...")
    print(f"Found {len(all_counts)} unique counts and {len(all_pitch_types)} pitch types")
    
    for count in all_counts:
        count_data = df[df['count'] == count]
        if len(count_data) == 0:
            continue
            
        # Overall swing rate for this count
        overall_swing_rate = count_data['is_swing'].mean()
        
        # Swing rates by pitch type
        pitch_type_swings = {}
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type):
                pitch_data = count_data[count_data['pitch_type'] == pitch_type]
                if len(pitch_data) >= 3:  # Lower threshold to include more data
                    swing_rate = pitch_data['is_swing'].mean()
                    pitch_type_swings[pitch_type] = {
                        'swing_rate': swing_rate,
                        'count': len(pitch_data)
                    }
        
        count_results[count] = {
            'overall_swing_rate': overall_swing_rate,
            'pitch_type_swings': pitch_type_swings,
            'total_pitches': len(count_data)
        }
        
        # Create features for this count
        count_features[f'{count}_overall_swing_rate'] = overall_swing_rate
        
        # Add pitch type specific rates
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type) and pitch_type in pitch_type_swings:
                swing_rate = pitch_type_swings[pitch_type]['swing_rate']
                count_features[f'{count}_{pitch_type}_swing_rate'] = swing_rate
                
                # Add weighted version for advantage counts
                if count in advantage_weights:
                    weight = advantage_weights[count]
                    count_features[f'{count}_{pitch_type}_weighted_swing_rate'] = swing_rate * weight
                    count_features[f'{count}_weighted_swing_rate'] = overall_swing_rate * weight
    
    print(f"Generated {len(count_features)} count-specific features")
    
    # Print summary of key findings
    print("\nKey Count Analysis:")
    for count in sorted(count_results.keys()):
        result = count_results[count]
        print(f"  {count}: {result['overall_swing_rate']:.3f} overall ({result['total_pitches']} pitches)")
        
        # Show top 3 pitch types for this count
        pitch_rates = [(pt, data['swing_rate']) for pt, data in result['pitch_type_swings'].items()]
        pitch_rates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (pitch_type, rate) in enumerate(pitch_rates[:3]):
            print(f"    {pitch_type}: {rate:.3f}")
    
    return count_features, count_results

def get_count_features_for_pitch(balls, strikes, pitch_type, count_features):
    """
    Get count-specific features for a particular pitch.
    
    Args:
        balls: Number of balls
        strikes: Number of strikes
        pitch_type: Type of pitch
        count_features: Dictionary of calculated count features
        
    Returns:
        dict: Features for the current count and pitch type
    """
    current_count = f"{balls}-{strikes}"
    
    features = {}
    
    # Get overall count swing rate
    overall_key = f'{current_count}_overall_swing_rate'
    if overall_key in count_features:
        features['current_count_overall_swing_rate'] = count_features[overall_key]
    else:
        features['current_count_overall_swing_rate'] = 0.5  # Default
    
    # Get pitch-specific count swing rate
    pitch_key = f'{current_count}_{pitch_type}_swing_rate'
    if pitch_key in count_features:
        features['current_count_pitch_swing_rate'] = count_features[pitch_key]
    else:
        features['current_count_pitch_swing_rate'] = features['current_count_overall_swing_rate']
    
    # Get weighted versions for advantage counts
    advantage_weights = {'2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3}
    
    if current_count in advantage_weights:
        weight = advantage_weights[current_count]
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * weight
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * weight
        features['advantage_count_weight'] = weight
    else:
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * 0.5
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * 0.5
        features['advantage_count_weight'] = 0.5
    
    return features

def analyze_count_patterns(count_results):
    """
    Analyze patterns in count-specific swing rates.
    
    Args:
        count_results: Results from calculate_comprehensive_count_features
        
    Returns:
        dict: Analysis of count patterns
    """
    analysis = {}
    
    # Find most aggressive counts
    count_rates = [(count, result['overall_swing_rate']) for count, result in count_results.items()]
    count_rates.sort(key=lambda x: x[1], reverse=True)
    
    analysis['most_aggressive_counts'] = count_rates[:5]
    analysis['least_aggressive_counts'] = count_rates[-5:]
    
    # Find pitch type preferences by count
    pitch_preferences = {}
    for count, result in count_results.items():
        if result['pitch_type_swings']:
            # Find most swung at pitch type for this count
            best_pitch = max(result['pitch_type_swings'].items(), 
                           key=lambda x: x[1]['swing_rate'])
            pitch_preferences[count] = {
                'preferred_pitch': best_pitch[0],
                'swing_rate': best_pitch[1]['swing_rate']
            }
    
    analysis['pitch_preferences_by_count'] = pitch_preferences
    
    return analysis

if __name__ == "__main__":
    print("Loading career data for comprehensive count analysis...")
    
    # Load the career data
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"Loaded {len(df)} career pitches for Acuna Jr.")
        
        # Check required columns
        required_cols = ['balls', 'strikes', 'pitch_type', 'events']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            exit(1)
        
        # Calculate comprehensive count features
        print("\n" + "="*50)
        print("COMPREHENSIVE COUNT FEATURE ANALYSIS")
        print("="*50)
        
        count_features, count_results = calculate_comprehensive_count_features(df)
        
        # Analyze patterns
        analysis = analyze_count_patterns(count_results)
        
        print("\n" + "="*50)
        print("PATTERN ANALYSIS")
        print("="*50)
        
        print("\nMost aggressive counts:")
        for count, rate in analysis['most_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nLeast aggressive counts:")
        for count, rate in analysis['least_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nPitch preferences by count:")
        for count, pref in analysis['pitch_preferences_by_count'].items():
            print(f"  {count}: {pref['preferred_pitch']} ({pref['swing_rate']:.3f})")
        
        print(f"\nTotal features generated: {len(count_features)}")
        print("Sample features:")
        for i, (feature, value) in enumerate(list(count_features.items())[:10]):
            print(f"  {feature}: {value:.3f}")
        
    except FileNotFoundError:
        print("Error: Could not find ronald_acuna_jr_complete_career_statcast.csv")
        print("Please ensure the career data file exists in the current directory.")
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc() 
import numpy as np

def calculate_comprehensive_count_features(df):
    """
    Calculate comprehensive count-specific swing rates for all pitch types.
    
    Args:
        df: DataFrame with 'balls', 'strikes', 'pitch_type', and 'events' columns
        
    Returns:
        dict: Dictionary of comprehensive count features
    """
    # Make a copy to avoid fragmentation
    df = df.copy()
    
    # Define swing events - be more specific to avoid false positives
    # Only include events that clearly indicate a swing
    swing_events = [
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
        'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
        'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
        'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
        'sac_bunt_double_play', 'grounded_into_double_play'
    ]
    
    # Define non-swing events explicitly
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'walk', 'intent_walk', 'hit_by_pitch', 'strikeout', 'strikeout_double_play', 
        'strikeout_triple_play', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 
        'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home',
        'catcher_pickoff_1b', 'catcher_pickoff_2b', 'catcher_pickoff_3b', 
        'catcher_pickoff_caught_stealing_2b', 'catcher_pickoff_caught_stealing_3b', 
        'catcher_pickoff_caught_stealing_home', 'runner_double_play', 'batter_interference', 
        'catcher_interference', 'fan_interference', 'umpire_interference'
    ]
    
    # Define non-swing events to be more explicit
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_caught_stealing_2b',
        'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home', 'catcher_pickoff_1b',
        'catcher_pickoff_2b', 'catcher_pickoff_3b', 'catcher_pickoff_caught_stealing_2b',
        'catcher_pickoff_caught_stealing_3b', 'catcher_pickoff_caught_stealing_home',
        'runner_double_play', 'batter_interference', 'catcher_interference', 'umpire_interference'
    ]
    
    # Create swing indicator - be more careful about what constitutes a swing
    # First, check if events are in swing events
    df['is_swing'] = df['events'].isin(swing_events)
    
    # Then, explicitly mark non-swing events as False
    df.loc[df['events'].isin(non_swing_events), 'is_swing'] = False
    
    # Debug: Check what events we have and their swing classification
    unique_events = df['events'].unique()
    print(f"Found {len(unique_events)} unique events in dataset")
    print("Sample events:", unique_events[:10])
    
    # Show swing rate breakdown
    swing_count = df['is_swing'].sum()
    total_count = len(df)
    print(f"Total pitches: {total_count}")
    print(f"Swing pitches: {swing_count}")
    print(f"Overall swing rate: {swing_count/total_count:.3f}")
    
    # Show breakdown by event type
    event_swing_rates = df.groupby('events')['is_swing'].agg(['count', 'sum', 'mean']).sort_values('count', ascending=False)
    print("\nTop 10 events by frequency:")
    print(event_swing_rates.head(10))
    
    # Create count combinations
    df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    
    # Get all unique counts and pitch types
    all_counts = df['count'].unique()
    all_pitch_types = df['pitch_type'].unique()
    
    # Define advantage count weights for special handling
    advantage_weights = {
        '2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3
    }
    
    # Calculate comprehensive count features
    count_features = {}
    count_results = {}
    
    print("Calculating comprehensive count features...")
    print(f"Found {len(all_counts)} unique counts and {len(all_pitch_types)} pitch types")
    
    for count in all_counts:
        count_data = df[df['count'] == count]
        if len(count_data) == 0:
            continue
            
        # Overall swing rate for this count
        overall_swing_rate = count_data['is_swing'].mean()
        
        # Swing rates by pitch type
        pitch_type_swings = {}
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type):
                pitch_data = count_data[count_data['pitch_type'] == pitch_type]
                if len(pitch_data) >= 3:  # Lower threshold to include more data
                    swing_rate = pitch_data['is_swing'].mean()
                    pitch_type_swings[pitch_type] = {
                        'swing_rate': swing_rate,
                        'count': len(pitch_data)
                    }
        
        count_results[count] = {
            'overall_swing_rate': overall_swing_rate,
            'pitch_type_swings': pitch_type_swings,
            'total_pitches': len(count_data)
        }
        
        # Create features for this count
        count_features[f'{count}_overall_swing_rate'] = overall_swing_rate
        
        # Add pitch type specific rates
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type) and pitch_type in pitch_type_swings:
                swing_rate = pitch_type_swings[pitch_type]['swing_rate']
                count_features[f'{count}_{pitch_type}_swing_rate'] = swing_rate
                
                # Add weighted version for advantage counts
                if count in advantage_weights:
                    weight = advantage_weights[count]
                    count_features[f'{count}_{pitch_type}_weighted_swing_rate'] = swing_rate * weight
                    count_features[f'{count}_weighted_swing_rate'] = overall_swing_rate * weight
    
    print(f"Generated {len(count_features)} count-specific features")
    
    # Print summary of key findings
    print("\nKey Count Analysis:")
    for count in sorted(count_results.keys()):
        result = count_results[count]
        print(f"  {count}: {result['overall_swing_rate']:.3f} overall ({result['total_pitches']} pitches)")
        
        # Show top 3 pitch types for this count
        pitch_rates = [(pt, data['swing_rate']) for pt, data in result['pitch_type_swings'].items()]
        pitch_rates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (pitch_type, rate) in enumerate(pitch_rates[:3]):
            print(f"    {pitch_type}: {rate:.3f}")
    
    return count_features, count_results

def get_count_features_for_pitch(balls, strikes, pitch_type, count_features):
    """
    Get count-specific features for a particular pitch.
    
    Args:
        balls: Number of balls
        strikes: Number of strikes
        pitch_type: Type of pitch
        count_features: Dictionary of calculated count features
        
    Returns:
        dict: Features for the current count and pitch type
    """
    current_count = f"{balls}-{strikes}"
    
    features = {}
    
    # Get overall count swing rate
    overall_key = f'{current_count}_overall_swing_rate'
    if overall_key in count_features:
        features['current_count_overall_swing_rate'] = count_features[overall_key]
    else:
        features['current_count_overall_swing_rate'] = 0.5  # Default
    
    # Get pitch-specific count swing rate
    pitch_key = f'{current_count}_{pitch_type}_swing_rate'
    if pitch_key in count_features:
        features['current_count_pitch_swing_rate'] = count_features[pitch_key]
    else:
        features['current_count_pitch_swing_rate'] = features['current_count_overall_swing_rate']
    
    # Get weighted versions for advantage counts
    advantage_weights = {'2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3}
    
    if current_count in advantage_weights:
        weight = advantage_weights[current_count]
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * weight
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * weight
        features['advantage_count_weight'] = weight
    else:
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * 0.5
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * 0.5
        features['advantage_count_weight'] = 0.5
    
    return features

def analyze_count_patterns(count_results):
    """
    Analyze patterns in count-specific swing rates.
    
    Args:
        count_results: Results from calculate_comprehensive_count_features
        
    Returns:
        dict: Analysis of count patterns
    """
    analysis = {}
    
    # Find most aggressive counts
    count_rates = [(count, result['overall_swing_rate']) for count, result in count_results.items()]
    count_rates.sort(key=lambda x: x[1], reverse=True)
    
    analysis['most_aggressive_counts'] = count_rates[:5]
    analysis['least_aggressive_counts'] = count_rates[-5:]
    
    # Find pitch type preferences by count
    pitch_preferences = {}
    for count, result in count_results.items():
        if result['pitch_type_swings']:
            # Find most swung at pitch type for this count
            best_pitch = max(result['pitch_type_swings'].items(), 
                           key=lambda x: x[1]['swing_rate'])
            pitch_preferences[count] = {
                'preferred_pitch': best_pitch[0],
                'swing_rate': best_pitch[1]['swing_rate']
            }
    
    analysis['pitch_preferences_by_count'] = pitch_preferences
    
    return analysis

if __name__ == "__main__":
    print("Loading career data for comprehensive count analysis...")
    
    # Load the career data
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"Loaded {len(df)} career pitches for Acuna Jr.")
        
        # Check required columns
        required_cols = ['balls', 'strikes', 'pitch_type', 'events']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            exit(1)
        
        # Calculate comprehensive count features
        print("\n" + "="*50)
        print("COMPREHENSIVE COUNT FEATURE ANALYSIS")
        print("="*50)
        
        count_features, count_results = calculate_comprehensive_count_features(df)
        
        # Analyze patterns
        analysis = analyze_count_patterns(count_results)
        
        print("\n" + "="*50)
        print("PATTERN ANALYSIS")
        print("="*50)
        
        print("\nMost aggressive counts:")
        for count, rate in analysis['most_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nLeast aggressive counts:")
        for count, rate in analysis['least_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nPitch preferences by count:")
        for count, pref in analysis['pitch_preferences_by_count'].items():
            print(f"  {count}: {pref['preferred_pitch']} ({pref['swing_rate']:.3f})")
        
        print(f"\nTotal features generated: {len(count_features)}")
        print("Sample features:")
        for i, (feature, value) in enumerate(list(count_features.items())[:10]):
            print(f"  {feature}: {value:.3f}")
        
    except FileNotFoundError:
        print("Error: Could not find ronald_acuna_jr_complete_career_statcast.csv")
        print("Please ensure the career data file exists in the current directory.")
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc() 
 
import numpy as np

def calculate_comprehensive_count_features(df):
    """
    Calculate comprehensive count-specific swing rates for all pitch types.
    
    Args:
        df: DataFrame with 'balls', 'strikes', 'pitch_type', and 'events' columns
        
    Returns:
        dict: Dictionary of comprehensive count features
    """
    # Make a copy to avoid fragmentation
    df = df.copy()
    
    # Define swing events - be more specific to avoid false positives
    # Only include events that clearly indicate a swing
    swing_events = [
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
        'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
        'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
        'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
        'sac_bunt_double_play', 'grounded_into_double_play'
    ]
    
    # Define non-swing events explicitly
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'walk', 'intent_walk', 'hit_by_pitch', 'strikeout', 'strikeout_double_play', 
        'strikeout_triple_play', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 
        'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home',
        'catcher_pickoff_1b', 'catcher_pickoff_2b', 'catcher_pickoff_3b', 
        'catcher_pickoff_caught_stealing_2b', 'catcher_pickoff_caught_stealing_3b', 
        'catcher_pickoff_caught_stealing_home', 'runner_double_play', 'batter_interference', 
        'catcher_interference', 'fan_interference', 'umpire_interference'
    ]
    
    # Define non-swing events to be more explicit
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_caught_stealing_2b',
        'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home', 'catcher_pickoff_1b',
        'catcher_pickoff_2b', 'catcher_pickoff_3b', 'catcher_pickoff_caught_stealing_2b',
        'catcher_pickoff_caught_stealing_3b', 'catcher_pickoff_caught_stealing_home',
        'runner_double_play', 'batter_interference', 'catcher_interference', 'umpire_interference'
    ]
    
    # Create swing indicator - be more careful about what constitutes a swing
    # First, check if events are in swing events
    df['is_swing'] = df['events'].isin(swing_events)
    
    # Then, explicitly mark non-swing events as False
    df.loc[df['events'].isin(non_swing_events), 'is_swing'] = False
    
    # Debug: Check what events we have and their swing classification
    unique_events = df['events'].unique()
    print(f"Found {len(unique_events)} unique events in dataset")
    print("Sample events:", unique_events[:10])
    
    # Show swing rate breakdown
    swing_count = df['is_swing'].sum()
    total_count = len(df)
    print(f"Total pitches: {total_count}")
    print(f"Swing pitches: {swing_count}")
    print(f"Overall swing rate: {swing_count/total_count:.3f}")
    
    # Show breakdown by event type
    event_swing_rates = df.groupby('events')['is_swing'].agg(['count', 'sum', 'mean']).sort_values('count', ascending=False)
    print("\nTop 10 events by frequency:")
    print(event_swing_rates.head(10))
    
    # Create count combinations
    df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    
    # Get all unique counts and pitch types
    all_counts = df['count'].unique()
    all_pitch_types = df['pitch_type'].unique()
    
    # Define advantage count weights for special handling
    advantage_weights = {
        '2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3
    }
    
    # Calculate comprehensive count features
    count_features = {}
    count_results = {}
    
    print("Calculating comprehensive count features...")
    print(f"Found {len(all_counts)} unique counts and {len(all_pitch_types)} pitch types")
    
    for count in all_counts:
        count_data = df[df['count'] == count]
        if len(count_data) == 0:
            continue
            
        # Overall swing rate for this count
        overall_swing_rate = count_data['is_swing'].mean()
        
        # Swing rates by pitch type
        pitch_type_swings = {}
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type):
                pitch_data = count_data[count_data['pitch_type'] == pitch_type]
                if len(pitch_data) >= 3:  # Lower threshold to include more data
                    swing_rate = pitch_data['is_swing'].mean()
                    pitch_type_swings[pitch_type] = {
                        'swing_rate': swing_rate,
                        'count': len(pitch_data)
                    }
        
        count_results[count] = {
            'overall_swing_rate': overall_swing_rate,
            'pitch_type_swings': pitch_type_swings,
            'total_pitches': len(count_data)
        }
        
        # Create features for this count
        count_features[f'{count}_overall_swing_rate'] = overall_swing_rate
        
        # Add pitch type specific rates
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type) and pitch_type in pitch_type_swings:
                swing_rate = pitch_type_swings[pitch_type]['swing_rate']
                count_features[f'{count}_{pitch_type}_swing_rate'] = swing_rate
                
                # Add weighted version for advantage counts
                if count in advantage_weights:
                    weight = advantage_weights[count]
                    count_features[f'{count}_{pitch_type}_weighted_swing_rate'] = swing_rate * weight
                    count_features[f'{count}_weighted_swing_rate'] = overall_swing_rate * weight
    
    print(f"Generated {len(count_features)} count-specific features")
    
    # Print summary of key findings
    print("\nKey Count Analysis:")
    for count in sorted(count_results.keys()):
        result = count_results[count]
        print(f"  {count}: {result['overall_swing_rate']:.3f} overall ({result['total_pitches']} pitches)")
        
        # Show top 3 pitch types for this count
        pitch_rates = [(pt, data['swing_rate']) for pt, data in result['pitch_type_swings'].items()]
        pitch_rates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (pitch_type, rate) in enumerate(pitch_rates[:3]):
            print(f"    {pitch_type}: {rate:.3f}")
    
    return count_features, count_results

def get_count_features_for_pitch(balls, strikes, pitch_type, count_features):
    """
    Get count-specific features for a particular pitch.
    
    Args:
        balls: Number of balls
        strikes: Number of strikes
        pitch_type: Type of pitch
        count_features: Dictionary of calculated count features
        
    Returns:
        dict: Features for the current count and pitch type
    """
    current_count = f"{balls}-{strikes}"
    
    features = {}
    
    # Get overall count swing rate
    overall_key = f'{current_count}_overall_swing_rate'
    if overall_key in count_features:
        features['current_count_overall_swing_rate'] = count_features[overall_key]
    else:
        features['current_count_overall_swing_rate'] = 0.5  # Default
    
    # Get pitch-specific count swing rate
    pitch_key = f'{current_count}_{pitch_type}_swing_rate'
    if pitch_key in count_features:
        features['current_count_pitch_swing_rate'] = count_features[pitch_key]
    else:
        features['current_count_pitch_swing_rate'] = features['current_count_overall_swing_rate']
    
    # Get weighted versions for advantage counts
    advantage_weights = {'2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3}
    
    if current_count in advantage_weights:
        weight = advantage_weights[current_count]
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * weight
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * weight
        features['advantage_count_weight'] = weight
    else:
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * 0.5
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * 0.5
        features['advantage_count_weight'] = 0.5
    
    return features

def analyze_count_patterns(count_results):
    """
    Analyze patterns in count-specific swing rates.
    
    Args:
        count_results: Results from calculate_comprehensive_count_features
        
    Returns:
        dict: Analysis of count patterns
    """
    analysis = {}
    
    # Find most aggressive counts
    count_rates = [(count, result['overall_swing_rate']) for count, result in count_results.items()]
    count_rates.sort(key=lambda x: x[1], reverse=True)
    
    analysis['most_aggressive_counts'] = count_rates[:5]
    analysis['least_aggressive_counts'] = count_rates[-5:]
    
    # Find pitch type preferences by count
    pitch_preferences = {}
    for count, result in count_results.items():
        if result['pitch_type_swings']:
            # Find most swung at pitch type for this count
            best_pitch = max(result['pitch_type_swings'].items(), 
                           key=lambda x: x[1]['swing_rate'])
            pitch_preferences[count] = {
                'preferred_pitch': best_pitch[0],
                'swing_rate': best_pitch[1]['swing_rate']
            }
    
    analysis['pitch_preferences_by_count'] = pitch_preferences
    
    return analysis

if __name__ == "__main__":
    print("Loading career data for comprehensive count analysis...")
    
    # Load the career data
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"Loaded {len(df)} career pitches for Acuna Jr.")
        
        # Check required columns
        required_cols = ['balls', 'strikes', 'pitch_type', 'events']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            exit(1)
        
        # Calculate comprehensive count features
        print("\n" + "="*50)
        print("COMPREHENSIVE COUNT FEATURE ANALYSIS")
        print("="*50)
        
        count_features, count_results = calculate_comprehensive_count_features(df)
        
        # Analyze patterns
        analysis = analyze_count_patterns(count_results)
        
        print("\n" + "="*50)
        print("PATTERN ANALYSIS")
        print("="*50)
        
        print("\nMost aggressive counts:")
        for count, rate in analysis['most_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nLeast aggressive counts:")
        for count, rate in analysis['least_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nPitch preferences by count:")
        for count, pref in analysis['pitch_preferences_by_count'].items():
            print(f"  {count}: {pref['preferred_pitch']} ({pref['swing_rate']:.3f})")
        
        print(f"\nTotal features generated: {len(count_features)}")
        print("Sample features:")
        for i, (feature, value) in enumerate(list(count_features.items())[:10]):
            print(f"  {feature}: {value:.3f}")
        
    except FileNotFoundError:
        print("Error: Could not find ronald_acuna_jr_complete_career_statcast.csv")
        print("Please ensure the career data file exists in the current directory.")
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc() 
import numpy as np

def calculate_comprehensive_count_features(df):
    """
    Calculate comprehensive count-specific swing rates for all pitch types.
    
    Args:
        df: DataFrame with 'balls', 'strikes', 'pitch_type', and 'events' columns
        
    Returns:
        dict: Dictionary of comprehensive count features
    """
    # Make a copy to avoid fragmentation
    df = df.copy()
    
    # Define swing events - be more specific to avoid false positives
    # Only include events that clearly indicate a swing
    swing_events = [
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
        'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
        'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
        'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
        'sac_bunt_double_play', 'grounded_into_double_play'
    ]
    
    # Define non-swing events explicitly
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'walk', 'intent_walk', 'hit_by_pitch', 'strikeout', 'strikeout_double_play', 
        'strikeout_triple_play', 'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 
        'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home',
        'catcher_pickoff_1b', 'catcher_pickoff_2b', 'catcher_pickoff_3b', 
        'catcher_pickoff_caught_stealing_2b', 'catcher_pickoff_caught_stealing_3b', 
        'catcher_pickoff_caught_stealing_home', 'runner_double_play', 'batter_interference', 
        'catcher_interference', 'fan_interference', 'umpire_interference'
    ]
    
    # Define non-swing events to be more explicit
    non_swing_events = [
        'ball', 'called_strike', 'blocked_ball', 'pitchout', 'wild_pitch', 'passed_ball',
        'pickoff_1b', 'pickoff_2b', 'pickoff_3b', 'pickoff_caught_stealing_2b',
        'pickoff_caught_stealing_3b', 'pickoff_caught_stealing_home', 'catcher_pickoff_1b',
        'catcher_pickoff_2b', 'catcher_pickoff_3b', 'catcher_pickoff_caught_stealing_2b',
        'catcher_pickoff_caught_stealing_3b', 'catcher_pickoff_caught_stealing_home',
        'runner_double_play', 'batter_interference', 'catcher_interference', 'umpire_interference'
    ]
    
    # Create swing indicator - be more careful about what constitutes a swing
    # First, check if events are in swing events
    df['is_swing'] = df['events'].isin(swing_events)
    
    # Then, explicitly mark non-swing events as False
    df.loc[df['events'].isin(non_swing_events), 'is_swing'] = False
    
    # Debug: Check what events we have and their swing classification
    unique_events = df['events'].unique()
    print(f"Found {len(unique_events)} unique events in dataset")
    print("Sample events:", unique_events[:10])
    
    # Show swing rate breakdown
    swing_count = df['is_swing'].sum()
    total_count = len(df)
    print(f"Total pitches: {total_count}")
    print(f"Swing pitches: {swing_count}")
    print(f"Overall swing rate: {swing_count/total_count:.3f}")
    
    # Show breakdown by event type
    event_swing_rates = df.groupby('events')['is_swing'].agg(['count', 'sum', 'mean']).sort_values('count', ascending=False)
    print("\nTop 10 events by frequency:")
    print(event_swing_rates.head(10))
    
    # Create count combinations
    df['count'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    
    # Get all unique counts and pitch types
    all_counts = df['count'].unique()
    all_pitch_types = df['pitch_type'].unique()
    
    # Define advantage count weights for special handling
    advantage_weights = {
        '2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3
    }
    
    # Calculate comprehensive count features
    count_features = {}
    count_results = {}
    
    print("Calculating comprehensive count features...")
    print(f"Found {len(all_counts)} unique counts and {len(all_pitch_types)} pitch types")
    
    for count in all_counts:
        count_data = df[df['count'] == count]
        if len(count_data) == 0:
            continue
            
        # Overall swing rate for this count
        overall_swing_rate = count_data['is_swing'].mean()
        
        # Swing rates by pitch type
        pitch_type_swings = {}
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type):
                pitch_data = count_data[count_data['pitch_type'] == pitch_type]
                if len(pitch_data) >= 3:  # Lower threshold to include more data
                    swing_rate = pitch_data['is_swing'].mean()
                    pitch_type_swings[pitch_type] = {
                        'swing_rate': swing_rate,
                        'count': len(pitch_data)
                    }
        
        count_results[count] = {
            'overall_swing_rate': overall_swing_rate,
            'pitch_type_swings': pitch_type_swings,
            'total_pitches': len(count_data)
        }
        
        # Create features for this count
        count_features[f'{count}_overall_swing_rate'] = overall_swing_rate
        
        # Add pitch type specific rates
        for pitch_type in all_pitch_types:
            if pd.notna(pitch_type) and pitch_type in pitch_type_swings:
                swing_rate = pitch_type_swings[pitch_type]['swing_rate']
                count_features[f'{count}_{pitch_type}_swing_rate'] = swing_rate
                
                # Add weighted version for advantage counts
                if count in advantage_weights:
                    weight = advantage_weights[count]
                    count_features[f'{count}_{pitch_type}_weighted_swing_rate'] = swing_rate * weight
                    count_features[f'{count}_weighted_swing_rate'] = overall_swing_rate * weight
    
    print(f"Generated {len(count_features)} count-specific features")
    
    # Print summary of key findings
    print("\nKey Count Analysis:")
    for count in sorted(count_results.keys()):
        result = count_results[count]
        print(f"  {count}: {result['overall_swing_rate']:.3f} overall ({result['total_pitches']} pitches)")
        
        # Show top 3 pitch types for this count
        pitch_rates = [(pt, data['swing_rate']) for pt, data in result['pitch_type_swings'].items()]
        pitch_rates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (pitch_type, rate) in enumerate(pitch_rates[:3]):
            print(f"    {pitch_type}: {rate:.3f}")
    
    return count_features, count_results

def get_count_features_for_pitch(balls, strikes, pitch_type, count_features):
    """
    Get count-specific features for a particular pitch.
    
    Args:
        balls: Number of balls
        strikes: Number of strikes
        pitch_type: Type of pitch
        count_features: Dictionary of calculated count features
        
    Returns:
        dict: Features for the current count and pitch type
    """
    current_count = f"{balls}-{strikes}"
    
    features = {}
    
    # Get overall count swing rate
    overall_key = f'{current_count}_overall_swing_rate'
    if overall_key in count_features:
        features['current_count_overall_swing_rate'] = count_features[overall_key]
    else:
        features['current_count_overall_swing_rate'] = 0.5  # Default
    
    # Get pitch-specific count swing rate
    pitch_key = f'{current_count}_{pitch_type}_swing_rate'
    if pitch_key in count_features:
        features['current_count_pitch_swing_rate'] = count_features[pitch_key]
    else:
        features['current_count_pitch_swing_rate'] = features['current_count_overall_swing_rate']
    
    # Get weighted versions for advantage counts
    advantage_weights = {'2-0': 0.9, '3-1': 0.85, '2-1': 0.7, '1-0': 0.6, '3-0': 0.3}
    
    if current_count in advantage_weights:
        weight = advantage_weights[current_count]
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * weight
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * weight
        features['advantage_count_weight'] = weight
    else:
        features['current_count_weighted_swing_rate'] = features['current_count_overall_swing_rate'] * 0.5
        features['current_count_pitch_weighted_swing_rate'] = features['current_count_pitch_swing_rate'] * 0.5
        features['advantage_count_weight'] = 0.5
    
    return features

def analyze_count_patterns(count_results):
    """
    Analyze patterns in count-specific swing rates.
    
    Args:
        count_results: Results from calculate_comprehensive_count_features
        
    Returns:
        dict: Analysis of count patterns
    """
    analysis = {}
    
    # Find most aggressive counts
    count_rates = [(count, result['overall_swing_rate']) for count, result in count_results.items()]
    count_rates.sort(key=lambda x: x[1], reverse=True)
    
    analysis['most_aggressive_counts'] = count_rates[:5]
    analysis['least_aggressive_counts'] = count_rates[-5:]
    
    # Find pitch type preferences by count
    pitch_preferences = {}
    for count, result in count_results.items():
        if result['pitch_type_swings']:
            # Find most swung at pitch type for this count
            best_pitch = max(result['pitch_type_swings'].items(), 
                           key=lambda x: x[1]['swing_rate'])
            pitch_preferences[count] = {
                'preferred_pitch': best_pitch[0],
                'swing_rate': best_pitch[1]['swing_rate']
            }
    
    analysis['pitch_preferences_by_count'] = pitch_preferences
    
    return analysis

if __name__ == "__main__":
    print("Loading career data for comprehensive count analysis...")
    
    # Load the career data
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"Loaded {len(df)} career pitches for Acuna Jr.")
        
        # Check required columns
        required_cols = ['balls', 'strikes', 'pitch_type', 'events']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            exit(1)
        
        # Calculate comprehensive count features
        print("\n" + "="*50)
        print("COMPREHENSIVE COUNT FEATURE ANALYSIS")
        print("="*50)
        
        count_features, count_results = calculate_comprehensive_count_features(df)
        
        # Analyze patterns
        analysis = analyze_count_patterns(count_results)
        
        print("\n" + "="*50)
        print("PATTERN ANALYSIS")
        print("="*50)
        
        print("\nMost aggressive counts:")
        for count, rate in analysis['most_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nLeast aggressive counts:")
        for count, rate in analysis['least_aggressive_counts']:
            print(f"  {count}: {rate:.3f}")
        
        print("\nPitch preferences by count:")
        for count, pref in analysis['pitch_preferences_by_count'].items():
            print(f"  {count}: {pref['preferred_pitch']} ({pref['swing_rate']:.3f})")
        
        print(f"\nTotal features generated: {len(count_features)}")
        print("Sample features:")
        for i, (feature, value) in enumerate(list(count_features.items())[:10]):
            print(f"  {feature}: {value:.3f}")
        
    except FileNotFoundError:
        print("Error: Could not find ronald_acuna_jr_complete_career_statcast.csv")
        print("Please ensure the career data file exists in the current directory.")
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc() 
 
 
 
 
 
 
 
 
 
 
 
 
 