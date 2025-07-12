import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from comprehensive_count_features import calculate_comprehensive_count_features, get_count_features_for_pitch, analyze_count_patterns

def load_holdout_data():
    """Load the holdout data for testing"""
    try:
        df = pd.read_csv('ronald_acuna_jr_holdout_statcast.csv')
        print(f"✓ Loaded holdout data with {len(df)} pitches")
        return df
    except Exception as e:
        print(f"✗ Error loading holdout data: {e}")
        return None

def load_outcome_model():
    """Load the trained outcome prediction model"""
    try:
        with open('sequential_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        outcome_model = models.get('swing_outcome_model')
        outcome_preprocessor = models.get('swing_outcome_preprocessor')
        outcome_threshold = models.get('outcome_threshold', 0.5)
        
        if outcome_model is None:
            print("✗ Outcome model not found in model file")
            return None, None, None
        
        print("✓ Loaded outcome prediction model")
        return outcome_model, outcome_preprocessor, outcome_threshold
        
    except Exception as e:
        print(f"✗ Error loading outcome model: {e}")
        return None, None, None

def load_career_data():
    """Load career data for both hitter and pitcher to calculate actual features"""
    try:
        # Load hitter career data
        hitter_df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"✓ Loaded hitter career data with {len(hitter_df)} pitches")
        
        # Load pitcher career data  
        pitcher_df = pd.read_csv('sandy_alcantara_complete_career_statcast.csv')
        print(f"✓ Loaded pitcher career data with {len(pitcher_df)} pitches")
        
        return hitter_df, pitcher_df
    except Exception as e:
        print(f"✗ Error loading career data: {e}")
        return None, None

def calculate_hitter_features(hitter_df):
    """Calculate actual hitter-specific features from career data"""
    print("Calculating hitter features from career data...")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
                   'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
                   'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
                   'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
                   'sac_bunt_double_play', 'grounded_into_double_play', 'batter_interference',
                   'catcher_interference', 'fan_interference', 'strikeout', 'strikeout_double_play', 
                   'strikeout_triple_play', 'walk', 'intent_walk', 'hit_by_pitch',
                   'sacrifice_bunt_double_play', 'sacrifice_bunt_triple_play', 'umpire_interference']
    
    swing_df = hitter_df[hitter_df['events'].isin(swing_events)].copy()
    
    if len(swing_df) == 0:
        print("No swing events found in hitter career data")
        return {}
    
    # Calculate movement magnitude for hitter data
    if 'horizontal_break' in swing_df.columns and 'vertical_break' in swing_df.columns:
        swing_df['movement_magnitude'] = np.sqrt(swing_df['horizontal_break']**2 + swing_df['vertical_break']**2)
    elif 'pfx_x' in swing_df.columns and 'pfx_z' in swing_df.columns:
        swing_df['movement_magnitude'] = np.sqrt(swing_df['pfx_x']**2 + swing_df['pfx_z']**2)
    else:
        swing_df['movement_magnitude'] = swing_df['release_spin_rate'] / 1000
    
    # Calculate actual swing rates by pitch type
    fastball_types = ['FF', 'SI', 'FC', 'FT']
    breaking_types = ['SL', 'CU', 'KC', 'SV']
    offspeed_types = ['CH', 'FS', 'FO']
    
    fastball_swings = swing_df[swing_df['pitch_type'].isin(fastball_types)]
    breaking_swings = swing_df[swing_df['pitch_type'].isin(breaking_types)]
    offspeed_swings = swing_df[swing_df['pitch_type'].isin(offspeed_types)]
    
    hitter_features = {}
    
    # Pitch type swing rates
    hitter_features['acuna_fastball_swing_rate'] = len(fastball_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_breaking_swing_rate'] = len(breaking_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_offspeed_swing_rate'] = len(offspeed_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Zone swing rates
    zone_swings = swing_df[((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5))]
    outside_swings = swing_df[swing_df['plate_x'].abs() > 0.7]
    high_swings = swing_df[swing_df['plate_z'] > 3.0]
    low_swings = swing_df[swing_df['plate_z'] < 2.0]
    
    hitter_features['acuna_zone_swing_rate'] = len(zone_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_outside_swing_rate'] = len(outside_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_high_swing_rate'] = len(high_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_low_swing_rate'] = len(low_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Count-based swing rates
    ahead_swings = swing_df[swing_df['balls'] > swing_df['strikes']]
    behind_swings = swing_df[swing_df['strikes'] > swing_df['balls']]
    two_strikes_swings = swing_df[swing_df['strikes'] >= 2]
    full_count_swings = swing_df[(swing_df['balls'] == 3) & (swing_df['strikes'] == 2)]
    
    hitter_features['acuna_ahead_swing_rate'] = len(ahead_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_behind_swing_rate'] = len(behind_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_two_strikes_swing_rate'] = len(two_strikes_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_full_count_swing_rate'] = len(full_count_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Velocity-based swing rates
    high_vel_swings = swing_df[swing_df['release_speed'] > 95]
    low_vel_swings = swing_df[swing_df['release_speed'] < 85]
    
    hitter_features['acuna_high_vel_swing_rate'] = len(high_vel_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_low_vel_swing_rate'] = len(low_vel_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Movement-based swing rates
    high_movement_swings = swing_df[swing_df['movement_magnitude'] > 8]
    low_movement_swings = swing_df[swing_df['movement_magnitude'] < 4]
    
    hitter_features['acuna_high_movement_swing_rate'] = len(high_movement_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_low_movement_swing_rate'] = len(low_movement_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Situational swing rates
    # Calculate count_total from balls and strikes if not present
    if 'count_total' not in swing_df.columns:
        swing_df['count_total'] = swing_df['balls'] + swing_df['strikes']
    
    first_pitch_swings = swing_df[swing_df['count_total'] == 0]
    last_pitch_swings = swing_df[swing_df['count_total'] >= 5]
    
    hitter_features['acuna_first_pitch_swing_rate'] = len(first_pitch_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_last_pitch_swing_rate'] = len(last_pitch_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Calculate zone-specific swing rates from actual data
    zone_heart_swings = swing_df[((swing_df['plate_x'].abs() <= 0.5) & (swing_df['plate_z'] >= 2.0) & (swing_df['plate_z'] <= 3.0))]
    zone_corner_swings = swing_df[((swing_df['plate_x'].abs() >= 0.7) | (swing_df['plate_z'] >= 3.5) | (swing_df['plate_z'] <= 1.5))]
    zone_shadow_swings = swing_df[((swing_df['plate_x'].abs() <= 1.0) & (swing_df['plate_z'] >= 1.2) & (swing_df['plate_z'] <= 3.8)) & ~((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5))]
    
    hitter_features['acuna_zone_heart_swing_rate'] = len(zone_heart_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_zone_corner_swing_rate'] = len(zone_corner_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_zone_shadow_swing_rate'] = len(zone_shadow_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # NEW: Zone-specific contact rate features (for outcome prediction)
    # Calculate contact rates by zone - hits divided by swings in each zone
    hit_events = ['single', 'double', 'triple', 'home_run']
    
    # Zone heart contact rate
    zone_heart_hits = swing_df[((swing_df['plate_x'].abs() <= 0.5) & (swing_df['plate_z'] >= 2.0) & (swing_df['plate_z'] <= 3.0)) & 
                               (swing_df['events'].isin(hit_events))]
    hitter_features['acuna_zone_heart_hit_rate'] = len(zone_heart_hits) / len(zone_heart_swings) if len(zone_heart_swings) > 0 else 0.0
    
    # Zone corner contact rate
    zone_corner_hits = swing_df[((swing_df['plate_x'].abs() >= 0.7) | (swing_df['plate_z'] >= 3.5) | (swing_df['plate_z'] <= 1.5)) & 
                                (swing_df['events'].isin(hit_events))]
    hitter_features['acuna_zone_corner_hit_rate'] = len(zone_corner_hits) / len(zone_corner_swings) if len(zone_corner_swings) > 0 else 0.0
    
    # Zone shadow contact rate
    zone_shadow_hits = swing_df[((swing_df['plate_x'].abs() <= 1.0) & (swing_df['plate_z'] >= 1.2) & (swing_df['plate_z'] <= 3.8)) & 
                                ~((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5)) & 
                                (swing_df['events'].isin(hit_events))]
    hitter_features['acuna_zone_shadow_hit_rate'] = len(zone_shadow_hits) / len(zone_shadow_swings) if len(zone_shadow_swings) > 0 else 0.0
    
    # Overall zone contact rate
    zone_hits = swing_df[((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5)) & 
                         (swing_df['events'].isin(hit_events))]
    hitter_features['acuna_zone_hit_rate'] = len(zone_hits) / len(zone_swings) if len(zone_swings) > 0 else 0.0
    
    # Outside zone contact rate
    outside_hits = swing_df[(swing_df['plate_x'].abs() > 0.7) & (swing_df['events'].isin(hit_events))]
    hitter_features['acuna_outside_hit_rate'] = len(outside_hits) / len(outside_swings) if len(outside_swings) > 0 else 0.0
    
    # Calculate location-specific swing rates
    location_extreme_swings = swing_df[((swing_df['plate_x'].abs() > 1.2) | (swing_df['plate_z'] > 4.0) | (swing_df['plate_z'] < 1.0))]
    location_heart_swings = swing_df[((swing_df['plate_x'].abs() <= 0.5) & (swing_df['plate_z'] >= 2.0) & (swing_df['plate_z'] <= 3.0))]
    
    hitter_features['acuna_location_extreme_swing_rate'] = len(location_extreme_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_location_heart_swing_rate'] = len(location_heart_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Calculate situational swing rates
    pressure_swings = swing_df[swing_df['strikes'] >= 2]
    opportunity_swings = swing_df[swing_df['balls'] > swing_df['strikes']]
    
    hitter_features['acuna_pressure_swing_rate'] = len(pressure_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_opportunity_swing_rate'] = len(opportunity_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
    # Calculate velocity change swing rates (using velocity differences)
    if 'release_speed' in swing_df.columns:
        avg_velocity = swing_df['release_speed'].mean()
        velocity_drop_swings = swing_df[swing_df['release_speed'] < (avg_velocity - 5)]
        velocity_surge_swings = swing_df[swing_df['release_speed'] > (avg_velocity + 5)]
        
        hitter_features['acuna_velocity_drop_swing_rate'] = len(velocity_drop_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        hitter_features['acuna_velocity_surge_swing_rate'] = len(velocity_surge_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    else:
        hitter_features['acuna_velocity_drop_swing_rate'] = 0.0
        hitter_features['acuna_velocity_surge_swing_rate'] = 0.0
    
    # Calculate pitch type change swing rate (would need sequence data, using pitch type distribution for now)
    if 'pitch_type' in swing_df.columns:
        pitch_type_counts = swing_df['pitch_type'].value_counts()
        most_common_pitch = pitch_type_counts.index[0] if len(pitch_type_counts) > 0 else 'FF'
        non_most_common_swings = swing_df[swing_df['pitch_type'] != most_common_pitch]
        hitter_features['acuna_pitch_type_change_swing_rate'] = len(non_most_common_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    else:
        hitter_features['acuna_pitch_type_change_swing_rate'] = 0.0
    
    # Calculate inning and game situation swing rates (if data available)
    if 'inning' in swing_df.columns:
        late_inning_swings = swing_df[swing_df['inning'] >= 7]
        hitter_features['acuna_late_inning_swing_rate'] = len(late_inning_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    else:
        hitter_features['acuna_late_inning_swing_rate'] = 0.0
    
    if 'home_score' in swing_df.columns and 'away_score' in swing_df.columns:
        close_game_swings = swing_df[abs(swing_df['home_score'] - swing_df['away_score']) <= 2]
        hitter_features['acuna_close_game_swing_rate'] = len(close_game_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    else:
        hitter_features['acuna_close_game_swing_rate'] = 0.0
    
    # Calculate comprehensive count features dynamically from career data
    print("Calculating comprehensive count features from career data...")
    count_features, count_results = calculate_comprehensive_count_features(hitter_df)
    
    # Add count features to hitter features
    for feature_name, feature_value in count_features.items():
        hitter_features[f'acuna_{feature_name}'] = feature_value
    
    print("Comprehensive count analysis from career data:")
    for count in sorted(count_results.keys()):
        result = count_results[count]
        print(f"  {count}: {result['overall_swing_rate']:.3f} overall ({result['total_pitches']} pitches)")
        
        # Show top 3 pitch types for this count
        pitch_rates = [(pt, data['swing_rate']) for pt, data in result['pitch_type_swings'].items()]
        pitch_rates.sort(key=lambda x: x[1], reverse=True)
        
        for i, (pitch_type, rate) in enumerate(pitch_rates[:3]):
            print(f"    {pitch_type}: {rate:.3f}")
    
    # Analyze patterns
    analysis = analyze_count_patterns(count_results)
    print("\nKey Insights:")
    print("Most aggressive counts:")
    for count, rate in analysis['most_aggressive_counts'][:3]:
        print(f"  {count}: {rate:.3f}")
    print("Least aggressive counts:")
    for count, rate in analysis['least_aggressive_counts'][:3]:
        print(f"  {count}: {rate:.3f}")
    
    print(f"Calculated {len(hitter_features)} hitter features")
    return hitter_features

def calculate_pitcher_features(pitcher_df):
    """Calculate actual pitcher-specific features from career data"""
    print("Calculating pitcher features from career data...")
    
    # Calculate movement magnitude for pitcher data
    if 'horizontal_break' in pitcher_df.columns and 'vertical_break' in pitcher_df.columns:
        pitcher_df['movement_magnitude'] = np.sqrt(pitcher_df['horizontal_break']**2 + pitcher_df['vertical_break']**2)
    elif 'pfx_x' in pitcher_df.columns and 'pfx_z' in pitcher_df.columns:
        pitcher_df['movement_magnitude'] = np.sqrt(pitcher_df['pfx_x']**2 + pitcher_df['pfx_z']**2)
    else:
        pitcher_df['movement_magnitude'] = pitcher_df['release_spin_rate'] / 1000
    
    pitcher_features = {}
    
    # Average velocity and movement
    pitcher_features['avg_velocity'] = pitcher_df['release_speed'].mean()
    pitcher_features['avg_movement'] = pitcher_df['movement_magnitude'].mean()
    pitcher_features['velocity_std'] = pitcher_df['release_speed'].std()
    pitcher_features['movement_std'] = pitcher_df['movement_magnitude'].std()
    
    # Pitch type averages
    fastball_types = ['FF', 'SI', 'FC', 'FT']
    breaking_types = ['SL', 'CU', 'KC', 'SV']
    offspeed_types = ['CH', 'FS', 'FO']
    
    fastballs = pitcher_df[pitcher_df['pitch_type'].isin(fastball_types)]
    breaking_balls = pitcher_df[pitcher_df['pitch_type'].isin(breaking_types)]
    offspeed = pitcher_df[pitcher_df['pitch_type'].isin(offspeed_types)]
    
    if len(fastballs) > 0:
        pitcher_features['fastball_avg_velocity'] = fastballs['release_speed'].mean()
        pitcher_features['fastball_avg_movement'] = fastballs['movement_magnitude'].mean()
    
    if len(breaking_balls) > 0:
        pitcher_features['breaking_avg_velocity'] = breaking_balls['release_speed'].mean()
        pitcher_features['breaking_avg_movement'] = breaking_balls['movement_magnitude'].mean()
    
    if len(offspeed) > 0:
        pitcher_features['offspeed_avg_velocity'] = offspeed['release_speed'].mean()
        pitcher_features['offspeed_avg_movement'] = offspeed['movement_magnitude'].mean()
    
    print(f"Calculated {len(pitcher_features)} pitcher features")
    return pitcher_features

def prepare_outcome_features(df, hitter_features=None, pitcher_features=None):
    """Prepare features for outcome prediction - focusing only on hitting events"""
    # Focus only on hitting events (not defensive events like stolen bases, pickoffs, etc.)
    hitting_events = [
        'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
        'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
        'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
        'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
        'sac_bunt_double_play', 'grounded_into_double_play', 'batter_interference',
        'catcher_interference', 'fan_interference', 'strikeout', 'strikeout_double_play', 
        'strikeout_triple_play', 'walk', 'intent_walk', 'hit_by_pitch',
        'sacrifice_bunt_double_play', 'sacrifice_bunt_triple_play', 'umpire_interference'
    ]
    
    # Filter to only hitting events
    swing_df = df[df['events'].isin(hitting_events)].copy()
    print(f"Found {len(swing_df)} hitting events for outcome prediction")
    
    # FIX ZONE 0.0 ISSUE - Validate and clean zone data
    print(f"\nZone validation:")
    print(f"Before cleaning: {len(swing_df)} pitches")
    
    # Check for invalid zones - including NaN values
    invalid_zones = swing_df[(swing_df['zone'] <= 0) | (swing_df['zone'].isna())].copy()
    print(f"Pitches with invalid zones (≤0 or NaN): {len(invalid_zones)}")
    
    if len(invalid_zones) > 0:
        print("Sample invalid zone pitches:")
        print(invalid_zones[['plate_x', 'plate_z', 'zone', 'events']].head())
        print(f"Zone value distribution in invalid zones:")
        print(invalid_zones['zone'].value_counts().head())
    
    # Fix invalid zones by recalculating from plate_x and plate_z
    def fix_zone(row):
        if pd.isna(row['zone']) or row['zone'] <= 0:
            # Recalculate zone from plate coordinates
            plate_x, plate_z = row['plate_x'], row['plate_z']
            
            # Standard zone calculation
            if pd.isna(plate_x) or pd.isna(plate_z):
                return 1  # Default to zone 1 if coordinates missing
            elif abs(plate_x) <= 0.7 and 1.5 <= plate_z <= 3.5:
                # In strike zone
                if plate_x >= 0:
                    if plate_z >= 2.5:
                        return 1  # High inside
                    else:
                        return 3  # Low inside
                else:
                    if plate_z >= 2.5:
                        return 2  # High outside
                    else:
                        return 4  # Low outside
            else:
                # Outside strike zone
                if plate_x >= 0:
                    if plate_z >= 2.5:
                        return 5  # High inside out
                    else:
                        return 7  # Low inside out
                else:
                    if plate_z >= 2.5:
                        return 6  # High outside out
                    else:
                        return 8  # Low outside out
        else:
            return row['zone']
    
    # Apply zone fixing ONLY to zones that need fixing (≤ 0 or NaN)
    # Don't override valid zone data from Statcast
    invalid_zone_mask = (swing_df['zone'] <= 0) | (swing_df['zone'].isna())
    if invalid_zone_mask.sum() > 0:
        print(f"Fixing {invalid_zone_mask.sum()} invalid zones...")
        swing_df.loc[invalid_zone_mask, 'zone'] = swing_df[invalid_zone_mask].apply(fix_zone, axis=1)
    else:
        print("No invalid zones found - using original Statcast zone data")
    
    # Remove any remaining invalid zones
    swing_df = swing_df[swing_df['zone'] > 0]
    print(f"After cleaning: {len(swing_df)} pitches")
    
    if len(swing_df) == 0:
        print("No valid hitting events found after zone cleaning")
        return None, None
    
    # Create outcome labels - focus on the three main outcomes
    outcome_mapping = {
        'swinging_strike': 'whiff',
        'swinging_strike_blocked': 'whiff',
        'foul': 'field_out',
        'foul_tip': 'whiff',
        'foul_bunt': 'field_out',
        'missed_bunt': 'whiff',
        'bunt_foul_tip': 'whiff',
        'single': 'hit_safely',
        'double': 'hit_safely',
        'triple': 'hit_safely',
        'home_run': 'hit_safely',
        'groundout': 'field_out',
        'force_out': 'field_out',
        'double_play': 'field_out',
        'triple_play': 'field_out',
        'sac_fly': 'field_out',
        'sac_bunt': 'field_out',
        'field_error': 'field_out',
        'fielders_choice': 'field_out',
        'fielders_choice_out': 'field_out',
        'sac_fly_double_play': 'field_out',
        'sac_bunt_double_play': 'field_out',
        'grounded_into_double_play': 'field_out',
        'batter_interference': 'field_out',
        'catcher_interference': 'field_out',
        'fan_interference': 'field_out',
        'strikeout': 'whiff',
        'strikeout_double_play': 'whiff',
        'strikeout_triple_play': 'whiff',
        'walk': 'field_out',
        'intent_walk': 'field_out',
        'hit_by_pitch': 'field_out',
        'sacrifice_bunt_double_play': 'field_out',
        'sacrifice_bunt_triple_play': 'field_out',
        'umpire_interference': 'field_out'
    }
    
    # Map events to outcomes
    swing_df['outcome'] = swing_df['events'].map(outcome_mapping)
    
    # Remove any rows where we couldn't map the outcome
    swing_df = swing_df.dropna(subset=['outcome'])
    print(f"After outcome mapping: {len(swing_df)} valid hitting events")
    
    # Check outcome distribution
    outcome_counts = swing_df['outcome'].value_counts()
    print(f"\nOutcome distribution:")
    for outcome, count in outcome_counts.items():
        print(f"  {outcome}: {count} ({count/len(swing_df)*100:.1f}%)")
    
    # Debug: Check for missing pitch type data
    print(f"\nPitch type data quality:")
    print(f"  Total pitches: {len(swing_df)}")
    print(f"  Missing pitch_type: {swing_df['pitch_type'].isna().sum()}")
    print(f"  Unique pitch types: {swing_df['pitch_type'].nunique()}")
    print(f"  Pitch type distribution:")
    pitch_type_counts = swing_df['pitch_type'].value_counts()
    for pitch_type, count in pitch_type_counts.head(10).items():
        print(f"    {pitch_type}: {count}")
    
    # Debug: Check zone data quality
    print(f"\nZone data quality:")
    print(f"  Unique zones: {swing_df['zone'].nunique()}")
    print(f"  Zone distribution:")
    zone_counts = swing_df['zone'].value_counts()
    for zone, count in zone_counts.head(10).items():
        print(f"    Zone {zone}: {count}")
    
    # Now add comprehensive feature engineering using actual career data
    print(f"Before engineer_features: {len(swing_df)} pitches")
    swing_df = engineer_features(swing_df, hitter_features, pitcher_features)
    print(f"After engineer_features: {len(swing_df)} pitches")
    
    return swing_df, outcome_counts

def engineer_features(df, hitter_features=None, pitcher_features=None):
    """Engineer all features needed for the outcome model using actual career data"""
    print(f"Engineering features using actual career data...")
    print(f"Input dataset size: {len(df)} pitches")
    
    # Check for any data expansion
    original_size = len(df)
    
    # Basic movement calculation - try different approaches
    if 'horizontal_break' in df.columns and 'vertical_break' in df.columns:
        df['movement_magnitude'] = np.sqrt(df['horizontal_break']**2 + df['vertical_break']**2)
    elif 'pfx_x' in df.columns and 'pfx_z' in df.columns:
        df['movement_magnitude'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    else:
        # Fallback: use spin rate as proxy for movement
        df['movement_magnitude'] = df['release_spin_rate'] / 1000  # Normalize spin rate
    
    # Ensure we have basic features
    required_features = ['release_speed', 'release_spin_rate', 'plate_x', 'plate_z', 'zone', 'balls', 'strikes']
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0.0
    
    # Calculate zone-related features
    df['zone_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)  # Distance from zone center
    df['zone_center_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
    
    # Count-based features
    df['count_total'] = df['balls'] + df['strikes']
    df['count_pressure'] = df['strikes'] / 3.0  # Higher pressure with more strikes
    df['behind_in_count'] = (df['balls'] < df['strikes']).astype(int)
    df['ahead_in_count'] = (df['balls'] > df['strikes']).astype(int)
    df['two_strikes'] = (df['strikes'] >= 2).astype(int)
    df['three_balls'] = (df['balls'] >= 3).astype(int)
    df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
    
    # Pitch type features
    df['is_fastball'] = df['pitch_type'].isin(['FF', 'SI', 'FC', 'FT']).astype(int)
    df['is_breaking_ball'] = df['pitch_type'].isin(['SL', 'CU', 'KC', 'SV']).astype(int)
    df['is_offspeed'] = df['pitch_type'].isin(['CH', 'FS', 'FO']).astype(int)
    
    # Location features
    df['in_strike_zone'] = ((df['plate_x'].abs() <= 0.7) & (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5)).astype(int)
    df['far_from_zone'] = (df['zone_distance'] > 1.0).astype(int)
    df['high_pitch'] = (df['plate_z'] > 3.0).astype(int)
    df['low_pitch'] = (df['plate_z'] < 2.0).astype(int)
    df['inside_pitch'] = (df['plate_x'] < -0.5).astype(int)
    df['outside_pitch'] = (df['plate_x'] > 0.5).astype(int)
    
    # Velocity features
    df['high_velocity'] = (df['release_speed'] > 95).astype(int)
    df['low_velocity'] = (df['release_speed'] < 85).astype(int)
    df['velocity_diff_from_avg'] = df['release_speed'] - df['release_speed'].mean()
    
    # Movement features
    df['high_movement'] = (df['movement_magnitude'] > 8).astype(int)
    df['velocity_movement_ratio'] = df['release_speed'] / (df['movement_magnitude'] + 0.1)
    
    # Interaction features
    df['zone_distance_x_count_pressure'] = df['zone_distance'] * df['count_pressure']
    df['movement_x_count_pressure'] = df['movement_magnitude'] * df['count_pressure']
    df['in_zone_x_two_strikes'] = df['in_strike_zone'] * df['two_strikes']
    df['far_from_zone_x_ahead'] = df['far_from_zone'] * df['ahead_in_count']
    
    # Count advantage features
    df['hitters_count'] = (df['balls'] > df['strikes']).astype(int)
    df['pitchers_count'] = (df['strikes'] > df['balls']).astype(int)
    df['neutral_count'] = (df['balls'] == df['strikes']).astype(int)
    
    # Situational features
    df['pressure_situation'] = df['two_strikes'].astype(int)
    df['must_swing'] = df['two_strikes'].astype(int)
    df['can_take'] = (df['balls'] < 2).astype(int)
    
    # Zone edge features
    df['zone_edge_distance'] = np.minimum(
        np.abs(df['plate_x'] - 0.7),
        np.abs(df['plate_x'] + 0.7)
    )
    df['zone_top_distance'] = np.abs(df['plate_z'] - 3.5)
    df['zone_bottom_distance'] = np.abs(df['plate_z'] - 1.5)
    df['closest_zone_edge'] = np.minimum(
        df['zone_edge_distance'],
        np.minimum(df['zone_top_distance'], df['zone_bottom_distance'])
    )
    
    # Zone quadrants
    df['zone_quadrant'] = 'unknown'
    df.loc[(df['plate_x'] >= 0) & (df['plate_z'] >= 2.5), 'zone_quadrant'] = 'high_inside'
    df.loc[(df['plate_x'] < 0) & (df['plate_z'] >= 2.5), 'zone_quadrant'] = 'high_outside'
    df.loc[(df['plate_x'] >= 0) & (df['plate_z'] < 2.5), 'zone_quadrant'] = 'low_inside'
    df.loc[(df['plate_x'] < 0) & (df['plate_z'] < 2.5), 'zone_quadrant'] = 'low_outside'
    
    # Location quadrants
    df['location_quadrant'] = 'unknown'
    df.loc[(df['plate_x'] >= 0) & (df['plate_z'] >= 2.5), 'location_quadrant'] = 'high_inside'
    df.loc[(df['plate_x'] < 0) & (df['plate_z'] >= 2.5), 'location_quadrant'] = 'high_outside'
    df.loc[(df['plate_x'] >= 0) & (df['plate_z'] < 2.5), 'location_quadrant'] = 'low_inside'
    df.loc[(df['plate_x'] < 0) & (df['plate_z'] < 2.5), 'location_quadrant'] = 'low_outside'
    
    # Count advantage
    df['count_advantage'] = 'neutral'
    df.loc[df['balls'] > df['strikes'], 'count_advantage'] = 'hitter'
    df.loc[df['strikes'] > df['balls'], 'count_advantage'] = 'pitcher'
    
    # Additional features
    df['velocity_surprise'] = np.abs(df['release_speed'] - df['release_speed'].mean())
    df['velocity_consistency'] = 1.0 / (1.0 + df['velocity_surprise'])
    df['breaking_ball_velocity'] = df['is_breaking_ball'] * df['release_speed']
    
    # Movement deception
    df['expected_movement'] = df['release_speed'] * 0.1  # Simple expectation
    df['movement_deception'] = df['movement_magnitude'] - df['expected_movement']
    
    # Pitch type specific features
    df['fastball_high'] = df['is_fastball'] * df['high_pitch']
    df['breaking_ball_low'] = df['is_breaking_ball'] * df['low_pitch']
    df['offspeed_middle'] = df['is_offspeed'] * (1 - df['high_pitch'] - df['low_pitch'])
    
    # Location extreme
    df['location_extreme'] = ((df['plate_x'].abs() > 1.0) | (df['plate_z'] > 4.0) | (df['plate_z'] < 1.0)).astype(int)
    df['high_leverage'] = df['full_count'].astype(int)
    
    # Count ratios
    df['count_ratio'] = df['balls'] / (df['strikes'] + 0.1)
    df['behind_by_two'] = (df['strikes'] - df['balls'] >= 2).astype(int)
    df['ahead_by_two'] = (df['balls'] - df['strikes'] >= 2).astype(int)
    df['full_count_pressure'] = df['full_count'] * df['count_pressure']
    
    # Zone-specific features
    df['in_zone_two_strikes'] = df['in_strike_zone'] * df['two_strikes']
    df['out_zone_ahead'] = df['far_from_zone'] * df['ahead_in_count']
    df['edge_zone_decision'] = ((df['zone_distance'] > 0.5) & (df['zone_distance'] < 1.0)).astype(int)
    
    # Velocity and movement interactions
    df['velocity_x_location'] = df['release_speed'] * df['zone_distance']
    df['pitch_type_x_location'] = df['is_fastball'] * df['zone_distance']
    df['count_x_zone'] = df['count_pressure'] * df['zone_distance']
    
    # Situational decision features
    df['early_count_swing'] = ((df['count_total'] <= 2) & (df['in_strike_zone'] == 0)).astype(int)
    df['late_count_take'] = ((df['count_total'] >= 4) & (df['far_from_zone'] == 1)).astype(int)
    df['pressure_swing'] = df['two_strikes'] * df['in_strike_zone']
    df['opportunity_take'] = df['ahead_in_count'] * df['far_from_zone']
    
    # Zone and location encoding
    df['zone_quadrant_encoded'] = pd.Categorical(df['zone_quadrant']).codes
    df['location_quadrant_encoded'] = pd.Categorical(df['location_quadrant']).codes
    df['count_advantage_encoded'] = pd.Categorical(df['count_advantage']).codes
    
    # ADDING ALL MISSING FEATURES IDENTIFIED IN ANALYSIS
    print("Adding missing features to align with model expectations...")
    
    # 1. Breaking ball high
    df['breaking_ball_high'] = df['is_breaking_ball'] * df['high_pitch']
    
    # 2. Close game (simplified - would need game context)
    df['close_game'] = 0.0  # Default, would need game score data
    
    # 3. Corner pitch
    df['corner_pitch'] = ((df['plate_x'].abs() >= 0.7) | (df['plate_z'] >= 3.5) | (df['plate_z'] <= 1.5)).astype(int)
    
    # 4-7. Count-specific rates (will be calculated from career data)
    df['count_field_out_rate'] = 0.0
    df['count_hit_rate'] = 0.0
    df['count_swing_rate_adjustment'] = 0.0
    df['count_whiff_rate'] = 0.0
    
    # 8-16. Early count features
    df['early_count'] = (df['count_total'] <= 2).astype(int)
    df['early_count_breaking_penalty'] = df['early_count'] * df['is_breaking_ball'] * 0.1
    df['early_count_high_vel_penalty'] = df['early_count'] * df['high_velocity'] * 0.1
    df['early_count_location_penalty'] = df['early_count'] * df['far_from_zone'] * 0.1
    df['early_count_low_vel_penalty'] = df['early_count'] * df['low_velocity'] * 0.1
    df['early_count_offspeed_penalty'] = df['early_count'] * df['is_offspeed'] * 0.1
    df['early_count_outside_penalty'] = df['early_count'] * df['outside_pitch'] * 0.1
    df['early_count_penalty'] = df['early_count'] * 0.1
    df['early_count_zone_penalty'] = df['early_count'] * df['zone_distance'] * 0.1
    
    # 17. Extreme location
    df['extreme_location'] = ((df['plate_x'].abs() > 1.2) | (df['plate_z'] > 4.0) | (df['plate_z'] < 1.0)).astype(int)
    
    # 18-19. First/last pitch
    df['first_pitch'] = (df['count_total'] == 0).astype(int)
    df['last_pitch'] = (df['count_total'] >= 5).astype(int)
    
    # 20-22. Heart pitch and movement features
    df['heart_pitch'] = ((df['plate_x'].abs() <= 0.5) & (df['plate_z'] >= 2.0) & (df['plate_z'] <= 3.0)).astype(int)
    df['high_horizontal_movement'] = (df['movement_magnitude'] > 10).astype(int)
    df['high_movement_fastball'] = df['is_fastball'] * df['high_movement']
    df['high_vertical_movement'] = (df['movement_magnitude'] > 8).astype(int)
    
    # 23-24. Movement features
    df['horizontal_movement'] = df['movement_magnitude']  # Simplified
    df['vertical_movement'] = df['movement_magnitude']  # Simplified
    
    # 25. Inning late
    df['inning_late'] = 0.0  # Would need inning data
    
    # 26-27. Late count and low movement breaking
    df['late_count'] = (df['count_total'] >= 4).astype(int)
    df['low_movement_breaking'] = df['is_breaking_ball'] * (df['movement_magnitude'] < 4).astype(int)
    
    # 28-29. Middle count and movement ratio
    df['middle_count'] = ((df['count_total'] >= 2) & (df['count_total'] <= 3)).astype(int)
    df['movement_ratio'] = df['movement_magnitude'] / (df['release_speed'] + 0.1)
    
    # 30. Offspeed low
    df['offspeed_low'] = df['is_offspeed'] * df['low_pitch']
    
    # 31-32. Pitch in at bat and pitch type change
    df['pitch_in_at_bat'] = df['count_total']  # Simplified
    df['pitch_type_change'] = 0.0  # Would need sequence data
    
    # 33-35. Pitch type rates
    df['pitch_type_field_out_rate'] = 0.0
    df['pitch_type_hit_rate'] = 0.0
    df['pitch_type_whiff_rate'] = 0.0
    
    # 36-37. Plate location normalized movement
    df['plate_x_norm_x_movement'] = df['plate_x'] * df['movement_magnitude']
    df['plate_z_norm_x_movement'] = df['plate_z'] * df['movement_magnitude']
    
    # 38-41. Pressure features
    df['pressure_count'] = (df['strikes'] >= 2).astype(int)
    df['pressure_field_out_rate'] = 0.0
    df['pressure_hit_rate'] = 0.0
    df['pressure_whiff_rate'] = 0.0
    
    # 42-43. Shadow pitch and unexpected movement
    df['shadow_pitch'] = ((df['plate_x'].abs() <= 1.0) & (df['plate_z'] >= 1.2) & (df['plate_z'] <= 3.8) & 
                          ~((df['plate_x'].abs() <= 0.7) & (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5))).astype(int)
    df['unexpected_movement'] = np.abs(df['movement_magnitude'] - df['expected_movement'])
    
    # 44. Velocity drop
    df['velocity_drop'] = (df['release_speed'] < df['release_speed'].mean() - 5).astype(int)
    
    # 45-52. Zone-specific features
    df['zone_center_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
    df['zone_corner'] = ((df['plate_x'].abs() >= 0.7) | (df['plate_z'] >= 3.5) | (df['plate_z'] <= 1.5)).astype(int)
    df['zone_field_out_rate'] = 0.0
    df['zone_heart'] = ((df['plate_x'].abs() <= 0.5) & (df['plate_z'] >= 2.0) & (df['plate_z'] <= 3.0)).astype(int)
    df['zone_hit_rate'] = 0.0
    df['zone_shadow'] = df['shadow_pitch']
    df['zone_whiff_rate'] = 0.0
    
    # Hitter-specific features (using actual career data)
    if hitter_features:
        for feature_name, value in hitter_features.items():
            df[feature_name] = value
        
        # Add comprehensive count features dynamically for each pitch
        print("Adding comprehensive count features to each pitch...")
        
        # Extract count features from hitter_features (remove 'acuna_' prefix)
        count_features = {k.replace('acuna_', ''): v for k, v in hitter_features.items() 
                         if any(x in k for x in ['swing_rate', 'weighted_swing_rate'])}
        
        # Calculate current count features for each pitch
        current_count_features_list = []
        for idx, row in df.iterrows():
            current_count_features = get_count_features_for_pitch(
                row['balls'], row['strikes'], row['pitch_type'], count_features
            )
            current_count_features_list.append(current_count_features)
        
        # Add current count features to dataframe
        current_count_df = pd.DataFrame(current_count_features_list, index=df.index)
        df = pd.concat([df, current_count_df], axis=1)
        
        print(f"Added {len(current_count_features_list[0])} count-specific features per pitch")
        
        # NEW: Add proxy contact features that combine zone location with swing rates
        # These are especially useful for outcome prediction
        df['zone_heart_contact'] = df['zone_heart'] * df['acuna_zone_heart_swing_rate']
        df['zone_corner_contact'] = df['zone_corner'] * df['acuna_zone_corner_swing_rate']
        df['zone_shadow_contact'] = df['zone_shadow'] * df['acuna_zone_shadow_swing_rate']
        df['zone_overall_contact'] = df['in_strike_zone'] * df['acuna_zone_swing_rate']
        df['outside_zone_contact'] = df['far_from_zone'] * df['acuna_outside_swing_rate']
        
        # Additional contact features for different pitch types
        df['fastball_zone_contact'] = df['is_fastball'] * df['zone_overall_contact']
        df['breaking_zone_contact'] = df['is_breaking_ball'] * df['zone_overall_contact']
        df['offspeed_zone_contact'] = df['is_offspeed'] * df['zone_overall_contact']
        
        # Contact features for different count situations
        df['pressure_zone_contact'] = df['pressure_situation'] * df['zone_overall_contact']
        df['opportunity_zone_contact'] = df['ahead_in_count'] * df['zone_overall_contact']
        df['two_strikes_zone_contact'] = df['two_strikes'] * df['zone_overall_contact']
        
    else:
        # Fallback to zeros if career data not available
        df['acuna_fastball_swing_rate'] = 0.0
        df['acuna_breaking_swing_rate'] = 0.0
        df['acuna_offspeed_swing_rate'] = 0.0
        df['acuna_zone_swing_rate'] = 0.0
        df['acuna_outside_swing_rate'] = 0.0
        df['acuna_high_swing_rate'] = 0.0
        df['acuna_low_swing_rate'] = 0.0
        df['acuna_ahead_swing_rate'] = 0.0
        df['acuna_behind_swing_rate'] = 0.0
        df['acuna_two_strikes_swing_rate'] = 0.0
        df['acuna_full_count_swing_rate'] = 0.0
        df['acuna_high_vel_swing_rate'] = 0.0
        df['acuna_low_vel_swing_rate'] = 0.0
        df['acuna_high_movement_swing_rate'] = 0.0
        df['acuna_low_movement_swing_rate'] = 0.0
        df['acuna_late_inning_swing_rate'] = 0.0
        df['acuna_close_game_swing_rate'] = 0.0
        
        # Default count features
        df['current_count_overall_swing_rate'] = 0.0
        df['current_count_pitch_swing_rate'] = 0.0
        df['current_count_weighted_swing_rate'] = 0.0
        df['current_count_pitch_weighted_swing_rate'] = 0.0
        df['advantage_count_weight'] = 0.0
        df['acuna_first_pitch_swing_rate'] = 0.0
        df['acuna_last_pitch_swing_rate'] = 0.0
        df['acuna_pitch_type_change_swing_rate'] = 0.0
        df['acuna_velocity_drop_swing_rate'] = 0.0
        df['acuna_velocity_surge_swing_rate'] = 0.0
        df['acuna_location_extreme_swing_rate'] = 0.0
        df['acuna_location_heart_swing_rate'] = 0.0
        df['acuna_pressure_swing_rate'] = 0.0
        df['acuna_opportunity_swing_rate'] = 0.0
        df['acuna_zone_corner_swing_rate'] = 0.0
        df['acuna_zone_shadow_swing_rate'] = 0.0
        df['acuna_zone_heart_swing_rate'] = 0.0
        
        # NEW: Add proxy contact features with default values when no career data
        df['zone_heart_contact'] = df['zone_heart'] * 0.0  # Default to 0 when no career data
        df['zone_corner_contact'] = df['zone_corner'] * 0.0
        df['zone_shadow_contact'] = df['zone_shadow'] * 0.0
        df['zone_overall_contact'] = df['in_strike_zone'] * 0.0
        df['outside_zone_contact'] = df['far_from_zone'] * 0.0
        
        # Additional contact features for different pitch types
        df['fastball_zone_contact'] = df['is_fastball'] * df['zone_overall_contact']
        df['breaking_zone_contact'] = df['is_breaking_ball'] * df['zone_overall_contact']
        df['offspeed_zone_contact'] = df['is_offspeed'] * df['zone_overall_contact']
        
        # Contact features for different count situations
        df['pressure_zone_contact'] = df['pressure_situation'] * df['zone_overall_contact']
        df['opportunity_zone_contact'] = df['ahead_in_count'] * df['zone_overall_contact']
        df['two_strikes_zone_contact'] = df['two_strikes'] * df['zone_overall_contact']
    
    print(f"Engineered {len(df.columns)} features")
    print(f"Output dataset size: {len(df)} pitches")
    if len(df) != original_size:
        print(f"WARNING: Dataset size changed from {original_size} to {len(df)} pitches!")
    return df

def analyze_predictions(df, y_true, y_pred, probabilities):
    """Analyze prediction results in detail"""
    print("\n" + "="*60)
    print("DETAILED OUTCOME PREDICTION ANALYSIS")
    print("="*60)
    
    # Convert to numpy arrays for easier indexing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Overall accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class analysis
    outcomes = ['whiff', 'hit_safely', 'field_out']
    
    print(f"\nPER-CLASS ANALYSIS:")
    print("-" * 40)
    
    for outcome in outcomes:
        mask = y_true == outcome
        if mask.sum() > 0:
            class_accuracy = (y_true[mask] == y_pred[mask]).mean()
            class_count = mask.sum()
            print(f"\n{outcome.upper()} (n={class_count}):")
            print(f"  Accuracy: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
            
            # Find most common misclassifications
            misclassified = df[mask & (y_true != y_pred)]
            if len(misclassified) > 0:
                print(f"  Misclassified as:")
                for pred_outcome in outcomes:
                    if pred_outcome != outcome:
                        # Use boolean indexing correctly
                        pred_mask = y_pred[mask] == pred_outcome
                        count = pred_mask.sum()
                        if count > 0:
                            print(f"    {pred_outcome}: {count} ({count/len(misclassified)*100:.1f}%)")
    
    # Feature analysis for misclassifications
    print(f"\nMISCLASSIFICATION ANALYSIS:")
    print("-" * 40)
    
    misclassified_mask = y_true != y_pred
    if misclassified_mask.sum() > 0:
        misclassified_df = df[misclassified_mask].copy()
        misclassified_df['true_outcome'] = y_true[misclassified_mask]
        misclassified_df['predicted_outcome'] = y_pred[misclassified_mask]
        
        print(f"\nTop misclassification patterns:")
        
        # Analyze by pitch type
        pitch_type_errors = misclassified_df.groupby(['pitch_type', 'true_outcome', 'predicted_outcome']).size().sort_values(ascending=False)
        print(f"\nPitch Type Errors (top 10):")
        for (pitch_type, true_out, pred_out), count in pitch_type_errors.head(10).items():
            print(f"  {pitch_type}: {true_out} → {pred_out} ({count} times)")
        
        # Analyze by zone
        print(f"\nZone data quality in misclassifications:")
        print(f"  Total misclassified pitches: {len(misclassified_df)}")
        print(f"  Unique zone values: {misclassified_df['zone'].nunique()}")
        print(f"  Zone value range: {misclassified_df['zone'].min()} to {misclassified_df['zone'].max()}")
        print(f"  NaN zones: {misclassified_df['zone'].isna().sum()}")
        print(f"  Zero zones: {(misclassified_df['zone'] == 0).sum()}")
        
        zone_errors = misclassified_df.groupby(['zone', 'true_outcome', 'predicted_outcome']).size().sort_values(ascending=False)
        print(f"\nZone Errors (top 10):")
        for (zone, true_out, pred_out), count in zone_errors.head(10).items():
            print(f"  Zone {zone}: {true_out} → {pred_out} ({count} times)")
        
        # Analyze by velocity
        print(f"\nVelocity Analysis for Misclassifications:")
        for outcome in outcomes:
            outcome_errors = misclassified_df[misclassified_df['true_outcome'] == outcome]
            if len(outcome_errors) > 0:
                avg_vel = outcome_errors['release_speed'].mean()
                print(f"  {outcome} misclassifications - Avg velocity: {avg_vel:.1f} mph")
        
        # Analyze by movement
        print(f"\nMovement Analysis for Misclassifications:")
        for outcome in outcomes:
            outcome_errors = misclassified_df[misclassified_df['true_outcome'] == outcome]
            if len(outcome_errors) > 0:
                avg_movement = outcome_errors['movement_magnitude'].mean()
                print(f"  {outcome} misclassifications - Avg movement: {avg_movement:.2f}")
        
        # Analyze by count
        print(f"\nCount Analysis for Misclassifications:")
        for outcome in outcomes:
            outcome_errors = misclassified_df[misclassified_df['true_outcome'] == outcome]
            if len(outcome_errors) > 0:
                avg_balls = outcome_errors['balls'].mean()
                avg_strikes = outcome_errors['strikes'].mean()
                print(f"  {outcome} misclassifications - Avg count: {avg_balls:.1f}-{avg_strikes:.1f}")
    
    # Confidence analysis
    print(f"\nCONFIDENCE ANALYSIS:")
    print("-" * 40)
    
    # Get max probability for each prediction
    max_probs = np.max(probabilities, axis=1)
    
    # Analyze confidence by accuracy
    high_conf_mask = max_probs >= 0.8
    med_conf_mask = (max_probs >= 0.6) & (max_probs < 0.8)
    low_conf_mask = max_probs < 0.6
    
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (y_true[high_conf_mask] == y_pred[high_conf_mask]).mean()
        print(f"High confidence (≥80%): {high_conf_accuracy:.3f} accuracy on {high_conf_mask.sum()} predictions")
    
    if med_conf_mask.sum() > 0:
        med_conf_accuracy = (y_true[med_conf_mask] == y_pred[med_conf_mask]).mean()
        print(f"Medium confidence (60-80%): {med_conf_accuracy:.3f} accuracy on {med_conf_mask.sum()} predictions")
    
    if low_conf_mask.sum() > 0:
        low_conf_accuracy = (y_true[low_conf_mask] == y_pred[low_conf_mask]).mean()
        print(f"Low confidence (<60%): {low_conf_accuracy:.3f} accuracy on {low_conf_mask.sum()} predictions")
    
    # Show some example misclassifications
    print(f"\nEXAMPLE MISCLASSIFICATIONS:")
    print("-" * 40)
    
    for i in range(min(5, len(misclassified_df))):
        row = misclassified_df.iloc[i]
        max_prob = max_probs[misclassified_mask][i]
        print(f"\nExample {i+1}:")
        print(f"  {row['pitch_type']} pitch at ({row['plate_x']:.2f}, {row['plate_z']:.2f}) - Zone {row['zone']}")
        print(f"  Count: {row['balls']:.0f}-{row['strikes']:.0f} | Velocity: {row['release_speed']:.0f} mph | Movement: {row['movement_magnitude']:.1f}")
        print(f"  True: {row['true_outcome']} | Predicted: {row['predicted_outcome']} | Confidence: {max_prob:.1%}")

def main():
    """Main function to test outcome prediction model on holdout data"""
    print("Testing Outcome Prediction Model on Holdout Data")
    print("=" * 60)
    
    # Load holdout data
    df = load_holdout_data()
    if df is None:
        return
    
    # Load outcome model
    outcome_model, outcome_preprocessor, outcome_threshold = load_outcome_model()
    if outcome_model is None:
        return
    
    # Load career data to calculate actual features
    hitter_df, pitcher_df = load_career_data()
    if hitter_df is None or pitcher_df is None:
        return

    # Calculate actual hitter features
    hitter_features = calculate_hitter_features(hitter_df)
    # Calculate actual pitcher features
    pitcher_features = calculate_pitcher_features(pitcher_df)

    # Prepare features for outcome prediction
    swing_df, outcome_counts = prepare_outcome_features(df, hitter_features, pitcher_features)
    if swing_df is None:
        return
    
    # Make predictions
    try:
        # Filter features to only include what the preprocessor expects
        available_feats = [f for f in outcome_preprocessor.feature_names_in_ if f in swing_df.columns]
        missing_feats = [f for f in outcome_preprocessor.feature_names_in_ if f not in swing_df.columns]
        
        print(f"Available features: {len(available_feats)}")
        print(f"Missing features: {len(missing_feats)}")
        
        if missing_feats:
            # Add default values for missing features
            for feat in missing_feats:
                if feat in ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team', 'zone_quadrant', 'location_quadrant', 'count_advantage']:
                    swing_df[feat] = 'unknown'
                else:
                    swing_df[feat] = 0.0
        
        # Ensure categorical features are strings
        categorical_features = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team', 'zone_quadrant', 'location_quadrant', 'count_advantage']
        for feat in categorical_features:
            if feat in swing_df.columns:
                swing_df[feat] = swing_df[feat].astype(str)
        
        # Ensure numeric features are numeric
        numeric_features = [f for f in outcome_preprocessor.feature_names_in_ if f not in categorical_features]
        for feat in numeric_features:
            if feat in swing_df.columns:
                swing_df[feat] = pd.to_numeric(swing_df[feat], errors='coerce').fillna(0.0)
        
        # Transform the features using only what the preprocessor expects
        X = outcome_preprocessor.transform(swing_df[outcome_preprocessor.feature_names_in_])
        
        # Clean NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Get prediction probabilities
        probabilities = outcome_model.predict_proba(X)
        
        # Get predictions
        predictions = outcome_model.predict(X)
        
        # Get true outcomes
        y_true = swing_df['outcome'].values
        
        # Create outcome mapping for predictions
        outcome_classes = ['whiff', 'hit_safely', 'field_out']
        y_pred = [outcome_classes[pred] for pred in predictions]
        
        # Clean target variables - ensure all are strings and handle NaN values
        y_true = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_true]
        y_pred = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_pred]
        
        print(f"\nPREDICTION RESULTS:")
        print("-" * 40)
        print(f"Total swing pitches: {len(swing_df)}")
        print(f"Predictions made: {len(y_pred)}")
        
        # Print classification report
        print(f"\nCLASSIFICATION REPORT:")
        print("-" * 40)
        print(classification_report(y_true, y_pred, target_names=outcome_classes))
        
        # Print confusion matrix
        print(f"\nCONFUSION MATRIX:")
        print("-" * 40)
        cm = confusion_matrix(y_true, y_pred, labels=outcome_classes)
        print("True\Pred\t" + "\t".join(outcome_classes))
        for i, true_outcome in enumerate(outcome_classes):
            print(f"{true_outcome}\t" + "\t".join([str(cm[i][j]) for j in range(len(outcome_classes))]))
        
        # Clean zones in the final dataset before analysis
        print(f"\nFinal zone validation before analysis:")
        print(f"Before final cleaning: {len(swing_df)} pitches")
        
        # Check for invalid zones in final dataset
        invalid_zones_final = swing_df[(swing_df['zone'] <= 0) | (swing_df['zone'].isna())].copy()
        print(f"Pitches with invalid zones (≤0 or NaN): {len(invalid_zones_final)}")
        
        if len(invalid_zones_final) > 0:
            print("Sample invalid zone pitches in final dataset:")
            print(invalid_zones_final[['plate_x', 'plate_z', 'zone', 'events']].head())
            print(f"Zone value distribution in invalid zones:")
            print(invalid_zones_final['zone'].value_counts().head())
        
        # Apply zone fixing to final dataset
        invalid_zone_mask_final = (swing_df['zone'] <= 0) | (swing_df['zone'].isna())
        if invalid_zone_mask_final.sum() > 0:
            print(f"Fixing {invalid_zone_mask_final.sum()} invalid zones in final dataset...")
            
            def fix_zone_final(row):
                if pd.isna(row['zone']) or row['zone'] <= 0:
                    plate_x, plate_z = row['plate_x'], row['plate_z']
                    if pd.isna(plate_x) or pd.isna(plate_z):
                        return 1
                    elif abs(plate_x) <= 0.7 and 1.5 <= plate_z <= 3.5:
                        if plate_x >= 0:
                            return 1 if plate_z >= 2.5 else 3
                        else:
                            return 2 if plate_z >= 2.5 else 4
                    else:
                        if plate_x >= 0:
                            return 5 if plate_z >= 2.5 else 7
                        else:
                            return 6 if plate_z >= 2.5 else 8
                else:
                    return row['zone']
            
            swing_df.loc[invalid_zone_mask_final, 'zone'] = swing_df[invalid_zone_mask_final].apply(fix_zone_final, axis=1)
        else:
            print("No invalid zones found in final dataset")
        
        # Remove any remaining invalid zones
        swing_df = swing_df[swing_df['zone'] > 0]
        print(f"After final cleaning: {len(swing_df)} pitches")
        
        # Update arrays to match cleaned dataset
        if len(swing_df) != len(y_true):
            print(f"Warning: Dataset size changed from {len(y_true)} to {len(swing_df)} after zone cleaning")
            # Re-run predictions on cleaned dataset
            X = outcome_preprocessor.transform(swing_df[outcome_preprocessor.feature_names_in_])
            X = np.nan_to_num(X, nan=0.0)
            probabilities = outcome_model.predict_proba(X)
            predictions = outcome_model.predict(X)
            y_true = swing_df['outcome'].values
            y_pred = [outcome_classes[pred] for pred in predictions]
            y_true = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_true]
            y_pred = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_pred]
        
        # Detailed analysis
        analyze_predictions(swing_df, y_true, y_pred, probabilities)
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 