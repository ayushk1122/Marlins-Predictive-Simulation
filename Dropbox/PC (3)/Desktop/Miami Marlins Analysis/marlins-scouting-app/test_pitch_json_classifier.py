import json
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')
from comprehensive_count_features import calculate_comprehensive_count_features, get_count_features_for_pitch, analyze_count_patterns

def load_career_data():
    """Load career data for both hitter and pitcher to calculate actual features"""
    try:
        # Load hitter career data
        hitter_df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        
        # Load pitcher career data  
        pitcher_df = pd.read_csv('sandy_alcantara_complete_career_statcast.csv')
        
        return hitter_df, pitcher_df
    except Exception as e:
        print(f"✗ Error loading career data: {e}")
        return None, None

def calculate_hitter_features(hitter_df):
    """Calculate actual hitter-specific features from career data"""
    
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
    
    # Calculate comprehensive count features dynamically from career data
    count_features, count_results = calculate_comprehensive_count_features(hitter_df)
    
    # Add count features to hitter features
    for feature_name, feature_value in count_features.items():
        hitter_features[f'acuna_{feature_name}'] = feature_value
    
    # Calculate zone-specific swing rates from actual data
    zone_heart_swings = swing_df[((swing_df['plate_x'].abs() <= 0.5) & (swing_df['plate_z'] >= 2.0) & (swing_df['plate_z'] <= 3.0))]
    zone_corner_swings = swing_df[((swing_df['plate_x'].abs() >= 0.7) | (swing_df['plate_z'] >= 3.5) | (swing_df['plate_z'] <= 1.5))]
    zone_shadow_swings = swing_df[((swing_df['plate_x'].abs() <= 1.0) & (swing_df['plate_z'] >= 1.2) & (swing_df['plate_z'] <= 3.8)) & ~((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5))]
    
    hitter_features['acuna_zone_heart_swing_rate'] = len(zone_heart_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_zone_corner_swing_rate'] = len(zone_corner_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    hitter_features['acuna_zone_shadow_swing_rate'] = len(zone_shadow_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
    
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
    
    print(f"Calculated {len(hitter_features)} hitter features")
    return hitter_features

def load_average_metrics():
    """
    Load average metrics for different pitch types from the training data.
    This will be used to fill in missing features in the JSON pitch data.
    """
    # Load the training data to calculate averages
    try:
        df = pd.read_csv('ronald_acuna_jr_complete_career_statcast.csv')
        print(f"Loaded training data with {len(df)} pitches")
        
        # Calculate averages by pitch type
        averages = {}
        for pitch_type in df['pitch_type'].unique():
            if pd.notna(pitch_type):
                pitch_data = df[df['pitch_type'] == pitch_type]
                averages[pitch_type] = {
                    'release_speed': pitch_data['release_speed'].mean(),
                    'release_spin_rate': pitch_data['release_spin_rate'].mean(),
                    'pfx_x': pitch_data['pfx_x'].mean(),
                    'pfx_z': pitch_data['pfx_z'].mean(),
                    'sz_top': pitch_data['sz_top'].mean(),
                    'sz_bot': pitch_data['sz_bot'].mean(),
                    'balls': pitch_data['balls'].mean(),
                    'strikes': pitch_data['strikes'].mean(),
                    'inning': pitch_data['inning'].mean() if 'inning' in pitch_data.columns else 5,
                    'home_score': pitch_data['home_score'].mean() if 'home_score' in pitch_data.columns else 3,
                    'away_score': pitch_data['away_score'].mean() if 'away_score' in pitch_data.columns else 3,
                    'release_extension': pitch_data['release_extension'].mean() if 'release_extension' in pitch_data.columns else 6.0,
                    'release_pos_x': pitch_data['release_pos_x'].mean() if 'release_pos_x' in pitch_data.columns else -1.0,
                    'release_pos_y': pitch_data['release_pos_y'].mean() if 'release_pos_y' in pitch_data.columns else 50.0,
                    'release_pos_z': pitch_data['release_pos_z'].mean() if 'release_pos_z' in pitch_data.columns else 6.0,
                    'vx0': pitch_data['vx0'].mean() if 'vx0' in pitch_data.columns else 0.0,
                    'vy0': pitch_data['vy0'].mean() if 'vy0' in pitch_data.columns else -130.0,
                    'vz0': pitch_data['vz0'].mean() if 'vz0' in pitch_data.columns else -5.0,
                    'api_break_z_with_gravity': pitch_data['api_break_z_with_gravity'].mean() if 'api_break_z_with_gravity' in pitch_data.columns else 0.0,
                    'api_break_x_batter_in': pitch_data['api_break_x_batter_in'].mean() if 'api_break_x_batter_in' in pitch_data.columns else 0.0,
                    'api_break_x_arm': pitch_data['api_break_x_arm'].mean() if 'api_break_x_arm' in pitch_data.columns else 0.0,
                    'arm_angle': pitch_data['arm_angle'].mean() if 'arm_angle' in pitch_data.columns else 90.0,
                    'spin_dir': pitch_data['spin_dir'].mean() if 'spin_dir' in pitch_data.columns else 0.0,
                    'spin_rate_deprecated': pitch_data['spin_rate_deprecated'].mean() if 'spin_rate_deprecated' in pitch_data.columns else 0.0,
                    'break_angle_deprecated': pitch_data['break_angle_deprecated'].mean() if 'break_angle_deprecated' in pitch_data.columns else 0.0,
                    'break_length_deprecated': pitch_data['break_length_deprecated'].mean() if 'break_length_deprecated' in pitch_data.columns else 0.0,
                    'effective_speed': pitch_data['effective_speed'].mean() if 'effective_speed' in pitch_data.columns else pitch_data['release_speed'].mean(),
                    'age_pit': pitch_data['age_pit'].mean() if 'age_pit' in pitch_data.columns else 28.0,
                    'spin_axis': pitch_data['spin_axis'].mean() if 'spin_axis' in pitch_data.columns else 0.0,
                }
        
        # Also calculate overall averages for missing pitch types
        overall_avg = {
            'release_speed': df['release_speed'].mean(),
            'release_spin_rate': df['release_spin_rate'].mean(),
            'pfx_x': df['pfx_x'].mean(),
            'pfx_z': df['pfx_z'].mean(),
            'sz_top': df['sz_top'].mean(),
            'sz_bot': df['sz_bot'].mean(),
            'balls': df['balls'].mean(),
            'strikes': df['strikes'].mean(),
            'inning': df['inning'].mean() if 'inning' in df.columns else 5,
            'home_score': df['home_score'].mean() if 'home_score' in df.columns else 3,
            'away_score': df['away_score'].mean() if 'away_score' in df.columns else 3,
            'release_extension': df['release_extension'].mean() if 'release_extension' in df.columns else 6.0,
            'release_pos_x': df['release_pos_x'].mean() if 'release_pos_x' in df.columns else -1.0,
            'release_pos_y': df['release_pos_y'].mean() if 'release_pos_y' in df.columns else 50.0,
            'release_pos_z': df['release_pos_z'].mean() if 'release_pos_z' in df.columns else 6.0,
            'vx0': df['vx0'].mean() if 'vx0' in df.columns else 0.0,
            'vy0': df['vy0'].mean() if 'vy0' in df.columns else -130.0,
            'vz0': df['vz0'].mean() if 'vz0' in df.columns else -5.0,
            'api_break_z_with_gravity': df['api_break_z_with_gravity'].mean() if 'api_break_z_with_gravity' in df.columns else 0.0,
            'api_break_x_batter_in': df['api_break_x_batter_in'].mean() if 'api_break_x_batter_in' in df.columns else 0.0,
            'api_break_x_arm': df['api_break_x_arm'].mean() if 'api_break_x_arm' in df.columns else 0.0,
            'arm_angle': df['arm_angle'].mean() if 'arm_angle' in df.columns else 90.0,
            'spin_dir': df['spin_dir'].mean() if 'spin_dir' in df.columns else 0.0,
            'spin_rate_deprecated': df['spin_rate_deprecated'].mean() if 'spin_rate_deprecated' in df.columns else 0.0,
            'break_angle_deprecated': df['break_angle_deprecated'].mean() if 'break_angle_deprecated' in df.columns else 0.0,
            'break_length_deprecated': df['break_length_deprecated'].mean() if 'break_length_deprecated' in df.columns else 0.0,
            'effective_speed': df['effective_speed'].mean() if 'effective_speed' in df.columns else df['release_speed'].mean(),
            'age_pit': df['age_pit'].mean() if 'age_pit' in df.columns else 28.0,
            'spin_axis': df['spin_axis'].mean() if 'spin_axis' in df.columns else 0.0,
        }
        
        averages['overall'] = overall_avg
        
        print(f"Calculated averages for {len(averages)} pitch types")
        return averages
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        # Return default averages if training data not available
        return {
            'FF': {
                'release_speed': 95.0, 'release_spin_rate': 2200.0, 'pfx_x': 0.5, 'pfx_z': -5.0,
                'sz_top': 3.5, 'sz_bot': 1.5, 'balls': 1.5, 'strikes': 1.5, 'inning': 5,
                'home_score': 3, 'away_score': 3, 'release_extension': 6.0, 'release_pos_x': -1.0,
                'release_pos_y': 50.0, 'release_pos_z': 6.0, 'vx0': 0.0, 'vy0': -130.0, 'vz0': -5.0,
                'api_break_z_with_gravity': 0.0, 'api_break_x_batter_in': 0.0, 'api_break_x_arm': 0.0,
                'arm_angle': 90.0, 'spin_dir': 0.0, 'spin_rate_deprecated': 0.0,
                'break_angle_deprecated': 0.0, 'break_length_deprecated': 0.0, 'effective_speed': 95.0, 'age_pit': 28.0,
                'spin_axis': 0.0
            },
            'overall': {
                'release_speed': 92.0, 'release_spin_rate': 2100.0, 'pfx_x': 0.3, 'pfx_z': -4.0,
                'sz_top': 3.5, 'sz_bot': 1.5, 'balls': 1.5, 'strikes': 1.5, 'inning': 5,
                'home_score': 3, 'away_score': 3, 'release_extension': 6.0, 'release_pos_x': -1.0,
                'release_pos_y': 50.0, 'release_pos_z': 6.0, 'vx0': 0.0, 'vy0': -130.0, 'vz0': -5.0,
                'api_break_z_with_gravity': 0.0, 'api_break_x_batter_in': 0.0, 'api_break_x_arm': 0.0,
                'arm_angle': 90.0, 'spin_dir': 0.0, 'spin_rate_deprecated': 0.0,
                'break_angle_deprecated': 0.0, 'break_length_deprecated': 0.0, 'effective_speed': 92.0, 'age_pit': 28.0,
                'spin_axis': 0.0
            }
        }

def prepare_pitch_features(pitch_data, averages):
    """
    Prepare features for a single pitch from JSON data.
    Fill in missing features using averages or calculated values.
    """
    # Start with the basic pitch data
    features = {}
    
    # Get pitch type and corresponding averages
    pitch_type = pitch_data.get('pitch_type', 'FF')
    pitch_avg = averages.get(pitch_type, averages.get('overall', averages['FF']))
    
    # Basic features from JSON
    features['pitch_type'] = pitch_type
    features['plate_x'] = pitch_data.get('plate_x', 0.0)
    features['plate_z'] = pitch_data.get('plate_z', 2.5)
    features['zone'] = pitch_data.get('zone', 5)
    features['handedness'] = pitch_data.get('handedness', 'R')
    
    # Fill in missing features using averages
    features['release_speed'] = pitch_data.get('release_speed', pitch_avg['release_speed'])
    features['release_spin_rate'] = pitch_data.get('release_spin_rate', pitch_avg['release_spin_rate'])
    features['pfx_x'] = pitch_data.get('pfx_x', pitch_avg['pfx_x'])
    features['pfx_z'] = pitch_data.get('pfx_z', pitch_avg['pfx_z'])
    features['sz_top'] = pitch_data.get('sz_top', pitch_avg['sz_top'])
    features['sz_bot'] = pitch_data.get('sz_bot', pitch_avg['sz_bot'])
    features['balls'] = pitch_data.get('balls', pitch_avg['balls'])
    features['strikes'] = pitch_data.get('strikes', pitch_avg['strikes'])
    features['inning'] = pitch_data.get('inning', pitch_avg['inning'])
    features['home_score'] = pitch_data.get('home_score', pitch_avg['home_score'])
    features['away_score'] = pitch_data.get('away_score', pitch_avg['away_score'])
    
    # Additional features from averages
    features['release_extension'] = pitch_avg['release_extension']
    features['release_pos_x'] = pitch_avg['release_pos_x']
    features['release_pos_y'] = pitch_avg['release_pos_y']
    features['release_pos_z'] = pitch_avg['release_pos_z']
    features['vx0'] = pitch_avg['vx0']
    features['vy0'] = pitch_avg['vy0']
    features['vz0'] = pitch_avg['vz0']
    features['api_break_z_with_gravity'] = pitch_avg['api_break_z_with_gravity']
    features['api_break_x_batter_in'] = pitch_avg['api_break_x_batter_in']
    features['api_break_x_arm'] = pitch_avg['api_break_x_arm']
    features['arm_angle'] = pitch_avg['arm_angle']
    features['spin_dir'] = pitch_avg['spin_dir']
    features['spin_rate_deprecated'] = pitch_avg['spin_rate_deprecated']
    features['break_angle_deprecated'] = pitch_avg['break_angle_deprecated']
    features['break_length_deprecated'] = pitch_avg['break_length_deprecated']
    features['effective_speed'] = pitch_avg['effective_speed']
    features['age_pit'] = pitch_avg['age_pit']
    features['spin_axis'] = pitch_avg.get('spin_axis', 0.0)  # Add missing spin_axis feature
    
    # Categorical features
    features['p_throws'] = 'R'  # Default to right-handed pitcher
    features['if_fielding_alignment'] = 'Standard'
    features['of_fielding_alignment'] = 'Standard'
    features['stand'] = 'R'  # Default to right-handed batter
    features['home_team'] = 'MIA'
    
    # Calculate derived features
    features['zone_center_x'] = 0
    features['zone_center_z'] = (features['sz_top'] + features['sz_bot']) / 2
    features['zone_distance'] = np.sqrt(
        (features['plate_x'] - features['zone_center_x'])**2 + 
        (features['plate_z'] - features['zone_center_z'])**2
    )
    
    # IMPROVED Movement Quantification using Statcast break values
    features['horizontal_break'] = features.get('api_break_x_batter_in', 0) or 0
    features['vertical_break'] = features.get('api_break_z_with_gravity', 0) or 0
    features['arm_side_break'] = features.get('api_break_x_arm', 0) or 0
    
    # Calculate total movement magnitude using the more accurate break values
    features['movement_magnitude'] = np.sqrt(features['horizontal_break']**2 + features['vertical_break']**2)
    
    # Normalize location
    features['plate_x_norm'] = features['plate_x'] / 1.417
    features['plate_z_norm'] = (features['plate_z'] - features['sz_bot']) / (features['sz_top'] - features['sz_bot'])
    
    # Interaction features
    features['plate_x_norm_x_movement'] = features['plate_x_norm'] * features['movement_magnitude']
    features['plate_z_norm_x_movement'] = features['plate_z_norm'] * features['movement_magnitude']
    
    # Additional movement features using improved break values
    features['movement_direction'] = np.arctan2(features['vertical_break'], features['horizontal_break']) * 180 / np.pi
    features['movement_ratio'] = np.abs(features['horizontal_break']) / (np.abs(features['vertical_break']) + 0.1)
    
    # Count-based features
    features['count_pressure'] = features['balls'] - features['strikes']
    features['count_total'] = features['balls'] + features['strikes']
    features['behind_in_count'] = int(features['strikes'] > features['balls'])
    features['ahead_in_count'] = int(features['balls'] > features['strikes'])
    features['two_strikes'] = int(features['strikes'] >= 2)
    features['three_balls'] = int(features['balls'] >= 3)
    

    
    # Zone-specific features
    features['in_strike_zone'] = int(
        (features['plate_x'] >= -0.85) and (features['plate_x'] <= 0.85) and
        (features['plate_z'] >= features['sz_bot']) and (features['plate_z'] <= features['sz_top'])
    )
    features['far_from_zone'] = int(features['zone_distance'] > 1.0)
    features['high_pitch'] = int(features['plate_z'] > features['sz_top'])
    features['low_pitch'] = int(features['plate_z'] < features['sz_bot'])
    features['inside_pitch'] = int(features['plate_x'] < -0.85)
    features['outside_pitch'] = int(features['plate_x'] > 0.85)
    
    # Pitch type features
    features['is_fastball'] = int(pitch_type in ['FF', 'SI', 'FC'])
    features['is_breaking_ball'] = int(pitch_type in ['SL', 'CU', 'KC'])
    features['is_offspeed'] = int(pitch_type in ['CH', 'FS'])
    

    

    
    # Interaction features
    features['zone_distance_x_count_pressure'] = features['zone_distance'] * features['count_pressure']
    features['movement_x_count_pressure'] = features['movement_magnitude'] * features['count_pressure']
    features['in_zone_x_two_strikes'] = features['in_strike_zone'] * features['two_strikes']
    features['far_from_zone_x_ahead'] = features['far_from_zone'] * features['ahead_in_count']
    
    # Velocity and movement features
    features['velocity_movement_ratio'] = features['release_speed'] / (features['movement_magnitude'] + 0.1)
    features['high_velocity'] = int(features['release_speed'] > 95)
    features['low_velocity'] = int(features['release_speed'] < 85)
    

    
    # Improved high movement detection based on pitch type
    if features['pitch_type'] in ['SL', 'CU', 'KC']:  # Breaking balls
        features['high_movement'] = int(features['movement_magnitude'] > 8)
    elif features['pitch_type'] in ['CH', 'FS']:  # Offspeed
        features['high_movement'] = int(features['movement_magnitude'] > 6)
    else:  # Fastballs
        features['high_movement'] = int(features['movement_magnitude'] > 4)
    
    # Zone edge features
    features['zone_edge_distance'] = min(
        abs(features['plate_x'] - (-0.85)),
        abs(features['plate_x'] - 0.85)
    )
    features['zone_top_distance'] = abs(features['plate_z'] - features['sz_top'])
    features['zone_bottom_distance'] = abs(features['plate_z'] - features['sz_bot'])
    features['closest_zone_edge'] = min(
        min(features['zone_edge_distance'], features['zone_top_distance']),
        features['zone_bottom_distance']
    )
    
    # Count-specific features
    features['full_count'] = int((features['balls'] == 3) and (features['strikes'] == 2))
    features['hitters_count'] = int((features['balls'] >= 2) and (features['strikes'] <= 1))
    features['pitchers_count'] = int((features['balls'] <= 1) and (features['strikes'] >= 2))
    features['neutral_count'] = int((features['balls'] == 1) and (features['strikes'] == 1))
    
    # Zone quadrant
    if features['plate_x'] >= 0:
        if features['plate_z'] >= (features['sz_top'] + features['sz_bot']) / 2:
            features['zone_quadrant'] = 'up_out'
        else:
            features['zone_quadrant'] = 'down_out'
    else:
        if features['plate_z'] >= (features['sz_top'] + features['sz_bot']) / 2:
            features['zone_quadrant'] = 'up_in'
        else:
            features['zone_quadrant'] = 'down_in'
    
    # Pitch deception features
    features['velocity_drop'] = int((features['release_speed'] < 90) and features['is_fastball'])
    features['breaking_ball_high'] = int(features['is_breaking_ball'] and (features['plate_z'] > features['sz_top']))
    features['offspeed_low'] = int(features['is_offspeed'] and (features['plate_z'] < features['sz_bot']))
    
    # Game situation features
    features['inning_late'] = int(features['inning'] >= 7)
    features['close_game'] = int(abs(features['home_score'] - features['away_score']) <= 2)
    
    # At-bat features (default values)
    features['pitch_in_at_bat'] = 1
    features['first_pitch'] = 1
    features['last_pitch'] = 1
    
    # Velocity features
    features['velocity_diff_from_avg'] = 0  # Default since we don't have pitcher history
    
    # Movement features (using improved break values)
    features['horizontal_movement'] = features['horizontal_break']  # Use the better break values
    features['vertical_movement'] = features['vertical_break']  # Use the better break values
    # movement_ratio already calculated above with better values
    features['high_horizontal_movement'] = int(abs(features['horizontal_break']) > 5)
    features['high_vertical_movement'] = int(abs(features['vertical_break']) > 5)
    
    # Zone precision features
    features['zone_center_distance'] = np.sqrt(
        features['plate_x']**2 + (features['plate_z'] - (features['sz_top'] + features['sz_bot'])/2)**2
    )
    features['zone_corner'] = int(
        ((features['plate_x'] >= 0.7) or (features['plate_x'] <= -0.7)) and
        ((features['plate_z'] >= features['sz_top'] - 0.2) or (features['plate_z'] <= features['sz_bot'] + 0.2))
    )
    features['zone_heart'] = int(features['zone_center_distance'] < 0.5)
    features['zone_shadow'] = int(
        ((features['plate_x'] >= -1.0) and (features['plate_x'] <= 1.0) and
         (features['plate_z'] >= features['sz_bot'] - 0.2) and (features['plate_z'] <= features['sz_top'] + 0.2)) and
        not features['in_strike_zone']
    )
    
    # Count psychology features
    if features['count_pressure'] > 0:
        features['count_advantage'] = 'hitter_ahead'
    elif features['count_pressure'] < 0:
        features['count_advantage'] = 'pitcher_ahead'
    else:
        features['count_advantage'] = 'neutral'
    
    features['pressure_situation'] = int(features['two_strikes'] or features['three_balls'])
    features['must_swing'] = features['two_strikes']
    features['can_take'] = int(features['ahead_in_count'] and not features['in_strike_zone'])
    
    # Pitch type deception features
    features['fastball_high'] = int(features['is_fastball'] and (features['plate_z'] > features['sz_top']))
    features['breaking_ball_low'] = int(features['is_breaking_ball'] and (features['plate_z'] < features['sz_bot']))
    features['offspeed_middle'] = int(features['is_offspeed'] and features['in_strike_zone'])
    features['pitch_type_change'] = 0  # Default since we don't have sequence data
    
    # Location features
    if features['plate_x'] >= 0:
        if features['plate_z'] >= (features['sz_top'] + features['sz_bot']) / 2:
            features['location_quadrant'] = 'up_right'
        else:
            features['location_quadrant'] = 'down_right'
    else:
        if features['plate_z'] >= (features['sz_top'] + features['sz_bot']) / 2:
            features['location_quadrant'] = 'up_left'
        else:
            features['location_quadrant'] = 'down_left'
    
    features['location_extreme'] = int(
        (abs(features['plate_x']) > 1.0) or
        (features['plate_z'] > features['sz_top'] + 0.5) or
        (features['plate_z'] < features['sz_bot'] - 0.5)
    )
    
    # Spin features (use averages)
    features['spin_movement_correlation'] = features['release_spin_rate'] * features['movement_magnitude']
    features['high_spin'] = int(features['release_spin_rate'] > 2500)
    features['low_spin'] = int(features['release_spin_rate'] < 2000)
    
    # Game situation features
    features['late_inning'] = int(features['inning'] >= 8)
    features['close_score'] = int(abs(features['home_score'] - features['away_score']) <= 1)
    features['high_leverage'] = int(features['late_inning'] and features['close_score'])
    
    # Batter and pitcher swing rates (default values)
    features['batter_swing_rate'] = 0.5
    features['batter_zone_swing_rate'] = 0.5
    features['pitcher_swing_rate'] = 0.5
    features['pitcher_zone_swing_rate'] = 0.5
    
    # Advanced count-based features
    features['count_ratio'] = features['strikes'] / (features['balls'] + features['strikes'] + 0.1)
    features['behind_by_two'] = int(features['strikes'] - features['balls'] >= 2)
    features['ahead_by_two'] = int(features['balls'] - features['strikes'] >= 2)
    features['full_count_pressure'] = int((features['balls'] == 3) and (features['strikes'] == 2))
    
    # Zone-specific count features
    features['in_zone_two_strikes'] = int(features['in_strike_zone'] and features['two_strikes'])
    features['out_zone_ahead'] = int(not features['in_strike_zone'] and features['ahead_in_count'])
    features['edge_zone_decision'] = int((features['zone_distance'] > 0.5) and (features['zone_distance'] < 1.0))
    
    # Velocity deception features
    features['velocity_surprise'] = int((features['release_speed'] < 90) and features['is_fastball'])
    features['velocity_consistency'] = int((features['release_speed'] > 95) and features['is_fastball'])
    features['breaking_ball_velocity'] = int(features['is_breaking_ball'] and (features['release_speed'] > 85))
    
    # Movement deception features
    features['high_movement_fastball'] = int(features['is_fastball'] and (features['movement_magnitude'] > 8))
    features['low_movement_breaking'] = int(features['is_breaking_ball'] and (features['movement_magnitude'] < 5))
    features['unexpected_movement'] = int((features['movement_magnitude'] > 12) or (features['movement_magnitude'] < 3))
    
    # Location deception features
    features['corner_pitch'] = features['zone_corner']
    features['heart_pitch'] = features['zone_heart']
    features['shadow_pitch'] = features['zone_shadow']
    features['extreme_location'] = int(
        (abs(features['plate_x']) > 1.2) or
        (features['plate_z'] > features['sz_top'] + 0.8) or
        (features['plate_z'] < features['sz_bot'] - 0.8)
    )
    
    # Advanced interaction features
    features['velocity_x_location'] = features['release_speed'] * features['zone_distance']
    features['pitch_type_x_location'] = features['is_fastball'] * features['in_strike_zone']
    features['count_x_zone'] = features['count_pressure'] * features['in_strike_zone']
    
    # Context-based features
    features['early_count_swing'] = int((features['count_total'] <= 2) and features['in_strike_zone'])
    features['late_count_take'] = int((features['count_total'] >= 4) and not features['in_strike_zone'])
    features['pressure_swing'] = int(features['two_strikes'] and features['in_strike_zone'])
    features['opportunity_take'] = int(features['ahead_in_count'] and not features['in_strike_zone'])
    
    # Encode categorical features
    features['zone_quadrant_encoded'] = ['up_in', 'up_out', 'down_in', 'down_out'].index(features['zone_quadrant'])
    features['location_quadrant_encoded'] = ['up_left', 'up_right', 'down_left', 'down_right'].index(features['location_quadrant'])
    features['count_advantage_encoded'] = ['hitter_ahead', 'neutral', 'pitcher_ahead'].index(features['count_advantage'])
    
    # Acuna-specific swing tendency features (will be calculated from career data)
    # These will be filled in by the calling function
    acuna_features = {}
    
    # Dynamic advantage count features based on current count and pitch type
    # These will be calculated from the career data and passed in as hitter_features
    current_count = f"{features['balls']}-{features['strikes']}"
    current_pitch_type = features['pitch_type']
    
    # Get comprehensive count features from career data (passed in as hitter_features)
    if 'hitter_features' in locals():
        # Extract count features from hitter_features (remove 'acuna_' prefix)
        count_features = {k.replace('acuna_', ''): v for k, v in hitter_features.items() 
                         if any(x in k for x in ['swing_rate', 'weighted_swing_rate'])}
        
        # Calculate current count features using the comprehensive function
        current_count_features = get_count_features_for_pitch(
            features['balls'], features['strikes'], current_pitch_type, count_features
        )
        
        # Add current count features to the pitch features
        for feature_name, feature_value in current_count_features.items():
            features[feature_name] = feature_value
    else:
        # Default values if no career data available
        features['current_count_overall_swing_rate'] = 0.5
        features['current_count_pitch_swing_rate'] = 0.5
        features['current_count_weighted_swing_rate'] = 0.25
        features['current_count_pitch_weighted_swing_rate'] = 0.25
        features['advantage_count_weight'] = 0.5
    
    # Add Acuna features (will be calculated from career data)
    for feature_name in ['acuna_fastball_swing_rate', 'acuna_breaking_swing_rate', 'acuna_offspeed_swing_rate',
                        'acuna_zone_swing_rate', 'acuna_outside_swing_rate', 'acuna_high_swing_rate', 'acuna_low_swing_rate',
                        'acuna_ahead_swing_rate', 'acuna_behind_swing_rate', 'acuna_two_strikes_swing_rate', 'acuna_full_count_swing_rate',
                        'acuna_high_vel_swing_rate', 'acuna_low_vel_swing_rate',
                        'acuna_high_movement_swing_rate', 'acuna_low_movement_swing_rate',
                        'acuna_late_inning_swing_rate', 'acuna_close_game_swing_rate',
                        'acuna_first_pitch_swing_rate', 'acuna_last_pitch_swing_rate',
                        'acuna_pitch_type_change_swing_rate', 'acuna_velocity_drop_swing_rate', 'acuna_velocity_surge_swing_rate',
                        'acuna_location_extreme_swing_rate', 'acuna_location_heart_swing_rate',
                        'acuna_pressure_swing_rate', 'acuna_opportunity_swing_rate',
                        'acuna_zone_corner_swing_rate', 'acuna_zone_shadow_swing_rate', 'acuna_zone_heart_swing_rate']:
        features[feature_name] = 0.0  # Placeholder, will be filled by career data
    
    # ADDING ALL MISSING FEATURES IDENTIFIED IN ANALYSIS
    print("Adding missing features to align with model expectations...")
    
    # 1. Breaking ball high (already calculated above)
    # 2. Close game (already calculated above)
    # 3. Corner pitch (already calculated above)
    
    # 4-7. Count-specific rates (will be calculated from career data)
    features['count_field_out_rate'] = 0.0
    features['count_hit_rate'] = 0.0
    features['count_swing_rate_adjustment'] = 0.0
    features['count_whiff_rate'] = 0.0
    
    # 8-16. Early count features
    features['early_count'] = int(features['count_total'] <= 2)
    features['early_count_breaking_penalty'] = features['early_count'] * features['is_breaking_ball'] * 0.1
    features['early_count_high_vel_penalty'] = features['early_count'] * features['high_velocity'] * 0.1
    features['early_count_location_penalty'] = features['early_count'] * features['far_from_zone'] * 0.1
    features['early_count_low_vel_penalty'] = features['early_count'] * features['low_velocity'] * 0.1
    features['early_count_offspeed_penalty'] = features['early_count'] * features['is_offspeed'] * 0.1
    features['early_count_outside_penalty'] = features['early_count'] * features['outside_pitch'] * 0.1
    features['early_count_penalty'] = features['early_count'] * 0.1
    features['early_count_zone_penalty'] = features['early_count'] * features['zone_distance'] * 0.1
    
    # 17. Extreme location (already calculated above)
    # 18-19. First/last pitch (already calculated above)
    # 20-22. Heart pitch and movement features (already calculated above)
    # 23-24. Movement features (already calculated above)
    # 25. Inning late (already calculated above)
    # 26-27. Late count and low movement breaking (already calculated above)
    # 28-29. Middle count and movement ratio (already calculated above)
    # 30. Offspeed low (already calculated above)
    # 31-32. Pitch in at bat and pitch type change (already calculated above)
    
    # 33-35. Pitch type rates
    features['pitch_type_field_out_rate'] = 0.0
    features['pitch_type_hit_rate'] = 0.0
    features['pitch_type_whiff_rate'] = 0.0
    
    # 36-37. Plate location normalized movement (already calculated above)
    # 38-41. Pressure features (already calculated above)
    # 42-43. Shadow pitch and unexpected movement (already calculated above)
    # 44. Velocity drop (already calculated above)
    # 45-52. Zone-specific features (already calculated above)
    
    return features

def test_pitch_classifier(json_file_path):
    """
    Test the swing classifier on a pitch JSON file.
    """
    print(f"Testing swing classifier on: {json_file_path}")
    
    # Load the pitch data
    try:
        with open(json_file_path, 'r') as f:
            pitch_data = json.load(f)
    except Exception as e:
        print(f"✗ Error loading JSON file: {e}")
        return
    
    # FIX ZONE 0.0 ISSUE - Validate zone data
    if 'zone' in pitch_data and pitch_data['zone'] <= 0:
        print(f"⚠️  Invalid zone detected: {pitch_data['zone']}")
        
        # Recalculate zone from plate coordinates
        plate_x = pitch_data.get('plate_x', 0)
        plate_z = pitch_data.get('plate_z', 2.5)
        
        if pd.isna(plate_x) or pd.isna(plate_z):
            pitch_data['zone'] = 1  # Default to zone 1
        elif abs(plate_x) <= 0.7 and 1.5 <= plate_z <= 3.5:
            # In strike zone
            if plate_x >= 0:
                pitch_data['zone'] = 1 if plate_z >= 2.5 else 3
            else:
                pitch_data['zone'] = 2 if plate_z >= 2.5 else 4
        else:
            # Outside strike zone
            if plate_x >= 0:
                pitch_data['zone'] = 5 if plate_z >= 2.5 else 7
            else:
                pitch_data['zone'] = 6 if plate_z >= 2.5 else 8
        
        print(f"✅ Fixed zone to: {pitch_data['zone']}")
    
    # Load career data to calculate actual features
    hitter_df, pitcher_df = load_career_data()
    if hitter_df is None or pitcher_df is None:
        print("✗ Could not load career data, using defaults")
        hitter_features = {}
    else:
        # Calculate actual hitter features
        hitter_features = calculate_hitter_features(hitter_df)
    
    # Load average metrics
    averages = load_average_metrics()
    
    # Prepare features for the pitch
    features = prepare_pitch_features(pitch_data, averages)
    
    # Update features with actual career data
    if hitter_features:
        for feature_name, value in hitter_features.items():
            if feature_name in features:
                features[feature_name] = value
        
        # Add comprehensive count features dynamically
        current_count = f"{features['balls']}-{features['strikes']}"
        current_pitch_type = features['pitch_type']
        
        # Extract count features from hitter_features (remove 'acuna_' prefix)
        count_features = {k.replace('acuna_', ''): v for k, v in hitter_features.items() 
                         if any(x in k for x in ['swing_rate', 'weighted_swing_rate'])}
        
        # Calculate current count features using the comprehensive function
        current_count_features = get_count_features_for_pitch(
            features['balls'], features['strikes'], current_pitch_type, count_features
        )
        
        for feature_name, feature_value in current_count_features.items():
            features[feature_name] = feature_value
    
    # Load the trained model
    try:
        with open('sequential_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        # Get the swing classifier
        swing_classifier = models.get('swing_calibrated_model')
        if swing_classifier is None:
            print("✗ Swing classifier not found in model file")
            return
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Create a DataFrame with the features
    df = pd.DataFrame([features])
    
    # Get the feature names that the model expects
    # We'll use the same feature list as in training
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'age_pit',
        'zone_distance', 'movement_magnitude', 'plate_x_norm_x_movement', 'plate_z_norm_x_movement',
        'count_pressure', 'count_total', 'behind_in_count', 'ahead_in_count', 'two_strikes', 'three_balls',
        'in_strike_zone', 'far_from_zone', 'high_pitch', 'low_pitch', 'inside_pitch', 'outside_pitch',
        'is_fastball', 'is_breaking_ball', 'is_offspeed',
        'zone_distance_x_count_pressure', 'movement_x_count_pressure', 'in_zone_x_two_strikes', 'far_from_zone_x_ahead',
        'velocity_movement_ratio', 'high_velocity', 'low_velocity', 'high_movement',
        'zone_edge_distance', 'zone_top_distance', 'zone_bottom_distance', 'closest_zone_edge',
        'full_count', 'hitters_count', 'pitchers_count', 'neutral_count',
        'velocity_drop', 'breaking_ball_high', 'offspeed_low',
        'inning_late', 'close_game',
        'pitch_in_at_bat', 'first_pitch', 'last_pitch',
        'velocity_diff_from_avg', 'horizontal_movement', 'vertical_movement', 'movement_ratio',
        'high_horizontal_movement', 'high_vertical_movement', 'zone_center_distance',
        'zone_corner', 'zone_heart', 'zone_shadow', 'pressure_situation', 'must_swing', 'can_take',
        'fastball_high', 'breaking_ball_low', 'offspeed_middle', 'pitch_type_change',
        'location_extreme', 'high_leverage',
        'count_ratio', 'behind_by_two', 'ahead_by_two', 'full_count_pressure',
        'in_zone_two_strikes', 'out_zone_ahead', 'edge_zone_decision',
        'velocity_surprise', 'velocity_consistency', 'breaking_ball_velocity',
        'high_movement_fastball', 'low_movement_breaking', 'unexpected_movement',
        'corner_pitch', 'heart_pitch', 'shadow_pitch', 'extreme_location',
        'velocity_x_location', 'pitch_type_x_location', 'count_x_zone',
        'early_count_swing', 'late_count_take', 'pressure_swing', 'opportunity_take',
        'zone_quadrant_encoded', 'location_quadrant_encoded', 'count_advantage_encoded',
        'acuna_fastball_swing_rate', 'acuna_breaking_swing_rate', 'acuna_offspeed_swing_rate',
        'acuna_zone_swing_rate', 'acuna_outside_swing_rate', 'acuna_high_swing_rate', 'acuna_low_swing_rate',
        'acuna_ahead_swing_rate', 'acuna_behind_swing_rate', 'acuna_two_strikes_swing_rate', 'acuna_full_count_swing_rate',
        'acuna_high_vel_swing_rate', 'acuna_low_vel_swing_rate',
        'acuna_high_movement_swing_rate', 'acuna_low_movement_swing_rate',
        'acuna_late_inning_swing_rate', 'acuna_close_game_swing_rate',
        'acuna_first_pitch_swing_rate', 'acuna_last_pitch_swing_rate',
        'acuna_pitch_type_change_swing_rate', 'acuna_velocity_drop_swing_rate', 'acuna_velocity_surge_swing_rate',
        'acuna_location_extreme_swing_rate', 'acuna_location_heart_swing_rate',
        'acuna_pressure_swing_rate', 'acuna_opportunity_swing_rate',
        'acuna_zone_corner_swing_rate', 'acuna_zone_shadow_swing_rate', 'acuna_zone_heart_swing_rate'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team', 'zone_quadrant', 'location_quadrant', 'count_advantage']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    all_feats = num_feats + cat_feats
    
    # Make prediction
    try:
        # Get the preprocessor from the model
        preprocessor = models['swing_preprocessor']
        
        # Filter features to only include what the preprocessor expects
        available_feats = [f for f in preprocessor.feature_names_in_ if f in df.columns]
        missing_feats = [f for f in preprocessor.feature_names_in_ if f not in df.columns]
        
        if missing_feats:
            # Add default values for missing features
            for feat in missing_feats:
                df[feat] = 0.0
        
        # Transform the features using only what the preprocessor expects
        X = preprocessor.transform(df[preprocessor.feature_names_in_])
        
        # Clean NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Get the calibrated model and threshold
        calibrated_model = swing_classifier
        # Use the threshold from the trained model (now set to 90% confidence)
        threshold = models.get('swing_threshold', 0.9)
        
        # Get prediction probability
        swing_probability = calibrated_model.predict_proba(X)[0][1]
        
        # Apply count-specific thresholds to reduce early count false positives
        count_thresholds = {
            'early_count': 0.95,    # Very high threshold for early counts (≤1 ball, ≤1 strike)
            'middle_count': 0.85,   # High threshold for middle counts (1-1)
            'pressure_count': 0.75,  # Lower threshold for pressure situations (≥2 strikes or ≥3 balls)
            'default': 0.9          # Default threshold for other situations
        }
        
        # Determine count situation
        balls = features.get('balls', 0)
        strikes = features.get('strikes', 0)
        
        if balls <= 1 and strikes <= 1:
            threshold = count_thresholds['early_count']
            count_situation = "early_count"
        elif balls == 1 and strikes == 1:
            threshold = count_thresholds['middle_count']
            count_situation = "middle_count"
        elif strikes >= 2 or balls >= 3:
            threshold = count_thresholds['pressure_count']
            count_situation = "pressure_count"
        else:
            threshold = count_thresholds['default']
            count_situation = "default"
        
        # Make prediction with count-specific threshold
        swing_prediction = 1 if swing_probability >= threshold else 0
        
        # Print results
        print(f"\n{pitch_data.get('pitcher', 'Unknown')} vs {pitch_data.get('hitter', 'Unknown')}")
        print(f"{pitch_data.get('pitch_type', 'Unknown')} pitch at ({pitch_data.get('plate_x', 0):.2f}, {pitch_data.get('plate_z', 0):.2f}) - Zone {pitch_data.get('zone', 'Unknown')}")
        print(f"Count: {features['balls']:.0f}-{features['strikes']:.0f} | Velocity: {features['release_speed']:.0f} mph | Movement: {features['movement_magnitude']:.1f}")
        print(f"PREDICTION: {'SWING' if swing_prediction else 'NO SWING'} | Confidence: {swing_probability:.1%} | Threshold: {threshold:.1%} ({count_situation})")
        
        return swing_prediction, swing_probability
        
    except Exception as e:
        print(f"✗ Error making prediction: {e}")
        return None, None

def main():
    """
    Main function to test the classifier on pitch JSON files.
    """
    # Find all JSON files in the Pitches folder
    pitches_dir = "Pitches"
    
    if not os.path.exists(pitches_dir):
        print(f"Pitches directory not found: {pitches_dir}")
        print("Please ensure the Pitches folder exists.")
        return
    
    # Get all JSON files in the Pitches folder
    json_files = [f for f in os.listdir(pitches_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {pitches_dir} folder")
        print("Please add pitch JSON files to the Pitches folder.")
        return
    
    # Test each JSON file
    for json_file in json_files:
        json_path = os.path.join(pitches_dir, json_file)
        print(f"\n{'='*60}")
        print(f"Testing file: {json_file}")
        print(f"{'='*60}")
        test_pitch_classifier(json_path)

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 