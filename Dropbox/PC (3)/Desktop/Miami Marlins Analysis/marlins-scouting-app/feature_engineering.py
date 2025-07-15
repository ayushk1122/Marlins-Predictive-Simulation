import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

def calculate_basic_features(df, model_type='whiff_vs_contact'):
    """Calculate basic engineered features - ALL 191 features for swing vs no swing model"""
    df = df.copy()
    
    # 1. Handle missing Statcast columns with fallbacks
    missing_columns = [
        'spin_axis', 'release_extension', 'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'arm_angle', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated', 'effective_speed', 'age_pit'
    ]
    
    for col in missing_columns:
        if col not in df.columns:
            if col in ['spin_axis', 'arm_angle', 'spin_dir']:
                df[col] = 0.0  # Angular measurements
            elif col in ['release_extension', 'release_pos_x', 'release_pos_y', 'release_pos_z']:
                df[col] = 0.0  # Position measurements
            elif col in ['vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z']:
                df[col] = 0.0  # Velocity components
            elif col in ['spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated']:
                df[col] = 0.0  # Deprecated fields
            elif col == 'effective_speed':
                df[col] = df['release_speed'] if 'release_speed' in df.columns else 90.0
            elif col == 'age_pit':
                df[col] = 25.0  # Default pitcher age
    
    # 2. Basic Statcast features (1-29)
    # These should already be in the DataFrame from Statcast data
    # release_speed, release_spin_rate, spin_axis, release_extension, release_pos_x, release_pos_y, release_pos_z
    # vx0, vy0, vz0, pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot, zone
    # api_break_z_with_gravity, api_break_x_batter_in, api_break_x_arm, arm_angle
    # balls, strikes, spin_dir, spin_rate_deprecated, break_angle_deprecated, break_length_deprecated
    # effective_speed, age_pit
    
    # 3. Basic engineered features (30-56)
    # Zone distance
    df['zone_distance'] = np.sqrt(
        (df['plate_x'] - 0)**2 + 
        (df['plate_z'] - (df['sz_top'] + df['sz_bot'])/2)**2
    )
    
    # Movement magnitude
    df['movement_magnitude'] = np.sqrt(df['api_break_x_batter_in']**2 + df['api_break_z_with_gravity']**2)
    
    # Add missing features for whiff vs contact model
    df['horizontal_break'] = df['api_break_x_batter_in'].fillna(0)
    df['vertical_break'] = df['api_break_z_with_gravity'].fillna(0)
    df['low_movement'] = (df['movement_magnitude'] < 2).astype(int)
    df['movement_diff_from_avg'] = df['movement_magnitude'] - df['movement_magnitude'].mean()
    
    # Normalized location features
    df['plate_x_norm'] = df['plate_x'] / 1.417
    df['plate_z_norm'] = (df['plate_z'] - df['sz_bot']) / (df['sz_top'] - df['sz_bot'])
    
    # Location x movement interactions
    df['plate_x_norm_x_movement'] = df['plate_x_norm'] * df['movement_magnitude']
    df['plate_z_norm_x_movement'] = df['plate_z_norm'] * df['movement_magnitude']
    
    # Count features
    df['count_pressure'] = df['balls'] - df['strikes']
    df['count_total'] = df['balls'] + df['strikes']
    df['behind_in_count'] = (df['strikes'] > df['balls']).astype(int)
    df['ahead_in_count'] = (df['balls'] > df['strikes']).astype(int)
    df['two_strikes'] = (df['strikes'] >= 2).astype(int)
    df['three_balls'] = (df['balls'] >= 3).astype(int)
    
    # Zone features
    df['in_strike_zone'] = ((df['plate_x'] >= -0.85) & (df['plate_x'] <= 0.85) & 
                           (df['plate_z'] >= df['sz_bot']) & (df['plate_z'] <= df['sz_top'])).astype(int)
    df['far_from_zone'] = (df['zone_distance'] > 1.0).astype(int)
    df['high_pitch'] = (df['plate_z'] > df['sz_top']).astype(int)
    df['low_pitch'] = (df['plate_z'] < df['sz_bot']).astype(int)
    df['inside_pitch'] = (df['plate_x'] < -0.85).astype(int)
    df['outside_pitch'] = (df['plate_x'] > 0.85).astype(int)
    
    # Pitch type features
    df['is_fastball'] = df['pitch_type'].isin(['FF', 'SI', 'FC']).astype(int)
    df['is_breaking_ball'] = df['pitch_type'].isin(['SL', 'CU', 'KC']).astype(int)
    df['is_offspeed'] = df['pitch_type'].isin(['CH', 'FS']).astype(int)
    
    # Interaction features
    df['zone_distance_x_count_pressure'] = df['zone_distance'] * df['count_pressure']
    df['movement_x_count_pressure'] = df['movement_magnitude'] * df['count_pressure']
    df['in_zone_x_two_strikes'] = df['in_strike_zone'] * df['two_strikes']
    df['far_from_zone_x_ahead'] = df['far_from_zone'] * df['ahead_in_count']
    
    # Velocity features
    df['velocity_movement_ratio'] = df['release_speed'] / (df['movement_magnitude'] + 0.1)
    df['high_velocity'] = (df['release_speed'] > 95).astype(int)
    df['low_velocity'] = (df['release_speed'] < 85).astype(int)
    df['high_movement'] = (df['movement_magnitude'] > 6).astype(int)
    
    # Zone edge features
    df['zone_edge_distance'] = np.minimum(
        np.abs(df['plate_x'] - (-0.85)),  # Distance from left edge
        np.abs(df['plate_x'] - 0.85)      # Distance from right edge
    )
    df['zone_top_distance'] = np.abs(df['plate_z'] - df['sz_top'])
    df['zone_bottom_distance'] = np.abs(df['plate_z'] - df['sz_bot'])
    df['closest_zone_edge'] = np.minimum(
        np.minimum(df['zone_edge_distance'], df['zone_top_distance']),
        df['zone_bottom_distance']
    )
    
    # 4. Count-based features (57-64)
    df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
    df['hitters_count'] = (df['balls'] >= 2) & (df['strikes'] <= 1)
    df['pitchers_count'] = (df['balls'] <= 1) & (df['strikes'] >= 2)
    df['neutral_count'] = (df['balls'] == 1) & (df['strikes'] == 1)
    
    # Pitch deception features
    df['velocity_drop'] = (df['release_speed'] < 90) & df['is_fastball']
    df['breaking_ball_high'] = df['is_breaking_ball'] & (df['plate_z'] > df['sz_top'])
    df['offspeed_low'] = df['is_offspeed'] & (df['plate_z'] < df['sz_bot'])
    
    # Game situation features
    df['inning_late'] = (df['inning'] >= 7).astype(int) if 'inning' in df.columns else 0
    df['close_game'] = (np.abs(df['home_score'] - df['away_score']) <= 2).astype(int) if 'home_score' in df.columns and 'away_score' in df.columns else 0
    
    # 5. Pitch sequence features (65-72)
    df['pitch_in_at_bat'] = df.groupby('at_bat_number').cumcount() + 1 if 'at_bat_number' in df.columns else 1
    df['first_pitch'] = (df['pitch_in_at_bat'] == 1).astype(int)
    df['last_pitch'] = df.groupby('at_bat_number')['pitch_in_at_bat'].transform('max') == df['pitch_in_at_bat'] if 'at_bat_number' in df.columns else 1
    
    # Velocity features
    df['velocity_diff_from_avg'] = df['release_speed'] - df['release_speed'].mean()
    
    # Movement features
    df['horizontal_movement'] = df['api_break_x_batter_in']
    df['vertical_movement'] = df['api_break_z_with_gravity']
    df['movement_ratio'] = np.abs(df['horizontal_movement']) / (np.abs(df['vertical_movement']) + 0.1)
    df['high_horizontal_movement'] = (np.abs(df['horizontal_movement']) > 5).astype(int)
    df['high_vertical_movement'] = (np.abs(df['vertical_movement']) > 5).astype(int)
    
    # 6. Zone features (73-82)
    df['zone_center_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - (df['sz_top'] + df['sz_bot'])/2)**2)
    df['zone_corner'] = ((df['zone'] == 1) | (df['zone'] == 3) | 
                         (df['zone'] == 7) | (df['zone'] == 9)).astype(int)
    df['zone_heart'] = ((df['zone'] == 2) | (df['zone'] == 5) | 
                        (df['zone'] == 8)).astype(int)
    df['zone_shadow'] = ((df['zone'] == 4) | (df['zone'] == 6)).astype(int)
    
    # Pressure features
    df['pressure_situation'] = (df['two_strikes'] | df['three_balls']).astype(int)
    df['must_swing'] = df['two_strikes'].astype(int)
    df['can_take'] = (df['ahead_in_count'] & ~df['in_strike_zone']).astype(int)
    
    # 7. Pitch type deception features (83-90)
    df['fastball_high'] = df['is_fastball'] & (df['plate_z'] > df['sz_top'])
    df['breaking_ball_low'] = df['is_breaking_ball'] & (df['plate_z'] < df['sz_bot'])
    df['offspeed_middle'] = df['is_offspeed'] & df['in_strike_zone']
    
    # Pitch type change
    df['pitch_type_change'] = df.groupby('at_bat_number')['pitch_type'].shift(1) != df['pitch_type'] if 'at_bat_number' in df.columns else 0
    
    # Location extreme
    df['location_extreme'] = (np.abs(df['plate_x']) > 1.0) | (df['plate_z'] > df['sz_top'] + 0.5) | (df['plate_z'] < df['sz_bot'] - 0.5)
    
    # High leverage
    df['high_leverage'] = (df['inning_late'] & df['close_game']).astype(int)
    
    # 8. Advanced count features (91-98)
    df['count_ratio'] = df['strikes'] / (df['balls'] + df['strikes'] + 0.1)
    df['behind_by_two'] = (df['strikes'] - df['balls'] >= 2).astype(int)
    df['ahead_by_two'] = (df['balls'] - df['strikes'] >= 2).astype(int)
    df['full_count_pressure'] = df['full_count'] * df['count_pressure']
    df['in_zone_two_strikes'] = df['in_strike_zone'] & df['two_strikes']
    df['out_zone_ahead'] = ~df['in_strike_zone'] & df['ahead_in_count']
    df['edge_zone_decision'] = ((df['zone_distance'] > 0.5) & (df['zone_distance'] < 1.0)).astype(int)
    
    # 9. Advanced velocity features (99-101)
    df['velocity_surprise'] = (df['release_speed'] < 90) & df['is_fastball']
    df['velocity_consistency'] = (df['release_speed'] > 95) & df['is_fastball']
    df['breaking_ball_velocity'] = df['is_breaking_ball'] & (df['release_speed'] > 85)
    
    # 10. Advanced movement features (102-104)
    df['high_movement_fastball'] = df['is_fastball'] & (df['movement_magnitude'] > 8)
    df['low_movement_breaking'] = df['is_breaking_ball'] & (df['movement_magnitude'] < 5)
    df['unexpected_movement'] = ((df['movement_magnitude'] > 12) | (df['movement_magnitude'] < 3)).astype(int)
    
    # 11. Advanced location features (105-108)
    df['corner_pitch'] = df['zone_corner'].astype(int)
    df['heart_pitch'] = df['zone_heart'].astype(int)
    df['shadow_pitch'] = df['zone_shadow'].astype(int)
    df['extreme_location'] = (np.abs(df['plate_x']) > 1.2) | (df['plate_z'] > df['sz_top'] + 0.8) | (df['plate_z'] < df['sz_bot'] - 0.8)
    
    # 12. Advanced interaction features (109-111)
    df['velocity_x_location'] = df['release_speed'] * df['zone_distance']
    df['pitch_type_x_location'] = df['is_fastball'] * df['in_strike_zone']
    df['count_x_zone'] = df['count_pressure'] * df['in_strike_zone']
    
    # 13. Advanced context features (112-115)
    df['early_count_swing'] = (df['count_total'] <= 2) & df['in_strike_zone']
    df['late_count_take'] = (df['count_total'] >= 4) & ~df['in_strike_zone']
    df['pressure_swing'] = df['two_strikes'] & df['in_strike_zone']
    df['opportunity_take'] = df['ahead_in_count'] & ~df['in_strike_zone']
    
    # 14. Encoded features (116-118) - Create as numeric directly
    # Zone quadrant
    df['zone_quadrant'] = np.where(
        df['plate_x'] >= 0,
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 0, 1),  # 0=up_out, 1=down_out
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 2, 3)   # 2=up_in, 3=down_in
    )
    df['zone_quadrant_encoded'] = df['zone_quadrant']  # Already numeric
    
    # Location quadrant
    df['location_quadrant'] = np.where(
        df['plate_x'] >= 0,
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 0, 1),  # 0=up_right, 1=down_right
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 2, 3)   # 2=up_left, 3=down_left
    )
    df['location_quadrant_encoded'] = df['location_quadrant']  # Already numeric
    
    # Count advantage
    df['count_advantage'] = np.where(df['count_pressure'] > 0, 0, 
                                    np.where(df['count_pressure'] < 0, 1, 2))  # 0=hitter_ahead, 1=pitcher_ahead, 2=neutral
    df['count_advantage_encoded'] = df['count_advantage']  # Already numeric
    
    # 15. Advanced count features (119-131)
    df['early_count'] = ((df['balls'] <= 1) & (df['strikes'] <= 1)).astype(int)
    df['middle_count'] = ((df['balls'] == 1) & (df['strikes'] == 1)).astype(int)
    df['late_count'] = ((df['balls'] >= 2) | (df['strikes'] >= 2)).astype(int)
    df['pressure_count'] = ((df['strikes'] >= 2) | (df['balls'] >= 3)).astype(int)
    
    # Count penalties
    df['early_count_penalty'] = df['early_count'] * 0.3
    df['early_count_zone_penalty'] = df['early_count'] * df['in_strike_zone'] * 0.2
    df['early_count_outside_penalty'] = df['early_count'] * (~df['in_strike_zone']).astype(int) * 0.5
    
    # Count swing rate adjustments
    df['count_swing_rate_adjustment'] = np.where(
        df['early_count'] == 1, -0.25,  # Reduce swing probability by 25% in early counts
        np.where(
            df['pressure_count'] == 1, 0.15,  # Increase swing probability by 15% in pressure situations
            0.0  # No adjustment for middle counts
        )
    )
    
    # Count-specific location penalties
    df['early_count_location_penalty'] = np.where(
        df['early_count'] == 1,
        np.where(
            df['zone_distance'] > 0.5, 0.4,  # 40% penalty for pitches >0.5 feet from zone in early counts
            np.where(
                df['zone_distance'] > 0.2, 0.2,  # 20% penalty for pitches >0.2 feet from zone in early counts
                0.0  # No penalty for very close pitches
            )
        ),
        0.0  # No penalty for non-early counts
    )
    
    # Count-specific pitch type penalties
    df['early_count_breaking_penalty'] = df['early_count'] * df['is_breaking_ball'] * 0.3
    df['early_count_offspeed_penalty'] = df['early_count'] * df['is_offspeed'] * 0.25
    df['early_count_low_vel_penalty'] = df['early_count'] * df['low_velocity'] * 0.35
    df['early_count_high_vel_penalty'] = df['early_count'] * df['high_velocity'] * 0.15
    
    # 16. Hitter-specific features (132-165) - These will be added by calculate_hitter_features
    # Placeholder values for now
    hitter_features = [
        'acuna_fastball_swing_rate', 'acuna_breaking_swing_rate', 'acuna_offspeed_swing_rate',
        'acuna_zone_swing_rate', 'acuna_outside_swing_rate', 'acuna_high_swing_rate', 'acuna_low_swing_rate',
        'acuna_ahead_swing_rate', 'acuna_behind_swing_rate', 'acuna_two_strikes_swing_rate',
        'acuna_full_count_swing_rate', 'acuna_high_vel_swing_rate', 'acuna_low_vel_swing_rate',
        'acuna_high_movement_swing_rate', 'acuna_low_movement_swing_rate', 'acuna_late_inning_swing_rate',
        'acuna_close_game_swing_rate', 'acuna_first_pitch_swing_rate', 'acuna_last_pitch_swing_rate',
        'acuna_pitch_type_change_swing_rate', 'acuna_velocity_drop_swing_rate', 'acuna_velocity_surge_swing_rate',
        'acuna_location_extreme_swing_rate', 'acuna_location_heart_swing_rate', 'acuna_pressure_swing_rate',
        'acuna_opportunity_swing_rate', 'acuna_zone_corner_swing_rate', 'acuna_zone_shadow_swing_rate',
        'acuna_zone_heart_swing_rate', 'acuna_zone_heart_hit_rate', 'acuna_zone_corner_hit_rate',
        'acuna_zone_shadow_hit_rate', 'acuna_zone_hit_rate', 'acuna_outside_hit_rate'
    ]
    
    for feature in hitter_features:
        df[feature] = 0.0  # Default values, will be calculated by calculate_hitter_features
    
    # 17. Contact rate features (166-176) - These will be added by calculate_contact_rate_features
    contact_features = [
        'zone_heart_contact', 'zone_corner_contact', 'zone_shadow_contact', 'zone_overall_contact',
        'outside_zone_contact', 'fastball_zone_contact', 'breaking_zone_contact', 'offspeed_zone_contact',
        'pressure_zone_contact', 'opportunity_zone_contact', 'two_strikes_zone_contact'
    ]
    
    for feature in contact_features:
        df[feature] = 0.0  # Default values, will be calculated by calculate_contact_rate_features
    
    # 18. BABIP features (177-182) - These will be added by calculate_babip_features
    # Don't add these here since they're already added by calculate_babip_features function
    # This prevents duplicate columns
    pass
    
    # 19. Categorical features (183-191) - These need to be encoded as numeric
    categorical_features = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    for feat in categorical_features:
        if feat in df.columns:
            # Convert categorical to numeric codes
            df[feat] = pd.Categorical(df[feat].astype(str)).codes
    
    return df

def calculate_babip_features(df):
    """Add BABIP features from CSV file"""
    df = df.copy()
    
    try:
        babip_df = pd.read_csv('pitch_type_zone_batting_averages.csv')
        babip_lookup = {}
        for _, row in babip_df.iterrows():
            key = (row['Pitch Type'], row['Zone'])
            babip_lookup[key] = {
                'batting_average_bip': row['Batting Average (BIP)'],
                'whiff_rate': row['Whiff Rate'],
                'field_out_rate_bip': row['Field Out Rate (BIP)'],
                'balls_in_play': row['Balls in Play'],
                'total_swings': row['Total Swings'],
                'total_whiffs': row['Total Whiffs']
            }
        
        # Add BABIP features for each pitch
        babip_features = []
        for idx, row in df.iterrows():
            pitch_type = row['pitch_type']
            zone = row['zone']
            key = (pitch_type, zone)
            if key in babip_lookup:
                babip_features.append(babip_lookup[key])
            else:
                babip_features.append({
                    'batting_average_bip': 0.25,
                    'whiff_rate': 0.35,
                    'field_out_rate_bip': 0.40,
                    'balls_in_play': 0,
                    'total_swings': 0,
                    'total_whiffs': 0
                })
        
        babip_df_features = pd.DataFrame(babip_features, index=df.index)
        df = pd.concat([df, babip_df_features], axis=1)
        print("✓ Added BABIP features")
    except:
        # Add default BABIP features
        df['batting_average_bip'] = 0.25
        df['whiff_rate'] = 0.35
        df['field_out_rate_bip'] = 0.40
        df['balls_in_play'] = 0
        df['total_swings'] = 0
        df['total_whiffs'] = 0
        print("Added default BABIP features")
    
    return df

def calculate_hitter_features(df, hitter_name='acuna'):
    """Calculate hitter-specific features from career data"""
    df = df.copy()
    
    try:
        # Load career data for the hitter
        career_file = f'ronald_acuna_jr_complete_career_statcast.csv'
        career_df = pd.read_csv(career_file)
        
        # Calculate hitter averages
        hitter_swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
        career_df['is_swing'] = career_df['description'].isin(hitter_swing_events).astype(int)
        
        # Add pitch_in_at_bat if it doesn't exist
        if 'pitch_in_at_bat' not in career_df.columns:
            career_df['pitch_in_at_bat'] = career_df.groupby('at_bat_number').cumcount() + 1
        
        # Overall swing rate
        overall_swing_rate = career_df['is_swing'].mean()
        df[f'{hitter_name}_overall_swing_rate'] = overall_swing_rate
        
        # Zone-specific swing rates
        for zone in range(1, 15):
            zone_data = career_df[career_df['zone'] == zone]
            if len(zone_data) > 0:
                zone_swing_rate = zone_data['is_swing'].mean()
                df[f'{hitter_name}_zone_{zone}_swing_rate'] = zone_swing_rate
            else:
                df[f'{hitter_name}_zone_{zone}_swing_rate'] = overall_swing_rate
        
        # Pitch type swing rates
        for pitch_type in career_df['pitch_type'].unique():
            pitch_data = career_df[career_df['pitch_type'] == pitch_type]
            if len(pitch_data) > 0:
                pitch_swing_rate = pitch_data['is_swing'].mean()
                df[f'{hitter_name}_{pitch_type}_swing_rate'] = pitch_swing_rate
            else:
                df[f'{hitter_name}_{pitch_type}_swing_rate'] = overall_swing_rate
        
        # Count-specific swing rates
        for balls in range(4):
            for strikes in range(3):
                count_data = career_df[(career_df['balls'] == balls) & (career_df['strikes'] == strikes)]
                if len(count_data) > 0:
                    count_swing_rate = count_data['is_swing'].mean()
                    df[f'{hitter_name}_count_{balls}_{strikes}_swing_rate'] = count_swing_rate
                else:
                    df[f'{hitter_name}_count_{balls}_{strikes}_swing_rate'] = overall_swing_rate
        
        # Situational swing rates
        # Ahead in count
        ahead_data = career_df[career_df['balls'] > career_df['strikes']]
        if len(ahead_data) > 0:
            ahead_swing_rate = ahead_data['is_swing'].mean()
            df[f'{hitter_name}_ahead_swing_rate'] = ahead_swing_rate
        else:
            df[f'{hitter_name}_ahead_swing_rate'] = overall_swing_rate
        
        # Behind in count
        behind_data = career_df[career_df['strikes'] > career_df['balls']]
        if len(behind_data) > 0:
            behind_swing_rate = behind_data['is_swing'].mean()
            df[f'{hitter_name}_behind_swing_rate'] = behind_swing_rate
        else:
            df[f'{hitter_name}_behind_swing_rate'] = overall_swing_rate
        
        # Two strikes
        two_strikes_data = career_df[career_df['strikes'] >= 2]
        if len(two_strikes_data) > 0:
            two_strikes_swing_rate = two_strikes_data['is_swing'].mean()
            df[f'{hitter_name}_two_strikes_swing_rate'] = two_strikes_swing_rate
        else:
            df[f'{hitter_name}_two_strikes_swing_rate'] = overall_swing_rate
        
        # Velocity-based swing rates
        low_vel_data = career_df[career_df['release_speed'] < 85]
        if len(low_vel_data) > 0:
            low_vel_swing_rate = low_vel_data['is_swing'].mean()
            df[f'{hitter_name}_low_velocity_swing_rate'] = low_vel_swing_rate
        else:
            df[f'{hitter_name}_low_velocity_swing_rate'] = overall_swing_rate
        
        high_vel_data = career_df[career_df['release_speed'] > 95]
        if len(high_vel_data) > 0:
            high_vel_swing_rate = high_vel_data['is_swing'].mean()
            df[f'{hitter_name}_high_velocity_swing_rate'] = high_vel_swing_rate
        else:
            df[f'{hitter_name}_high_velocity_swing_rate'] = overall_swing_rate
        
        # Zone location swing rates
        heart_data = career_df[((career_df['zone'] == 2) | (career_df['zone'] == 5) | (career_df['zone'] == 8))]
        if len(heart_data) > 0:
            heart_swing_rate = heart_data['is_swing'].mean()
            df[f'{hitter_name}_zone_heart_swing_rate'] = heart_swing_rate
        else:
            df[f'{hitter_name}_zone_heart_swing_rate'] = overall_swing_rate
        
        corner_data = career_df[((career_df['zone'] == 1) | (career_df['zone'] == 3) | (career_df['zone'] == 7) | (career_df['zone'] == 9))]
        if len(corner_data) > 0:
            corner_swing_rate = corner_data['is_swing'].mean()
            df[f'{hitter_name}_zone_corner_swing_rate'] = corner_swing_rate
        else:
            df[f'{hitter_name}_zone_corner_swing_rate'] = overall_swing_rate
        
        shadow_data = career_df[((career_df['zone'] == 4) | (career_df['zone'] == 6))]
        if len(shadow_data) > 0:
            shadow_swing_rate = shadow_data['is_swing'].mean()
            df[f'{hitter_name}_zone_shadow_swing_rate'] = shadow_swing_rate
        else:
            df[f'{hitter_name}_zone_shadow_swing_rate'] = overall_swing_rate
        
        # Additional features that were missing
        # Fastball swing rate (combine FF, SI, FC)
        fastball_data = career_df[career_df['pitch_type'].isin(['FF', 'SI', 'FC'])]
        if len(fastball_data) > 0:
            fastball_swing_rate = fastball_data['is_swing'].mean()
            df[f'{hitter_name}_fastball_swing_rate'] = fastball_swing_rate
        else:
            df[f'{hitter_name}_fastball_swing_rate'] = overall_swing_rate
        
        # Breaking ball swing rate (combine SL, CU, KC)
        breaking_data = career_df[career_df['pitch_type'].isin(['SL', 'CU', 'KC'])]
        if len(breaking_data) > 0:
            breaking_swing_rate = breaking_data['is_swing'].mean()
            df[f'{hitter_name}_breaking_swing_rate'] = breaking_swing_rate
        else:
            df[f'{hitter_name}_breaking_swing_rate'] = breaking_swing_rate
        
        # Offspeed swing rate (combine CH, FS)
        offspeed_data = career_df[career_df['pitch_type'].isin(['CH', 'FS'])]
        if len(offspeed_data) > 0:
            offspeed_swing_rate = offspeed_data['is_swing'].mean()
            df[f'{hitter_name}_offspeed_swing_rate'] = offspeed_swing_rate
        else:
            df[f'{hitter_name}_offspeed_swing_rate'] = overall_swing_rate
        
        # Zone swing rate (combine all zones)
        zone_data = career_df[career_df['zone'].between(1, 9)]
        if len(zone_data) > 0:
            zone_swing_rate = zone_data['is_swing'].mean()
            df[f'{hitter_name}_zone_swing_rate'] = zone_swing_rate
        else:
            df[f'{hitter_name}_zone_swing_rate'] = overall_swing_rate
        
        # Outside zone swing rate (zones 11-14)
        outside_data = career_df[career_df['zone'].between(11, 14)]
        if len(outside_data) > 0:
            outside_swing_rate = outside_data['is_swing'].mean()
            df[f'{hitter_name}_outside_swing_rate'] = outside_swing_rate
        else:
            df[f'{hitter_name}_outside_swing_rate'] = overall_swing_rate
        
        # Additional missing features
        # High/low swing rates
        high_data = career_df[career_df['plate_z'] > career_df['sz_top']]
        if len(high_data) > 0:
            high_swing_rate = high_data['is_swing'].mean()
            df[f'{hitter_name}_high_swing_rate'] = high_swing_rate
        else:
            df[f'{hitter_name}_high_swing_rate'] = overall_swing_rate
        
        low_data = career_df[career_df['plate_z'] < career_df['sz_bot']]
        if len(low_data) > 0:
            low_swing_rate = low_data['is_swing'].mean()
            df[f'{hitter_name}_low_swing_rate'] = low_swing_rate
        else:
            df[f'{hitter_name}_low_swing_rate'] = overall_swing_rate
        
        # Full count swing rate
        full_count_data = career_df[(career_df['balls'] == 3) & (career_df['strikes'] == 2)]
        if len(full_count_data) > 0:
            full_count_swing_rate = full_count_data['is_swing'].mean()
            df[f'{hitter_name}_full_count_swing_rate'] = full_count_swing_rate
        else:
            df[f'{hitter_name}_full_count_swing_rate'] = overall_swing_rate
        
        # High/low velocity swing rates (different naming)
        high_vel_data = career_df[career_df['release_speed'] > 95]
        if len(high_vel_data) > 0:
            high_vel_swing_rate = high_vel_data['is_swing'].mean()
            df[f'{hitter_name}_high_vel_swing_rate'] = high_vel_swing_rate
        else:
            df[f'{hitter_name}_high_vel_swing_rate'] = overall_swing_rate
        
        low_vel_data = career_df[career_df['release_speed'] < 85]
        if len(low_vel_data) > 0:
            low_vel_swing_rate = low_vel_data['is_swing'].mean()
            df[f'{hitter_name}_low_vel_swing_rate'] = low_vel_swing_rate
        else:
            df[f'{hitter_name}_low_vel_swing_rate'] = overall_swing_rate
        
        # Movement-based swing rates
        career_df['movement_magnitude'] = np.sqrt(
            career_df['api_break_x_batter_in'].fillna(0)**2 + 
            career_df['api_break_z_with_gravity'].fillna(0)**2
        )
        
        high_movement_data = career_df[career_df['movement_magnitude'] > 8]
        if len(high_movement_data) > 0:
            high_movement_swing_rate = high_movement_data['is_swing'].mean()
            df[f'{hitter_name}_high_movement_swing_rate'] = high_movement_swing_rate
        else:
            df[f'{hitter_name}_high_movement_swing_rate'] = overall_swing_rate
        
        low_movement_data = career_df[career_df['movement_magnitude'] < 4]
        if len(low_movement_data) > 0:
            low_movement_swing_rate = low_movement_data['is_swing'].mean()
            df[f'{hitter_name}_low_movement_swing_rate'] = low_movement_swing_rate
        else:
            df[f'{hitter_name}_low_movement_swing_rate'] = overall_swing_rate
        
        # Situational swing rates
        # First pitch
        first_pitch_data = career_df[career_df['pitch_in_at_bat'] == 1]
        if len(first_pitch_data) > 0:
            first_pitch_swing_rate = first_pitch_data['is_swing'].mean()
            df[f'{hitter_name}_first_pitch_swing_rate'] = first_pitch_swing_rate
        else:
            df[f'{hitter_name}_first_pitch_swing_rate'] = overall_swing_rate
        
        # Last pitch (pitch 5+ in at bat)
        last_pitch_data = career_df[career_df['pitch_in_at_bat'] >= 5]
        if len(last_pitch_data) > 0:
            last_pitch_swing_rate = last_pitch_data['is_swing'].mean()
            df[f'{hitter_name}_last_pitch_swing_rate'] = last_pitch_swing_rate
        else:
            df[f'{hitter_name}_last_pitch_swing_rate'] = overall_swing_rate
        
        # Velocity change swing rates
        avg_velocity = career_df['release_speed'].mean()
        velocity_drop_data = career_df[career_df['release_speed'] < (avg_velocity - 5)]
        if len(velocity_drop_data) > 0:
            velocity_drop_swing_rate = velocity_drop_data['is_swing'].mean()
            df[f'{hitter_name}_velocity_drop_swing_rate'] = velocity_drop_swing_rate
        else:
            df[f'{hitter_name}_velocity_drop_swing_rate'] = overall_swing_rate
        
        velocity_surge_data = career_df[career_df['release_speed'] > (avg_velocity + 5)]
        if len(velocity_surge_data) > 0:
            velocity_surge_swing_rate = velocity_surge_data['is_swing'].mean()
            df[f'{hitter_name}_velocity_surge_swing_rate'] = velocity_surge_swing_rate
        else:
            df[f'{hitter_name}_velocity_surge_swing_rate'] = overall_swing_rate
        
        # Pitch type change swing rate (simplified)
        pitch_type_counts = career_df['pitch_type'].value_counts()
        most_common_pitch = pitch_type_counts.index[0] if len(pitch_type_counts) > 0 else 'FF'
        non_most_common_data = career_df[career_df['pitch_type'] != most_common_pitch]
        if len(non_most_common_data) > 0:
            pitch_type_change_swing_rate = non_most_common_data['is_swing'].mean()
            df[f'{hitter_name}_pitch_type_change_swing_rate'] = pitch_type_change_swing_rate
        else:
            df[f'{hitter_name}_pitch_type_change_swing_rate'] = overall_swing_rate
        
        # Inning-based swing rates
        if 'inning' in career_df.columns:
            late_inning_data = career_df[career_df['inning'] >= 7]
            if len(late_inning_data) > 0:
                late_inning_swing_rate = late_inning_data['is_swing'].mean()
                df[f'{hitter_name}_late_inning_swing_rate'] = late_inning_swing_rate
            else:
                df[f'{hitter_name}_late_inning_swing_rate'] = overall_swing_rate
        else:
            df[f'{hitter_name}_late_inning_swing_rate'] = overall_swing_rate
        
        # Game situation swing rates
        if 'home_score' in career_df.columns and 'away_score' in career_df.columns:
            close_game_data = career_df[abs(career_df['home_score'] - career_df['away_score']) <= 2]
            if len(close_game_data) > 0:
                close_game_swing_rate = close_game_data['is_swing'].mean()
                df[f'{hitter_name}_close_game_swing_rate'] = close_game_swing_rate
            else:
                df[f'{hitter_name}_close_game_swing_rate'] = overall_swing_rate
        else:
            df[f'{hitter_name}_close_game_swing_rate'] = overall_swing_rate
        
        # Location-based swing rates
        # Extreme location (far from zone)
        extreme_location_data = career_df[
            (abs(career_df['plate_x']) > 1.2) | 
            (career_df['plate_z'] > career_df['sz_top'] + 0.5) | 
            (career_df['plate_z'] < career_df['sz_bot'] - 0.5)
        ]
        if len(extreme_location_data) > 0:
            location_extreme_swing_rate = extreme_location_data['is_swing'].mean()
            df[f'{hitter_name}_location_extreme_swing_rate'] = location_extreme_swing_rate
        else:
            df[f'{hitter_name}_location_extreme_swing_rate'] = overall_swing_rate
        
        # Heart location (center of zone)
        heart_location_data = career_df[
            (abs(career_df['plate_x']) <= 0.5) & 
            (career_df['plate_z'] >= 2.0) & 
            (career_df['plate_z'] <= 3.0)
        ]
        if len(heart_location_data) > 0:
            location_heart_swing_rate = heart_location_data['is_swing'].mean()
            df[f'{hitter_name}_location_heart_swing_rate'] = location_heart_swing_rate
        else:
            df[f'{hitter_name}_location_heart_swing_rate'] = overall_swing_rate
        
        # Pressure situation swing rates
        pressure_data = career_df[career_df['strikes'] >= 2]
        if len(pressure_data) > 0:
            pressure_swing_rate = pressure_data['is_swing'].mean()
            df[f'{hitter_name}_pressure_swing_rate'] = pressure_swing_rate
        else:
            df[f'{hitter_name}_pressure_swing_rate'] = overall_swing_rate
        
        # Opportunity situation swing rates
        opportunity_data = career_df[career_df['balls'] > career_df['strikes']]
        if len(opportunity_data) > 0:
            opportunity_swing_rate = opportunity_data['is_swing'].mean()
            df[f'{hitter_name}_opportunity_swing_rate'] = opportunity_swing_rate
        else:
            df[f'{hitter_name}_opportunity_swing_rate'] = overall_swing_rate
        
        # Zone-specific contact rate features (for outcome prediction)
        hit_events = ['single', 'double', 'triple', 'home_run']
        
        # Zone heart contact rate
        zone_heart_swings = career_df[
            ((abs(career_df['plate_x']) <= 0.5) & 
             (career_df['plate_z'] >= 2.0) & 
             (career_df['plate_z'] <= 3.0)) & 
            (career_df['is_swing'] == 1)
        ]
        zone_heart_hits = zone_heart_swings[zone_heart_swings['events'].isin(hit_events)]
        if len(zone_heart_swings) > 0:
            zone_heart_hit_rate = len(zone_heart_hits) / len(zone_heart_swings)
            df[f'{hitter_name}_zone_heart_hit_rate'] = zone_heart_hit_rate
        else:
            df[f'{hitter_name}_zone_heart_hit_rate'] = 0.0
        
        # Zone corner contact rate
        zone_corner_swings = career_df[
            ((abs(career_df['plate_x']) >= 0.7) | 
             (career_df['plate_z'] >= 3.5) | 
             (career_df['plate_z'] <= 1.5)) & 
            (career_df['is_swing'] == 1)
        ]
        zone_corner_hits = zone_corner_swings[zone_corner_swings['events'].isin(hit_events)]
        if len(zone_corner_swings) > 0:
            zone_corner_hit_rate = len(zone_corner_hits) / len(zone_corner_swings)
            df[f'{hitter_name}_zone_corner_hit_rate'] = zone_corner_hit_rate
        else:
            df[f'{hitter_name}_zone_corner_hit_rate'] = 0.0
        
        # Zone shadow contact rate
        zone_shadow_swings = career_df[
            ((abs(career_df['plate_x']) <= 1.0) & 
             (career_df['plate_z'] >= 1.2) & 
             (career_df['plate_z'] <= 3.8)) & 
            ~((abs(career_df['plate_x']) <= 0.7) & 
              (career_df['plate_z'] >= 1.5) & 
              (career_df['plate_z'] <= 3.5)) & 
            (career_df['is_swing'] == 1)
        ]
        zone_shadow_hits = zone_shadow_swings[zone_shadow_swings['events'].isin(hit_events)]
        if len(zone_shadow_swings) > 0:
            zone_shadow_hit_rate = len(zone_shadow_hits) / len(zone_shadow_swings)
            df[f'{hitter_name}_zone_shadow_hit_rate'] = zone_shadow_hit_rate
        else:
            df[f'{hitter_name}_zone_shadow_hit_rate'] = 0.0
        
        # Overall zone contact rate
        zone_swings = career_df[
            ((abs(career_df['plate_x']) <= 0.7) & 
             (career_df['plate_z'] >= 1.5) & 
             (career_df['plate_z'] <= 3.5)) & 
            (career_df['is_swing'] == 1)
        ]
        zone_hits = zone_swings[zone_swings['events'].isin(hit_events)]
        if len(zone_swings) > 0:
            zone_hit_rate = len(zone_hits) / len(zone_swings)
            df[f'{hitter_name}_zone_hit_rate'] = zone_hit_rate
        else:
            df[f'{hitter_name}_zone_hit_rate'] = 0.0
        
        # Outside zone contact rate
        outside_swings = career_df[
            (abs(career_df['plate_x']) > 0.7) & 
            (career_df['is_swing'] == 1)
        ]
        outside_hits = outside_swings[outside_swings['events'].isin(hit_events)]
        if len(outside_swings) > 0:
            outside_hit_rate = len(outside_hits) / len(outside_swings)
            df[f'{hitter_name}_outside_hit_rate'] = outside_hit_rate
        else:
            df[f'{hitter_name}_outside_hit_rate'] = 0.0
        
        print(f"✓ Added {len([col for col in df.columns if col.startswith(f'{hitter_name}_')])} hitter features")
        
    except Exception as e:
        print(f"⚠️ Error calculating hitter features: {e}")
        print("Using default hitter features...")
        
        # Add default values for all expected features
        default_features = [
            f'{hitter_name}_fastball_swing_rate', f'{hitter_name}_breaking_swing_rate', f'{hitter_name}_offspeed_swing_rate',
            f'{hitter_name}_zone_swing_rate', f'{hitter_name}_outside_swing_rate', f'{hitter_name}_high_swing_rate', f'{hitter_name}_low_swing_rate',
            f'{hitter_name}_ahead_swing_rate', f'{hitter_name}_behind_swing_rate', f'{hitter_name}_two_strikes_swing_rate', f'{hitter_name}_full_count_swing_rate',
            f'{hitter_name}_high_vel_swing_rate', f'{hitter_name}_low_vel_swing_rate',
            f'{hitter_name}_high_movement_swing_rate', f'{hitter_name}_low_movement_swing_rate',
            f'{hitter_name}_late_inning_swing_rate', f'{hitter_name}_close_game_swing_rate',
            f'{hitter_name}_first_pitch_swing_rate', f'{hitter_name}_last_pitch_swing_rate',
            f'{hitter_name}_pitch_type_change_swing_rate', f'{hitter_name}_velocity_drop_swing_rate', f'{hitter_name}_velocity_surge_swing_rate',
            f'{hitter_name}_location_extreme_swing_rate', f'{hitter_name}_location_heart_swing_rate',
            f'{hitter_name}_pressure_swing_rate', f'{hitter_name}_opportunity_swing_rate',
            f'{hitter_name}_zone_corner_swing_rate', f'{hitter_name}_zone_shadow_swing_rate', f'{hitter_name}_zone_heart_swing_rate',
            # Contact rate features
            f'{hitter_name}_zone_heart_hit_rate', f'{hitter_name}_zone_corner_hit_rate', f'{hitter_name}_zone_shadow_hit_rate',
            f'{hitter_name}_zone_hit_rate', f'{hitter_name}_outside_hit_rate'
        ]
        
        for feature in default_features:
            df[feature] = 0.5  # Default 50% rate
    
    return df

def calculate_pitcher_features(df, pitcher_name='unknown'):
    """Calculate pitcher-specific features from career data"""
    df = df.copy()
    
    try:
        # Load career data for the pitcher
        career_file = f'sandy_alcántara_complete_career_statcast.csv'
        # Try alternative filename if the first one doesn't exist
        if not os.path.exists(career_file):
            career_file = f'sandy_alcantara_complete_career_statcast.csv'
        career_df = pd.read_csv(career_file)
        
        # Calculate pitcher averages
        # Pitch type usage rates
        for pitch_type in career_df['pitch_type'].unique():
            pitch_usage = (career_df['pitch_type'] == pitch_type).mean()
            df[f'{pitcher_name}_{pitch_type}_usage'] = pitch_usage
        
        # Zone usage rates
        for zone in range(1, 15):
            zone_data = career_df[career_df['zone'] == zone]
            if len(zone_data) > 0:
                zone_usage = (career_df['zone'] == zone).mean()
                df[f'{pitcher_name}_zone_{zone}_usage'] = zone_usage
            else:
                df[f'{pitcher_name}_zone_{zone}_usage'] = 0.0
        
        # Velocity averages
        avg_velocity = career_df['release_speed'].mean()
        df[f'{pitcher_name}_avg_velocity'] = avg_velocity
        
        # Movement averages
        career_df['movement_magnitude'] = np.sqrt(
            career_df['api_break_x_batter_in'].fillna(0)**2 + 
            career_df['api_break_z_with_gravity'].fillna(0)**2
        )
        avg_movement = career_df['movement_magnitude'].mean()
        df[f'{pitcher_name}_avg_movement'] = avg_movement
        
        # Count-specific tendencies
        for balls in range(4):
            for strikes in range(3):
                count_data = career_df[(career_df['balls'] == balls) & (career_df['strikes'] == strikes)]
                if len(count_data) > 0:
                    count_velocity = count_data['release_speed'].mean()
                    df[f'{pitcher_name}_count_{balls}_{strikes}_avg_velocity'] = count_velocity
                else:
                    df[f'{pitcher_name}_count_{balls}_{strikes}_avg_velocity'] = avg_velocity
        
        print(f"✓ Added {pitcher_name} features from career data")
        
    except Exception as e:
        print(f"⚠️ Error loading {pitcher_name} career data: {e}")
        # Add default values
        for pitch_type in ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'ST', 'KC', 'FS']:
            df[f'{pitcher_name}_{pitch_type}_usage'] = 0.1
        for zone in range(1, 15):
            df[f'{pitcher_name}_zone_{zone}_usage'] = 0.07
        df[f'{pitcher_name}_avg_velocity'] = 92.0
        df[f'{pitcher_name}_avg_movement'] = 3.0
        for balls in range(4):
            for strikes in range(3):
                df[f'{pitcher_name}_count_{balls}_{strikes}_avg_velocity'] = 92.0
    
    return df

def calculate_comprehensive_count_features(df, hitter_name='acuna'):
    """Calculate comprehensive count-specific features"""
    df = df.copy()
    
    try:
        # Load career data
        career_file = f'ronald_acuna_jr_complete_career_statcast.csv'
        career_df = pd.read_csv(career_file)
        
        # Create swing column
        swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
        career_df['is_swing'] = career_df['description'].isin(swing_events).astype(int)
        
        # Calculate count-specific swing rates for all counts and pitch types
        for balls in range(4):
            for strikes in range(3):
                count_data = career_df[(career_df['balls'] == balls) & (career_df['strikes'] == strikes)]
                if len(count_data) > 0:
                    # Overall swing rate for this count
                    count_swing_rate = count_data['is_swing'].mean()
                    df[f'{hitter_name}_count_{balls}_{strikes}_swing_rate'] = count_swing_rate
                    
                    # Count-specific swing rates by pitch type
                    for pitch_type in count_data['pitch_type'].unique():
                        pitch_count_data = count_data[count_data['pitch_type'] == pitch_type]
                        if len(pitch_count_data) > 0:
                            pitch_count_swing_rate = pitch_count_data['is_swing'].mean()
                            df[f'{hitter_name}_count_{balls}_{strikes}_{pitch_type}_swing_rate'] = pitch_count_swing_rate
                        else:
                            df[f'{hitter_name}_count_{balls}_{strikes}_{pitch_type}_swing_rate'] = count_swing_rate
                    
                    # Weighted advantage count features
                    if (balls >= 2 and strikes <= 1) or (balls >= 3):  # Advantage counts
                        df[f'{hitter_name}_advantage_count_{balls}_{strikes}_swing_rate'] = count_swing_rate * 1.2  # Weighted higher
                    else:
                        df[f'{hitter_name}_advantage_count_{balls}_{strikes}_swing_rate'] = count_swing_rate
                else:
                    # Default values if no data
                    df[f'{hitter_name}_count_{balls}_{strikes}_swing_rate'] = 0.45
                    for pitch_type in ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'ST', 'KC', 'FS']:
                        df[f'{hitter_name}_count_{balls}_{strikes}_{pitch_type}_swing_rate'] = 0.45
                    df[f'{hitter_name}_advantage_count_{balls}_{strikes}_swing_rate'] = 0.45
        
        print(f"✓ Added comprehensive count features for {hitter_name}")
        
    except Exception as e:
        print(f"⚠️ Error calculating comprehensive count features: {e}")
        # Add default values
        for balls in range(4):
            for strikes in range(3):
                df[f'{hitter_name}_count_{balls}_{strikes}_swing_rate'] = 0.45
                for pitch_type in ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'ST', 'KC', 'FS']:
                    df[f'{hitter_name}_count_{balls}_{strikes}_{pitch_type}_swing_rate'] = 0.45
                df[f'{hitter_name}_advantage_count_{balls}_{strikes}_swing_rate'] = 0.45
    
    return df

def calculate_contact_rate_features(df, hitter_name='acuna'):
    """Calculate contact rate features (proxy for hit rates)"""
    df = df.copy()
    
    # Contact rate features (combining zone location with swing rates)
    df[f'{hitter_name}_zone_heart_hit_rate'] = df[f'{hitter_name}_zone_heart_swing_rate'] * 0.7  # 70% contact rate in heart
    df[f'{hitter_name}_zone_corner_hit_rate'] = df[f'{hitter_name}_zone_corner_swing_rate'] * 0.6  # 60% contact rate in corners
    df[f'{hitter_name}_zone_shadow_hit_rate'] = df[f'{hitter_name}_zone_shadow_swing_rate'] * 0.5  # 50% contact rate in shadows
    
    # Pitch type contact rates
    for pitch_type in ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'ST', 'KC', 'FS']:
        if f'{hitter_name}_{pitch_type}_swing_rate' in df.columns:
            # Different contact rates by pitch type
            if pitch_type in ['FF', 'SI', 'FC']:  # Fastballs
                contact_rate = 0.75
            elif pitch_type in ['SL', 'CU', 'KC']:  # Breaking balls
                contact_rate = 0.65
            else:  # Offspeed
                contact_rate = 0.70
            
            df[f'{hitter_name}_{pitch_type}_hit_rate'] = df[f'{hitter_name}_{pitch_type}_swing_rate'] * contact_rate
    
    print(f"✓ Added contact rate features for {hitter_name}")
    
    return df

def engineer_features_for_model(df, model_type='whiff_vs_contact', hitter_name='acuna', pitcher_name='unknown'):
    """Engineer features for a specific model type"""
    print(f"🔧 FEATURE ENGINEERING MODULE")
    print("=" * 50)
    
    # Sample data for debugging
    print(f"Sample data shape: {df.shape}")
    if len(df) > 0:
        print(f"Sample columns: {list(df.columns[:20])}")
    
    # Calculate basic features
    df = calculate_basic_features(df, model_type)
    
    # Add BABIP features
    df = calculate_babip_features(df)
    
    # Add hitter-specific features
    try:
        df = calculate_hitter_features(df, hitter_name)
    except Exception as e:
        print(f"⚠️ Error calculating hitter features: {e}")
        print("Using default hitter features...")
    
    # Add unknown features from career data
    try:
        df = calculate_comprehensive_count_features(df, hitter_name)
    except Exception as e:
        print(f"⚠️ Error calculating comprehensive count features: {e}")
        print("Using default count features...")
    
    # Add contact rate features
    try:
        df = calculate_contact_rate_features(df, hitter_name)
    except Exception as e:
        print(f"⚠️ Error calculating contact rate features: {e}")
        print("Using default contact rate features...")
    
    # Add missing features that the training script expects
    print("Adding missing features to align with model expectations...")
    
    # Add BABIP and whiff rate features for outcome prediction
    if 'batting_average_bip' not in df.columns:
        df['batting_average_bip'] = 0.25  # Default BABIP
    if 'whiff_rate' not in df.columns:
        df['whiff_rate'] = 0.35  # Default whiff rate
    if 'field_out_rate_bip' not in df.columns:
        df['field_out_rate_bip'] = 0.40  # Default field out rate
    if 'balls_in_play' not in df.columns:
        df['balls_in_play'] = 0  # No balls in play data
    if 'total_swings' not in df.columns:
        df['total_swings'] = 0  # No swing data
    if 'total_whiffs' not in df.columns:
        df['total_whiffs'] = 0  # No whiff data
    
    # Add proxy contact features that combine zone location with swing rates
    # These are especially useful for outcome prediction
    if f'{hitter_name}_zone_heart_swing_rate' in df.columns:
        df['zone_heart_contact'] = df['zone_heart'] * df[f'{hitter_name}_zone_heart_swing_rate']
    else:
        df['zone_heart_contact'] = 0.0
    
    if f'{hitter_name}_zone_corner_swing_rate' in df.columns:
        df['zone_corner_contact'] = df['zone_corner'] * df[f'{hitter_name}_zone_corner_swing_rate']
    else:
        df['zone_corner_contact'] = 0.0
    
    if f'{hitter_name}_zone_shadow_swing_rate' in df.columns:
        df['zone_shadow_contact'] = df['zone_shadow'] * df[f'{hitter_name}_zone_shadow_swing_rate']
    else:
        df['zone_shadow_contact'] = 0.0
    
    if f'{hitter_name}_zone_swing_rate' in df.columns:
        df['zone_overall_contact'] = df['in_strike_zone'] * df[f'{hitter_name}_zone_swing_rate']
    else:
        df['zone_overall_contact'] = 0.0
    
    if f'{hitter_name}_outside_swing_rate' in df.columns:
        df['outside_zone_contact'] = df['far_from_zone'] * df[f'{hitter_name}_outside_swing_rate']
    else:
        df['outside_zone_contact'] = 0.0
    
    # Additional contact features for different pitch types
    if f'{hitter_name}_zone_swing_rate' in df.columns:
        df['fastball_zone_contact'] = df['is_fastball'] * df['zone_overall_contact']
        df['breaking_zone_contact'] = df['is_breaking_ball'] * df['zone_overall_contact']
        df['offspeed_zone_contact'] = df['is_offspeed'] * df['zone_overall_contact']
    else:
        df['fastball_zone_contact'] = 0.0
        df['breaking_zone_contact'] = 0.0
        df['offspeed_zone_contact'] = 0.0
    
    # Contact features for different count situations
    if f'{hitter_name}_zone_swing_rate' in df.columns:
        df['pressure_zone_contact'] = df['pressure_situation'] * df['zone_overall_contact']
        df['opportunity_zone_contact'] = df['ahead_in_count'] * df['zone_overall_contact']
        df['two_strikes_zone_contact'] = df['two_strikes'] * df['zone_overall_contact']
    else:
        df['pressure_zone_contact'] = 0.0
        df['opportunity_zone_contact'] = 0.0
        df['two_strikes_zone_contact'] = 0.0
    
    # Add missing outcome-specific features
    df['zone_whiff_rate'] = 0.0  # Will be calculated from career data
    df['zone_hit_rate'] = 0.0
    df['zone_field_out_rate'] = 0.0
    
    # Count-specific outcome features
    df['count_whiff_rate'] = 0.0
    df['count_hit_rate'] = 0.0
    df['count_field_out_rate'] = 0.0
    
    # Pitch type specific outcome features
    df['pitch_type_whiff_rate'] = 0.0
    df['pitch_type_hit_rate'] = 0.0
    df['pitch_type_field_out_rate'] = 0.0
    
    # Pressure situation features
    df['pressure_whiff_rate'] = 0.0
    df['pressure_hit_rate'] = 0.0
    df['pressure_field_out_rate'] = 0.0
    
    # Add missing count-based features
    df['count_field_out_rate'] = 0.0
    df['count_hit_rate'] = 0.0
    df['count_swing_rate_adjustment'] = 0.0
    df['count_whiff_rate'] = 0.0
    
    # Add missing pitch type rates
    df['pitch_type_field_out_rate'] = 0.0
    df['pitch_type_hit_rate'] = 0.0
    df['pitch_type_whiff_rate'] = 0.0
    
    # Add missing comprehensive count features
    count_features = [
        'count_0_0_swing_rate', 'count_1_0_swing_rate', 'count_0_1_swing_rate',
        'count_2_0_swing_rate', 'count_1_1_swing_rate', 'count_0_2_swing_rate',
        'count_3_0_swing_rate', 'count_2_1_swing_rate', 'count_1_2_swing_rate',
        'count_3_1_swing_rate', 'count_2_2_swing_rate', 'count_3_2_swing_rate'
    ]
    
    for feature in count_features:
        if f'{hitter_name}_{feature}' not in df.columns:
            df[f'{hitter_name}_{feature}'] = 0.5  # Default 50% rate
    
    print(f"✓ Engineered features for {model_type} model. DataFrame has {len(df.columns)} columns")
    print(f"Engineered data shape: {df.shape}")
    print(f"Sample engineered features:")
    print(list(df.columns[:20]))
    
    return df

def get_model_features(model_type='whiff_vs_contact'):
    """Get the list of features required for a specific model"""
    
    # Base features (always included)
    base_features = [
        'release_speed', 'release_spin_rate', 'plate_x', 'plate_z', 'zone',
        'sz_top', 'sz_bot', 'balls', 'strikes', 'pitch_type',
        'api_break_x_batter_in', 'api_break_z_with_gravity',
        'zone_distance', 'movement_magnitude', 'count_pressure', 'count_total',
        'behind_in_count', 'ahead_in_count', 'two_strikes', 'three_balls',
        'in_strike_zone', 'far_from_zone', 'high_pitch', 'low_pitch',
        'inside_pitch', 'outside_pitch', 'is_fastball', 'is_breaking_ball',
        'is_offspeed', 'high_velocity', 'low_velocity', 'velocity_movement_ratio',
        'high_movement', 'low_movement', 'movement_ratio',
        'zone_distance_x_count_pressure', 'movement_x_count_pressure',
        'in_zone_x_two_strikes', 'far_from_zone_x_ahead',
        'velocity_diff_from_avg', 'movement_diff_from_avg',
        'zone_corner', 'zone_heart', 'zone_shadow',
        'batting_average_bip', 'whiff_rate', 'field_out_rate_bip',
        'balls_in_play', 'total_swings', 'total_whiffs'
    ]
    
    # Additional features for swing vs no swing model
    if model_type == 'swing_vs_noswing':
        swing_features = [
            'arm_side_break', 'movement_direction', 'spin_axis_rad', 'spin_efficiency',
            'movement_per_mph', 'expected_movement', 'movement_deception',
            'plate_x_norm', 'plate_z_norm', 'plate_x_norm_x_movement', 'plate_z_norm_x_movement',
            'early_count', 'middle_count', 'late_count', 'pressure_count',
            'early_count_penalty', 'early_count_zone_penalty', 'early_count_outside_penalty',
            'count_swing_rate_adjustment', 'early_count_location_penalty',
            'early_count_breaking_penalty', 'early_count_offspeed_penalty',
            'zone_edge_distance', 'zone_top_distance', 'zone_bottom_distance', 'closest_zone_edge',
            'full_count', 'hitters_count', 'pitchers_count', 'neutral_count',
            'zone_quadrant', 'velocity_drop', 'breaking_ball_high', 'offspeed_low',
            'inning_late', 'close_game', 'pitch_in_at_bat', 'first_pitch', 'last_pitch',
            'velocity_bin', 'velocity_std', 'horizontal_movement', 'vertical_movement',
            'high_horizontal_movement', 'high_vertical_movement', 'zone_center_distance',
            'count_advantage', 'pressure_situation', 'must_swing', 'can_take',
            'fastball_high', 'breaking_ball_low', 'offspeed_middle', 'pitch_type_change',
            'location_quadrant', 'location_extreme', 'spin_movement_correlation',
            'high_spin', 'low_spin', 'late_inning', 'close_score', 'high_leverage',
            'count_ratio', 'behind_by_two', 'ahead_by_two', 'full_count_pressure',
            'in_zone_two_strikes', 'out_zone_ahead', 'edge_zone_decision',
            'velocity_surprise', 'velocity_consistency', 'breaking_ball_velocity',
            'high_movement_fastball', 'low_movement_breaking', 'unexpected_movement',
            'corner_pitch', 'heart_pitch', 'shadow_pitch', 'extreme_location',
            'velocity_x_location', 'pitch_type_x_location', 'count_x_zone',
            'early_count_swing', 'late_count_take', 'pressure_swing', 'opportunity_take',
            'early_count_low_vel_penalty', 'early_count_high_vel_penalty',
            'zone_quadrant_encoded', 'location_quadrant_encoded', 'count_advantage_encoded',
            'count_field_out_rate', 'count_hit_rate', 'count_whiff_rate',
            'pitch_type_field_out_rate', 'pitch_type_hit_rate', 'pitch_type_whiff_rate'
        ]
        base_features.extend(swing_features)
    
    # Hitter features
    hitter_features = []
    for zone in range(1, 15):
        hitter_features.append(f'acuna_zone_{zone}_swing_rate')
    for pitch_type in ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'ST', 'KC', 'FS']:
        hitter_features.append(f'acuna_{pitch_type}_swing_rate')
    for balls in range(4):
        for strikes in range(3):
            hitter_features.append(f'acuna_count_{balls}_{strikes}_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_FF_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_SI_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_SL_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_CU_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_CH_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_FC_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_ST_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_KC_swing_rate')
            hitter_features.append(f'acuna_count_{balls}_{strikes}_FS_swing_rate')
            hitter_features.append(f'acuna_advantage_count_{balls}_{strikes}_swing_rate')
    
    # Additional hitter features
    additional_hitter_features = [
        'acuna_overall_swing_rate', 'acuna_ahead_swing_rate', 'acuna_behind_swing_rate',
        'acuna_two_strikes_swing_rate', 'acuna_low_velocity_swing_rate',
        'acuna_high_velocity_swing_rate', 'acuna_zone_heart_swing_rate',
        'acuna_zone_corner_swing_rate', 'acuna_zone_shadow_swing_rate',
        'acuna_zone_heart_hit_rate', 'acuna_zone_corner_hit_rate', 'acuna_zone_shadow_hit_rate'
    ]
    
    # Pitch type contact rates
    for pitch_type in ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'ST', 'KC', 'FS']:
        additional_hitter_features.append(f'acuna_{pitch_type}_hit_rate')
    
    # Pitcher features
    pitcher_features = []
    for pitch_type in ['FF', 'SI', 'SL', 'CU', 'CH', 'FC', 'ST', 'KC', 'FS']:
        pitcher_features.append(f'unknown_{pitch_type}_usage')
    for zone in range(1, 15):
        pitcher_features.append(f'unknown_zone_{zone}_usage')
    for balls in range(4):
        for strikes in range(3):
            pitcher_features.append(f'unknown_count_{balls}_{strikes}_avg_velocity')
    
    additional_pitcher_features = [
        'unknown_avg_velocity', 'unknown_avg_movement'
    ]
    
    # Combine all features
    all_features = base_features + hitter_features + additional_hitter_features + pitcher_features + additional_pitcher_features
    
    return all_features

if __name__ == "__main__":
    # Example usage
    print("🔧 FEATURE ENGINEERING MODULE")
    print("=" * 50)
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'release_speed': [95.0, 88.0, 92.0],
        'plate_x': [0.1, -0.2, 0.0],
        'plate_z': [2.5, 1.8, 2.2],
        'zone': [5, 2, 8],
        'balls': [1, 0, 2],
        'strikes': [1, 0, 1],
        'pitch_type': ['FF', 'SL', 'CH'],
        'sz_top': [3.5, 3.5, 3.5],
        'sz_bot': [1.5, 1.5, 1.5],
        'api_break_x_batter_in': [2.0, -3.0, 1.5],
        'api_break_z_with_gravity': [1.0, -2.0, 0.8],
        'release_spin_rate': [2200, 1800, 1600],
        'description': ['swinging_strike', 'hit_into_play', 'foul']
    })
    
    print("Sample data shape:", sample_data.shape)
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data, 'whiff_vs_contact')
    
    print("Engineered data shape:", engineered_df.shape)
    print("Sample engineered features:")
    print(engineered_df.columns.tolist()[:20])  # Show first 20 features 
 
 
 
 
 
 
 
 
 
 
 
 
 
 