import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def fix_zone_data(df):
    """Fix zone data by recalculating only for missing/invalid zones"""
    df = df.copy()
    
    def calculate_zone(plate_x, plate_z):
        """Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates."""
        # Strike zone boundaries (approximate)
        sz_left = -0.85
        sz_right = 0.85
        sz_bot = 1.5
        sz_top = 3.5
        
        # Check if pitch is in strike zone
        in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
        
        if in_strike_zone:
            # Calculate zone within strike zone (1-9)
            x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
            z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
            
            # Clamp to valid ranges
            x_section = max(0, min(2, x_section))
            z_section = max(0, min(2, z_section))
            
            # Convert to zone number (1-9)
            zone = z_section * 3 + x_section + 1
        else:
            # Outside strike zone (11-14)
            if plate_x < sz_left:  # Left side
                zone = 11 if plate_z > sz_top else 13
            else:  # Right side
                zone = 12 if plate_z > sz_top else 14
        
        return zone
    
    # Only calculate zones for pitches that don't have valid zone data
    if 'zone' not in df.columns or df['zone'].isna().any() or (df['zone'] <= 0).any():
        print("Calculating zones for pitches with missing/invalid zone data...")
        # Only calculate for rows that need it
        zone_mask = df['zone'].isna() | (df['zone'] <= 0) if 'zone' in df.columns else pd.Series([True] * len(df))
        df.loc[zone_mask, 'zone'] = df[zone_mask].apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    else:
        print("Using original Statcast zone data")
    
    return df

def add_basic_features(df):
    """Add basic engineered features"""
    df = df.copy()
    
    # Zone distance
    df['zone_center_x'] = 0
    df['zone_center_z'] = (df['sz_top'] + df['sz_bot']) / 2
    df['zone_distance'] = np.sqrt(
        (df['plate_x'] - df['zone_center_x'])**2 + 
        (df['plate_z'] - df['zone_center_z'])**2
    )
    
    # Movement features
    df['horizontal_break'] = df['api_break_x_batter_in'].fillna(0)
    df['vertical_break'] = df['api_break_z_with_gravity'].fillna(0)
    df['movement_magnitude'] = np.sqrt(df['horizontal_break']**2 + df['vertical_break']**2)
    
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
    
    return df

def add_comprehensive_features(df):
    """Add all comprehensive features needed for swing vs no swing model"""
    df = df.copy()
    
    # IMPROVED Movement Quantification
    df['arm_side_break'] = df['api_break_x_arm'].fillna(0)
    df['movement_direction'] = np.arctan2(df['vertical_break'], df['horizontal_break']) * 180 / np.pi
    df['movement_ratio'] = np.abs(df['horizontal_break']) / (np.abs(df['vertical_break']) + 0.1)
    
    # Spin-based movement analysis
    if 'spin_axis' in df.columns:
        df['spin_axis_rad'] = df['spin_axis'].fillna(0) * np.pi / 180
        df['spin_efficiency'] = df['release_spin_rate'] / (df['release_speed'] + 0.1)
    else:
        df['spin_axis_rad'] = 0
        df['spin_efficiency'] = 0
    
    # Velocity-movement relationship
    df['velocity_movement_ratio'] = df['release_speed'] / (df['movement_magnitude'] + 0.1)
    df['movement_per_mph'] = df['movement_magnitude'] / (df['release_speed'] + 0.1)
    
    # Pitch type specific movement expectations
    df['expected_movement'] = np.where(
        df['pitch_type'].isin(['SL', 'CU', 'KC']),  # Breaking balls
        df['movement_magnitude'] * 1.2,  # Expect more movement
        np.where(
            df['pitch_type'].isin(['CH', 'FS']),  # Offspeed
            df['movement_magnitude'] * 1.1,  # Moderate movement
            df['movement_magnitude'] * 0.8  # Fastballs - less movement expected
        )
    )
    
    # Movement deception
    df['movement_deception'] = df['movement_magnitude'] - df['expected_movement']
    
    # High movement thresholds based on pitch type
    df['high_movement'] = np.where(
        df['pitch_type'].isin(['SL', 'CU', 'KC']),
        df['movement_magnitude'] > 8,  # Breaking balls
        np.where(
            df['pitch_type'].isin(['CH', 'FS']),
            df['movement_magnitude'] > 6,  # Offspeed
            df['movement_magnitude'] > 4   # Fastballs
        )
    ).astype(int)
    
    # Movement consistency
    if 'pitcher' in df.columns:
        df['movement_std'] = df.groupby('pitcher')['movement_magnitude'].transform('std')
        df['movement_diff_from_avg'] = df['movement_magnitude'] - df.groupby('pitcher')['movement_magnitude'].transform('mean')
    else:
        df['movement_std'] = df['movement_magnitude'].std()
        df['movement_diff_from_avg'] = 0
    
    # Location x Movement interactions
    df['plate_x_norm'] = df['plate_x'] / 1.417
    df['plate_z_norm'] = (df['plate_z'] - df['sz_bot']) / (df['sz_top'] - df['sz_bot'])
    df['plate_x_norm_x_movement'] = df['plate_x_norm'] * df['movement_magnitude']
    df['plate_z_norm_x_movement'] = df['plate_z_norm'] * df['movement_magnitude']
    
    # Advanced count features
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
        df['early_count'] == 1, -0.25,
        np.where(
            df['pressure_count'] == 1, 0.15,
            0.0
        )
    )
    
    # Count-specific location penalties
    df['early_count_location_penalty'] = np.where(
        df['early_count'] == 1,
        np.where(
            df['zone_distance'] > 0.5, 0.4,
            np.where(
                df['zone_distance'] > 0.2, 0.2,
                0.0
            )
        ),
        0.0
    )
    
    # Count-specific pitch type penalties
    df['early_count_breaking_penalty'] = df['early_count'] * df['is_breaking_ball'] * 0.3
    df['early_count_offspeed_penalty'] = df['early_count'] * df['is_offspeed'] * 0.25
    
    # Advanced interaction features
    df['zone_distance_x_count_pressure'] = df['zone_distance'] * df['count_pressure']
    df['movement_x_count_pressure'] = df['movement_magnitude'] * df['count_pressure']
    df['in_zone_x_two_strikes'] = df['in_strike_zone'] * df['two_strikes']
    df['far_from_zone_x_ahead'] = df['far_from_zone'] * df['ahead_in_count']
    
    # Velocity features
    df['high_velocity'] = (df['release_speed'] > 95).astype(int)
    df['low_velocity'] = (df['release_speed'] < 85).astype(int)
    
    # Zone edge features
    df['zone_edge_distance'] = np.minimum(
        np.abs(df['plate_x'] - (-0.85)),
        np.abs(df['plate_x'] - 0.85)
    )
    df['zone_top_distance'] = np.abs(df['plate_z'] - df['sz_top'])
    df['zone_bottom_distance'] = np.abs(df['plate_z'] - df['sz_bot'])
    df['closest_zone_edge'] = np.minimum(
        np.minimum(df['zone_edge_distance'], df['zone_top_distance']),
        df['zone_bottom_distance']
    )
    
    # Advanced count features
    df['full_count'] = (df['balls'] == 3) & (df['strikes'] == 2)
    df['hitters_count'] = (df['balls'] >= 2) & (df['strikes'] <= 1)
    df['pitchers_count'] = (df['balls'] <= 1) & (df['strikes'] >= 2)
    df['neutral_count'] = (df['balls'] == 1) & (df['strikes'] == 1)
    
    # Zone quadrant features
    df['zone_quadrant'] = np.where(
        df['plate_x'] >= 0,
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_out', 'down_out'),
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_in', 'down_in')
    )
    
    # Pitch deception features
    df['velocity_drop'] = (df['release_speed'] < 90) & df['is_fastball']
    df['breaking_ball_high'] = df['is_breaking_ball'] & (df['plate_z'] > df['sz_top'])
    df['offspeed_low'] = df['is_offspeed'] & (df['plate_z'] < df['sz_bot'])
    
    # Context features
    df['inning_late'] = (df['inning'] >= 7).astype(int) if 'inning' in df.columns else 0
    df['close_game'] = (np.abs(df['home_score'] - df['away_score']) <= 2).astype(int) if 'home_score' in df.columns and 'away_score' in df.columns else 0
    
    # Pitch sequencing features
    if 'at_bat_number' in df.columns:
        df['pitch_in_at_bat'] = df.groupby('at_bat_number').cumcount() + 1
        df['first_pitch'] = (df['pitch_in_at_bat'] == 1).astype(int)
        df['last_pitch'] = df.groupby('at_bat_number')['pitch_in_at_bat'].transform('max') == df['pitch_in_at_bat']
    else:
        df['pitch_in_at_bat'] = 1
        df['first_pitch'] = 1
        df['last_pitch'] = 1
    
    # Velocity consistency
    if 'pitcher' in df.columns:
        df['velocity_std'] = df.groupby('pitcher')['release_speed'].transform('std')
        df['velocity_diff_from_avg'] = df['release_speed'] - df.groupby('pitcher')['release_speed'].transform('mean')
    else:
        df['velocity_std'] = df['release_speed'].std()
        df['velocity_diff_from_avg'] = 0
    
    # Movement deception features
    df['horizontal_movement'] = df['horizontal_break']
    df['vertical_movement'] = df['vertical_break']
    df['high_horizontal_movement'] = (np.abs(df['horizontal_break']) > 5).astype(int)
    df['high_vertical_movement'] = (np.abs(df['vertical_break']) > 5).astype(int)
    
    # Zone precision features
    df['zone_center_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - (df['sz_top'] + df['sz_bot'])/2)**2)
    df['zone_corner'] = ((df['plate_x'] >= 0.7) | (df['plate_x'] <= -0.7)) & ((df['plate_z'] >= df['sz_top'] - 0.2) | (df['plate_z'] <= df['sz_bot'] + 0.2))
    df['zone_heart'] = (df['zone_center_distance'] < 0.5).astype(int)
    df['zone_shadow'] = ((df['plate_x'] >= -1.0) & (df['plate_x'] <= 1.0) & (df['plate_z'] >= df['sz_bot'] - 0.2) & (df['plate_z'] <= df['sz_top'] + 0.2)) & ~df['in_strike_zone']
    
    # Advanced count psychology features
    df['count_advantage'] = np.where(df['count_pressure'] > 0, 'hitter_ahead', np.where(df['count_pressure'] < 0, 'pitcher_ahead', 'neutral'))
    df['pressure_situation'] = (df['two_strikes'] | df['three_balls']).astype(int)
    df['must_swing'] = df['two_strikes'].astype(int)
    df['can_take'] = (df['ahead_in_count'] & ~df['in_strike_zone']).astype(int)
    
    # Pitch type deception features
    df['fastball_high'] = df['is_fastball'] & (df['plate_z'] > df['sz_top'])
    df['breaking_ball_low'] = df['is_breaking_ball'] & (df['plate_z'] < df['sz_bot'])
    df['offspeed_middle'] = df['is_offspeed'] & df['in_strike_zone']
    
    # Pitch type change
    if 'at_bat_number' in df.columns:
        df['pitch_type_change'] = df.groupby('at_bat_number')['pitch_type'].shift(1) != df['pitch_type']
    else:
        df['pitch_type_change'] = 0
    
    # Advanced location features
    df['location_quadrant'] = np.where(
        df['plate_x'] >= 0,
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_right', 'down_right'),
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_left', 'down_left')
    )
    df['location_extreme'] = (np.abs(df['plate_x']) > 1.0) | (df['plate_z'] > df['sz_top'] + 0.5) | (df['plate_z'] < df['sz_bot'] - 0.5)
    
    # Spin and movement correlation features
    if 'release_spin_rate' in df.columns:
        df['spin_movement_correlation'] = df['release_spin_rate'] * df['movement_magnitude']
        df['high_spin'] = (df['release_spin_rate'] > 2500).astype(int)
        df['low_spin'] = (df['release_spin_rate'] < 2000).astype(int)
    else:
        df['spin_movement_correlation'] = 0
        df['high_spin'] = 0
        df['low_spin'] = 0
    
    # Game situation features
    df['late_inning'] = (df['inning'] >= 8).astype(int) if 'inning' in df.columns else 0
    df['close_score'] = (np.abs(df['home_score'] - df['away_score']) <= 1).astype(int) if 'home_score' in df.columns and 'away_score' in df.columns else 0
    df['high_leverage'] = (df['late_inning'] & df['close_score']).astype(int)
    
    # Advanced count-based features
    df['count_ratio'] = df['strikes'] / (df['balls'] + df['strikes'] + 0.1)
    df['behind_by_two'] = (df['strikes'] - df['balls'] >= 2).astype(int)
    df['ahead_by_two'] = (df['balls'] - df['strikes'] >= 2).astype(int)
    df['full_count_pressure'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
    
    # Zone-specific count features
    df['in_zone_two_strikes'] = df['in_strike_zone'] & df['two_strikes']
    df['out_zone_ahead'] = ~df['in_strike_zone'] & df['ahead_in_count']
    df['edge_zone_decision'] = ((df['zone_distance'] > 0.5) & (df['zone_distance'] < 1.0)).astype(int)
    
    # Velocity deception features
    df['velocity_surprise'] = (df['release_speed'] < 90) & df['is_fastball']
    df['velocity_consistency'] = (df['release_speed'] > 95) & df['is_fastball']
    df['breaking_ball_velocity'] = df['is_breaking_ball'] & (df['release_speed'] > 85)
    
    # Movement deception features
    df['high_movement_fastball'] = df['is_fastball'] & (df['movement_magnitude'] > 8)
    df['low_movement_breaking'] = df['is_breaking_ball'] & (df['movement_magnitude'] < 5)
    df['unexpected_movement'] = ((df['movement_magnitude'] > 12) | (df['movement_magnitude'] < 3)).astype(int)
    
    # Location deception features
    df['corner_pitch'] = df['zone_corner'].astype(int)
    df['heart_pitch'] = df['zone_heart'].astype(int)
    df['shadow_pitch'] = df['zone_shadow'].astype(int)
    df['extreme_location'] = (np.abs(df['plate_x']) > 1.2) | (df['plate_z'] > df['sz_top'] + 0.8) | (df['plate_z'] < df['sz_bot'] - 0.8)
    
    # Advanced interaction features
    df['velocity_x_location'] = df['release_speed'] * df['zone_distance']
    df['pitch_type_x_location'] = df['is_fastball'] * df['in_strike_zone']
    df['count_x_zone'] = df['count_pressure'] * df['in_strike_zone']
    
    # Context-based features
    df['early_count_swing'] = (df['count_total'] <= 2) & df['in_strike_zone']
    df['late_count_take'] = (df['count_total'] >= 4) & ~df['in_strike_zone']
    df['pressure_swing'] = df['two_strikes'] & df['in_strike_zone']
    df['opportunity_take'] = df['ahead_in_count'] & ~df['in_strike_zone']
    
    # Count-specific velocity penalties
    df['early_count_low_vel_penalty'] = df['early_count'] * df['low_velocity'] * 0.35
    df['early_count_high_vel_penalty'] = df['early_count'] * df['high_velocity'] * 0.15
    
    # Categorical encodings
    df['zone_quadrant_encoded'] = pd.Categorical(df['zone_quadrant']).codes
    df['location_quadrant_encoded'] = pd.Categorical(df['location_quadrant']).codes
    df['count_advantage_encoded'] = pd.Categorical(df['count_advantage']).codes
    
    # Default values for missing features
    df['count_field_out_rate'] = 0.0
    df['count_hit_rate'] = 0.0
    df['count_whiff_rate'] = 0.0
    df['pitch_type_field_out_rate'] = 0.0
    df['pitch_type_hit_rate'] = 0.0
    df['pitch_type_whiff_rate'] = 0.0
    
    return df

def add_hitter_features(df):
    """Add hitter-specific features (Acuna features)"""
    df = df.copy()
    
    # Default Acuna features
    acuna_features = {
        'acuna_fastball_swing_rate': 0.0,
        'acuna_breaking_swing_rate': 0.0,
        'acuna_offspeed_swing_rate': 0.0,
        'acuna_zone_swing_rate': 0.0,
        'acuna_outside_swing_rate': 0.0,
        'acuna_high_swing_rate': 0.0,
        'acuna_low_swing_rate': 0.0,
        'acuna_ahead_swing_rate': 0.0,
        'acuna_behind_swing_rate': 0.0,
        'acuna_two_strikes_swing_rate': 0.0,
        'acuna_full_count_swing_rate': 0.0,
        'acuna_high_vel_swing_rate': 0.0,
        'acuna_low_vel_swing_rate': 0.0,
        'acuna_high_movement_swing_rate': 0.0,
        'acuna_low_movement_swing_rate': 0.0,
        'acuna_late_inning_swing_rate': 0.0,
        'acuna_close_game_swing_rate': 0.0,
        'acuna_first_pitch_swing_rate': 0.0,
        'acuna_last_pitch_swing_rate': 0.0,
        'acuna_pitch_type_change_swing_rate': 0.0,
        'acuna_velocity_drop_swing_rate': 0.0,
        'acuna_velocity_surge_swing_rate': 0.0,
        'acuna_location_extreme_swing_rate': 0.0,
        'acuna_location_heart_swing_rate': 0.0,
        'acuna_pressure_swing_rate': 0.0,
        'acuna_opportunity_swing_rate': 0.0,
        'acuna_zone_corner_swing_rate': 0.0,
        'acuna_zone_shadow_swing_rate': 0.0,
        'acuna_zone_heart_swing_rate': 0.0,
        # Zone-specific contact rate features
        'acuna_zone_heart_hit_rate': 0.0,
        'acuna_zone_corner_hit_rate': 0.0,
        'acuna_zone_shadow_hit_rate': 0.0,
        'acuna_zone_hit_rate': 0.0,
        'acuna_outside_hit_rate': 0.0
    }
    
    # Add comprehensive count features
    count_features = {
        'count_0_0_swing_rate': 0.0,
        'count_1_0_swing_rate': 0.0,
        'count_0_1_swing_rate': 0.0,
        'count_2_0_swing_rate': 0.0,
        'count_1_1_swing_rate': 0.0,
        'count_0_2_swing_rate': 0.0,
        'count_3_0_swing_rate': 0.0,
        'count_2_1_swing_rate': 0.0,
        'count_1_2_swing_rate': 0.0,
        'count_3_1_swing_rate': 0.0,
        'count_2_2_swing_rate': 0.0,
        'count_3_2_swing_rate': 0.0
    }
    
    for feature_name, feature_value in count_features.items():
        acuna_features[f'acuna_{feature_name}'] = feature_value
    
    # Add all features to dataframe
    acuna_features_df = pd.DataFrame([acuna_features] * len(df), index=df.index)
    df = pd.concat([df, acuna_features_df], axis=1)
    
    return df

def add_babip_features(df):
    """Add BABIP features"""
    df = df.copy()
    
    # Default BABIP features
    df['batting_average_bip'] = 0.25
    df['whiff_rate'] = 0.35
    df['field_out_rate_bip'] = 0.40
    df['balls_in_play'] = 0
    df['total_swings'] = 0
    df['total_whiffs'] = 0
    
    return df

def add_contact_features(df):
    """Add proxy contact features"""
    df = df.copy()
    
    # Proxy contact features
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
    
    return df

def fix_categorical_features(df):
    """Fix categorical features for XGBoost compatibility"""
    df = df.copy()
    
    # Convert categorical columns to numeric
    categorical_columns = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 
                         'stand', 'home_team', 'zone_quadrant', 'location_quadrant', 'count_advantage']
    
    for col in categorical_columns:
        if col in df.columns:
            df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
            # Drop original categorical column to avoid XGBoost errors
            df = df.drop(columns=[col])
    
    return df

def analyze_missing_features(df):
    """Analyze which features are missing"""
    print("Analyzing missing features...")
    
    # Expected features from training script
    expected_features = [
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
        'early_count', 'middle_count', 'late_count', 'pressure_count',
        'early_count_penalty', 'early_count_zone_penalty', 'early_count_outside_penalty',
        'count_swing_rate_adjustment', 'early_count_location_penalty',
        'early_count_breaking_penalty', 'early_count_offspeed_penalty',
        'early_count_low_vel_penalty', 'early_count_high_vel_penalty',
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
        'acuna_zone_corner_swing_rate', 'acuna_zone_shadow_swing_rate', 'acuna_zone_heart_swing_rate',
        'acuna_zone_heart_hit_rate', 'acuna_zone_corner_hit_rate', 'acuna_zone_shadow_hit_rate',
        'acuna_zone_hit_rate', 'acuna_outside_hit_rate',
        'zone_heart_contact', 'zone_corner_contact', 'zone_shadow_contact',
        'zone_overall_contact', 'outside_zone_contact',
        'fastball_zone_contact', 'breaking_zone_contact', 'offspeed_zone_contact',
        'pressure_zone_contact', 'opportunity_zone_contact', 'two_strikes_zone_contact',
        'batting_average_bip', 'whiff_rate', 'field_out_rate_bip', 'balls_in_play', 'total_swings', 'total_whiffs'
    ]
    
    missing_features = []
    for feature in expected_features:
        if feature not in df.columns:
            missing_features.append(feature)
    
    print(f"Missing {len(missing_features)} features:")
    for feature in missing_features:
        print(f"  - {feature}")
    
    return missing_features

def fix_feature_alignment(df):
    """Fix feature alignment by adding all missing features"""
    print("Fixing feature alignment...")
    
    # Step 1: Fix zone data
    df = fix_zone_data(df)
    
    # Step 2: Add basic features
    df = add_basic_features(df)
    
    # Step 3: Add comprehensive features
    df = add_comprehensive_features(df)
    
    # Step 4: Add hitter features
    df = add_hitter_features(df)
    
    # Step 5: Add BABIP features
    df = add_babip_features(df)
    
    # Step 6: Add contact features
    df = add_contact_features(df)
    
    # Step 7: Fix categorical features
    df = fix_categorical_features(df)
    
    # Step 8: Analyze missing features
    missing_features = analyze_missing_features(df)
    
    print(f"✓ Feature alignment complete. DataFrame has {len(df.columns)} columns")
    
    return df

if __name__ == "__main__":
    # Test the feature alignment
    print("Testing feature alignment...")
    
    # Create a sample DataFrame with basic Statcast columns
    sample_data = {
        'pitch_type': ['FF', 'SL', 'CH'],
        'plate_x': [0.1, -0.5, 0.8],
        'plate_z': [2.5, 3.2, 1.8],
        'release_speed': [95, 88, 85],
        'balls': [1, 2, 0],
        'strikes': [1, 1, 2],
        'sz_top': [3.5, 3.5, 3.5],
        'sz_bot': [1.5, 1.5, 1.5],
        'api_break_x_batter_in': [2.1, -3.2, 1.5],
        'api_break_z_with_gravity': [-1.8, 2.4, -0.9],
        'api_break_x_arm': [2.1, -3.2, 1.5],
        'release_spin_rate': [2200, 2800, 1800],
        'p_throws': ['R', 'R', 'R'],
        'stand': ['R', 'R', 'R'],
        'home_team': ['ATL', 'ATL', 'ATL'],
        'inning': [1, 1, 1],
        'home_score': [0, 0, 0],
        'away_score': [0, 0, 0]
    }
    
    df = pd.DataFrame(sample_data)
    df = fix_feature_alignment(df)
    
    print(f"✓ Test complete. Final DataFrame has {len(df.columns)} columns") 
 
 
 
 
 
 
 
 
 
 
 