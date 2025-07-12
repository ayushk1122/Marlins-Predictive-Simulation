import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_mismatch():
    """Analyze the feature mismatch between training and testing"""
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Load the trained model to see what features it expects
    try:
        with open('sequential_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        outcome_preprocessor = models.get('swing_outcome_preprocessor')
        if outcome_preprocessor is None:
            print("✗ Outcome preprocessor not found")
            return
        
        expected_features = outcome_preprocessor.feature_names_in_
        print(f"✅ Model expects {len(expected_features)} features")
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load test data to see what features we have
    try:
        df = pd.read_csv('ronald_acuna_jr_holdout_statcast.csv')
        print(f"✅ Loaded test data with {len(df)} pitches")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return
    
    # Prepare test features (simplified version)
    test_df = prepare_test_features(df)
    
    # Compare features
    available_features = set(test_df.columns)
    expected_features_set = set(expected_features)
    
    missing_features = expected_features_set - available_features
    extra_features = available_features - expected_features_set
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(expected_features_set)}")
    print(f"Available features: {len(available_features)}")
    print(f"Missing features: {len(missing_features)}")
    print(f"Extra features: {len(extra_features)}")
    
    if missing_features:
        print(f"\n❌ MISSING FEATURES ({len(missing_features)}):")
        for i, feature in enumerate(sorted(missing_features), 1):
            print(f"  {i:2d}. {feature}")
    
    if extra_features:
        print(f"\n➕ EXTRA FEATURES ({len(extra_features)}):")
        for i, feature in enumerate(sorted(extra_features), 1):
            print(f"  {i:2d}. {feature}")
    
    # Analyze feature categories
    analyze_feature_categories(expected_features, missing_features)
    
    return expected_features, missing_features

def analyze_feature_categories(expected_features, missing_features):
    """Analyze what types of features are missing"""
    print(f"\n📈 FEATURE CATEGORY ANALYSIS:")
    
    # Categorize features
    categories = {
        'basic_pitch': ['release_speed', 'release_spin_rate', 'plate_x', 'plate_z', 'zone'],
        'count': ['balls', 'strikes', 'count_total', 'two_strikes', 'three_balls'],
        'pitch_type': ['is_fastball', 'is_breaking_ball', 'is_offspeed'],
        'location': ['in_strike_zone', 'far_from_zone', 'high_pitch', 'low_pitch'],
        'velocity': ['high_velocity', 'low_velocity', 'velocity_diff_from_avg'],
        'movement': ['movement_magnitude', 'high_movement', 'velocity_movement_ratio'],
        'interaction': ['zone_distance_x_count_pressure', 'movement_x_count_pressure'],
        'acuna_specific': [f for f in expected_features if f.startswith('acuna_')],
        'count_specific': [f for f in expected_features if 'count' in f and 'rate' in f],
        'zone_specific': [f for f in expected_features if 'zone' in f],
        'pressure': [f for f in expected_features if 'pressure' in f],
        'situational': [f for f in expected_features if any(x in f for x in ['inning', 'game', 'leverage'])],
        'quadrant': [f for f in expected_features if 'quadrant' in f],
        'distance': [f for f in expected_features if 'distance' in f],
        'ratio': [f for f in expected_features if 'ratio' in f],
        'encoded': [f for f in expected_features if 'encoded' in f]
    }
    
    for category, features in categories.items():
        category_missing = [f for f in features if f in missing_features]
        if category_missing:
            print(f"  {category.upper()}: {len(category_missing)} missing")
            for feature in category_missing[:5]:  # Show first 5
                print(f"    - {feature}")
            if len(category_missing) > 5:
                print(f"    ... and {len(category_missing) - 5} more")

def prepare_test_features(df):
    """Prepare test features to match training"""
    print("\n🔧 PREPARING TEST FEATURES")
    
    # Filter to hitting events
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
    
    test_df = df[df['events'].isin(hitting_events)].copy()
    
    # Fix zones
    test_df['zone'] = test_df.apply(fix_zone, axis=1)
    test_df = test_df[test_df['zone'] > 0]
    
    # Add basic features
    test_df = add_basic_features(test_df)
    
    # Add comprehensive features
    test_df = add_comprehensive_features(test_df)
    
    return test_df

def fix_zone(row):
    """Fix invalid zones"""
    if row['zone'] <= 0:
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

def add_basic_features(df):
    """Add basic features"""
    print("  Adding basic features...")
    
    # Movement calculation
    if 'horizontal_break' in df.columns and 'vertical_break' in df.columns:
        df['movement_magnitude'] = np.sqrt(df['horizontal_break']**2 + df['vertical_break']**2)
    elif 'pfx_x' in df.columns and 'pfx_z' in df.columns:
        df['movement_magnitude'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    else:
        df['movement_magnitude'] = df['release_spin_rate'] / 1000
    
    # Count features
    df['count_total'] = df['balls'] + df['strikes']
    df['count_pressure'] = df['strikes'] / 3.0
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
    df['zone_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
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
    
    return df

def add_comprehensive_features(df):
    """Add comprehensive features"""
    print("  Adding comprehensive features...")
    
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
    df['expected_movement'] = df['release_speed'] * 0.1
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
    
    # Add Acuna-specific features (placeholder values for now)
    acuna_features = [
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
    
    for feature in acuna_features:
        df[feature] = 0.0  # Default values
    
    return df

def main():
    """Main function to analyze and fix feature alignment"""
    print("🔧 FEATURE ALIGNMENT ANALYSIS & FIX")
    print("=" * 50)
    
    # Analyze feature mismatch
    expected_features, missing_features = analyze_feature_mismatch()
    
    if missing_features:
        print(f"\n✅ SOLUTION: Add missing features to test scripts")
        print("The following features need to be calculated in test scripts:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. Update test scripts to calculate all missing features")
    print("2. Ensure feature calculation order matches training")
    print("3. Add proper default values for missing features")
    print("4. Test with aligned features")

if __name__ == "__main__":
    main() 