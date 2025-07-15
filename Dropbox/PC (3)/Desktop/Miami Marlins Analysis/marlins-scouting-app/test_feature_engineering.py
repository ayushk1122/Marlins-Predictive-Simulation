#!/usr/bin/env python3
"""
Test script to verify feature engineering works for both model types
"""

import pandas as pd
import numpy as np
from feature_engineering import engineer_features_for_model, get_model_features

def test_feature_engineering():
    """Test feature engineering for both model types"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'release_speed': [95.0, 88.0, 92.0, 87.0, 94.0],
        'plate_x': [0.1, -0.2, 0.0, 0.5, -0.3],
        'plate_z': [2.5, 1.8, 2.2, 3.0, 1.5],
        'zone': [5, 2, 8, 1, 9],
        'balls': [1, 0, 2, 1, 0],
        'strikes': [1, 0, 1, 2, 1],
        'pitch_type': ['FF', 'SL', 'CH', 'CU', 'SI'],
        'sz_top': [3.5, 3.5, 3.5, 3.5, 3.5],
        'sz_bot': [1.5, 1.5, 1.5, 1.5, 1.5],
        'api_break_x_batter_in': [2.0, -3.0, 1.5, -2.5, 1.8],
        'api_break_z_with_gravity': [1.0, -2.0, 0.8, -1.5, 1.2],
        'api_break_x_arm': [1.8, -2.8, 1.3, -2.3, 1.6],
        'release_spin_rate': [2200, 1800, 1600, 2000, 2400],
        'spin_axis': [180, 45, 90, 135, 225],
        'description': ['swinging_strike', 'hit_into_play', 'foul', 'ball', 'called_strike'],
        'inning': [1, 3, 5, 7, 9],
        'home_score': [0, 2, 1, 3, 2],
        'away_score': [0, 1, 2, 2, 3],
        'at_bat_number': [1, 2, 3, 4, 5]
    })
    
    print("🧪 TESTING FEATURE ENGINEERING")
    print("=" * 50)
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test whiff vs contact model
    print("\n🔍 Testing whiff vs contact model...")
    whiff_df = engineer_features_for_model(sample_data.copy(), 'whiff_vs_contact')
    print(f"Whiff model features: {whiff_df.shape[1]} columns")
    
    # Test swing vs no swing model
    print("\n🔍 Testing swing vs no swing model...")
    swing_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    print(f"Swing model features: {swing_df.shape[1]} columns")
    
    # Compare feature counts
    whiff_features = set(whiff_df.columns)
    swing_features = set(swing_df.columns)
    
    swing_only = swing_features - whiff_features
    whiff_only = whiff_features - swing_features
    common_features = whiff_features & swing_features
    
    print(f"\n📊 FEATURE COMPARISON:")
    print(f"Common features: {len(common_features)}")
    print(f"Swing-only features: {len(swing_only)}")
    print(f"Whiff-only features: {len(whiff_only)}")
    
    if swing_only:
        print(f"\n🎯 Swing model additional features:")
        for feature in sorted(swing_only):
            print(f"  - {feature}")
    
    # Test feature list generation
    print(f"\n📋 FEATURE LISTS:")
    whiff_feature_list = get_model_features('whiff_vs_contact')
    swing_feature_list = get_model_features('swing_vs_noswing')
    
    print(f"Whiff model expected features: {len(whiff_feature_list)}")
    print(f"Swing model expected features: {len(swing_feature_list)}")
    
    # Check if all expected features are present
    missing_whiff = set(whiff_feature_list) - set(whiff_df.columns)
    missing_swing = set(swing_feature_list) - set(swing_df.columns)
    
    if missing_whiff:
        print(f"\n⚠️ Missing whiff features: {len(missing_whiff)}")
        for feature in sorted(missing_whiff):
            print(f"  - {feature}")
    
    if missing_swing:
        print(f"\n⚠️ Missing swing features: {len(missing_swing)}")
        for feature in sorted(missing_swing):
            print(f"  - {feature}")
    
    if not missing_whiff and not missing_swing:
        print("\n✅ All expected features are present!")
    
    # Test specific swing features
    swing_specific_features = [
        'arm_side_break', 'movement_direction', 'early_count', 'pressure_count',
        'early_count_penalty', 'count_swing_rate_adjustment', 'zone_quadrant',
        'velocity_drop', 'breaking_ball_high', 'offspeed_low'
    ]
    
    print(f"\n🎯 Testing swing-specific features:")
    for feature in swing_specific_features:
        if feature in swing_df.columns:
            print(f"  ✅ {feature}: {swing_df[feature].dtype}")
        else:
            print(f"  ❌ {feature}: MISSING")
    
    print(f"\n✅ Feature engineering test complete!")
    return whiff_df, swing_df

if __name__ == "__main__":
    test_feature_engineering() 
"""
Test script to verify feature engineering works for both model types
"""

import pandas as pd
import numpy as np
from feature_engineering import engineer_features_for_model, get_model_features

def test_feature_engineering():
    """Test feature engineering for both model types"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'release_speed': [95.0, 88.0, 92.0, 87.0, 94.0],
        'plate_x': [0.1, -0.2, 0.0, 0.5, -0.3],
        'plate_z': [2.5, 1.8, 2.2, 3.0, 1.5],
        'zone': [5, 2, 8, 1, 9],
        'balls': [1, 0, 2, 1, 0],
        'strikes': [1, 0, 1, 2, 1],
        'pitch_type': ['FF', 'SL', 'CH', 'CU', 'SI'],
        'sz_top': [3.5, 3.5, 3.5, 3.5, 3.5],
        'sz_bot': [1.5, 1.5, 1.5, 1.5, 1.5],
        'api_break_x_batter_in': [2.0, -3.0, 1.5, -2.5, 1.8],
        'api_break_z_with_gravity': [1.0, -2.0, 0.8, -1.5, 1.2],
        'api_break_x_arm': [1.8, -2.8, 1.3, -2.3, 1.6],
        'release_spin_rate': [2200, 1800, 1600, 2000, 2400],
        'spin_axis': [180, 45, 90, 135, 225],
        'description': ['swinging_strike', 'hit_into_play', 'foul', 'ball', 'called_strike'],
        'inning': [1, 3, 5, 7, 9],
        'home_score': [0, 2, 1, 3, 2],
        'away_score': [0, 1, 2, 2, 3],
        'at_bat_number': [1, 2, 3, 4, 5]
    })
    
    print("🧪 TESTING FEATURE ENGINEERING")
    print("=" * 50)
    print(f"Sample data shape: {sample_data.shape}")
    
    # Test whiff vs contact model
    print("\n🔍 Testing whiff vs contact model...")
    whiff_df = engineer_features_for_model(sample_data.copy(), 'whiff_vs_contact')
    print(f"Whiff model features: {whiff_df.shape[1]} columns")
    
    # Test swing vs no swing model
    print("\n🔍 Testing swing vs no swing model...")
    swing_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    print(f"Swing model features: {swing_df.shape[1]} columns")
    
    # Compare feature counts
    whiff_features = set(whiff_df.columns)
    swing_features = set(swing_df.columns)
    
    swing_only = swing_features - whiff_features
    whiff_only = whiff_features - swing_features
    common_features = whiff_features & swing_features
    
    print(f"\n📊 FEATURE COMPARISON:")
    print(f"Common features: {len(common_features)}")
    print(f"Swing-only features: {len(swing_only)}")
    print(f"Whiff-only features: {len(whiff_only)}")
    
    if swing_only:
        print(f"\n🎯 Swing model additional features:")
        for feature in sorted(swing_only):
            print(f"  - {feature}")
    
    # Test feature list generation
    print(f"\n📋 FEATURE LISTS:")
    whiff_feature_list = get_model_features('whiff_vs_contact')
    swing_feature_list = get_model_features('swing_vs_noswing')
    
    print(f"Whiff model expected features: {len(whiff_feature_list)}")
    print(f"Swing model expected features: {len(swing_feature_list)}")
    
    # Check if all expected features are present
    missing_whiff = set(whiff_feature_list) - set(whiff_df.columns)
    missing_swing = set(swing_feature_list) - set(swing_df.columns)
    
    if missing_whiff:
        print(f"\n⚠️ Missing whiff features: {len(missing_whiff)}")
        for feature in sorted(missing_whiff):
            print(f"  - {feature}")
    
    if missing_swing:
        print(f"\n⚠️ Missing swing features: {len(missing_swing)}")
        for feature in sorted(missing_swing):
            print(f"  - {feature}")
    
    if not missing_whiff and not missing_swing:
        print("\n✅ All expected features are present!")
    
    # Test specific swing features
    swing_specific_features = [
        'arm_side_break', 'movement_direction', 'early_count', 'pressure_count',
        'early_count_penalty', 'count_swing_rate_adjustment', 'zone_quadrant',
        'velocity_drop', 'breaking_ball_high', 'offspeed_low'
    ]
    
    print(f"\n🎯 Testing swing-specific features:")
    for feature in swing_specific_features:
        if feature in swing_df.columns:
            print(f"  ✅ {feature}: {swing_df[feature].dtype}")
        else:
            print(f"  ❌ {feature}: MISSING")
    
    print(f"\n✅ Feature engineering test complete!")
    return whiff_df, swing_df

if __name__ == "__main__":
    test_feature_engineering() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 