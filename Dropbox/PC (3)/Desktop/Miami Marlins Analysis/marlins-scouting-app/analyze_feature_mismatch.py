#!/usr/bin/env python3
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
 
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
 
 
 
 
 
 
 
 
 
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
 
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
"""
Analyze feature mismatch between expected and actual features
"""

import pandas as pd
import numpy as np
import joblib
from feature_engineering import engineer_features_for_model, get_model_features

def analyze_feature_mismatch():
    """Analyze the feature mismatch issue"""
    
    print("🔍 ANALYZING FEATURE MISMATCH")
    print("=" * 50)
    
    # Get expected features from the model
    try:
        # Load the swing vs no swing model to see what features it expects
        model = joblib.load('improved_swing_classifier.joblib')
        feature_names = model.feature_names_in_
        print(f"✅ Model expects {len(feature_names)} features")
        print("First 20 expected features:")
        for i, feature in enumerate(feature_names[:20]):
            print(f"  {i+1:2d}. {feature}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create sample data and engineer features
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
        'api_break_x_arm': [1.8, -2.8, 1.3],
        'release_spin_rate': [2200, 1800, 1600],
        'spin_axis': [180, 45, 90],
        'description': ['swinging_strike', 'hit_into_play', 'foul'],
        'inning': [1, 3, 5],
        'home_score': [0, 2, 1],
        'away_score': [0, 1, 2],
        'at_bat_number': [1, 2, 3]
    })
    
    # Engineer features
    engineered_df = engineer_features_for_model(sample_data.copy(), 'swing_vs_noswing')
    actual_features = set(engineered_df.columns)
    
    print(f"\n📊 FEATURE ANALYSIS:")
    print(f"Expected features: {len(feature_names)}")
    print(f"Actual features: {len(actual_features)}")
    
    # Find missing features
    missing_features = set(feature_names) - actual_features
    extra_features = actual_features - set(feature_names)
    
    print(f"\n❌ Missing features: {len(missing_features)}")
    if missing_features:
        print("Missing features:")
        for feature in sorted(missing_features):
            print(f"  - {feature}")
    
    print(f"\n➕ Extra features: {len(extra_features)}")
    if extra_features:
        print("Extra features:")
        for feature in sorted(extra_features):
            print(f"  - {feature}")
    
    # Check if we can create the missing features
    print(f"\n🔧 ANALYZING MISSING FEATURES:")
    for feature in sorted(missing_features):
        if 'acuna_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Hitter feature, should be calculated from career data")
        elif 'unknown_' in feature and 'usage' in feature:
            print(f"  - {feature}: Pitcher feature, should be calculated from career data")
        elif 'count_' in feature and 'swing_rate' in feature:
            print(f"  - {feature}: Count feature, should be calculated from career data")
        else:
            print(f"  - {feature}: Unknown feature type")
    
    return feature_names, actual_features, missing_features

if __name__ == "__main__":
    analyze_feature_mismatch() 
 
 
 
 
 
 
 
 
 
 
 