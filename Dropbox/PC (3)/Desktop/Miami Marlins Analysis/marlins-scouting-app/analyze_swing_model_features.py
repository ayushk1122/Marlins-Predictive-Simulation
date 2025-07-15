import pickle
import pandas as pd
import numpy as np
import os

def analyze_swing_model_features():
    """Analyze the swing vs no swing model to get complete feature list"""
    
    # Load the sequential models
    models_path = "sequential_models.pkl"
    
    if not os.path.exists(models_path):
        print(f"Model file not found: {models_path}")
        return
    
    try:
        # Load the sequential models
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        
        print("=== Swing vs No Swing Model Analysis ===")
        print(f"Model file: {models_path}")
        print()
        
        # Find the swing vs no swing model
        swing_model = None
        for key, model in models.items():
            if 'swing' in key.lower() or 'noswing' in key.lower():
                swing_model = model
                print(f"Found swing model: {key}")
                break
        
        if swing_model is None:
            print("Could not find swing vs no swing model in sequential_models.pkl")
            print("Available models:")
            for key in models.keys():
                print(f"  - {key}")
            return None
        
        # Print model info
        print(f"Model type: {type(swing_model)}")
        print(f"Model attributes: {[attr for attr in dir(swing_model) if not attr.startswith('_')]}")
        print()
        
        # Always get actual feature names from feature engineering module
        print("Getting actual feature names from feature engineering module...")
        
        try:
            from feature_engineering import get_model_features
            
            # Get feature names directly from the function
            feature_names = get_model_features(model_type='swing_vs_noswing')
            print(f"Extracted {len(feature_names)} features from feature engineering module")
            print("\n=== Actual Feature Names from Feature Engineering Module ===")
            for i, feature in enumerate(feature_names, 1):
                print(f"{i:3d}. {feature}")
            print()
            
        except Exception as e:
            print(f"Could not extract features from feature engineering: {e}")
            print("Falling back to generic feature names...")
            
            # Get the number of features the model expects
            if hasattr(swing_model, 'n_features_in_'):
                n_features = swing_model.n_features_in_
            elif hasattr(swing_model, 'feature_importances_'):
                n_features = len(swing_model.feature_importances_)
            elif hasattr(swing_model, 'coef_'):
                n_features = len(swing_model.coef_)
            else:
                print("Could not determine number of features from model")
                return None
            
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if feature_names:
            print(f"Total features required: {len(feature_names)}")
            print()
            print("=== Complete Feature List ===")
            for i, feature in enumerate(feature_names, 1):
                print(f"{i:3d}. {feature}")
            print()
            
            # Group features by category
            feature_categories = {
                'Basic Statcast': [],
                'Pitch Movement': [],
                'Count-based': [],
                'Zone/Location': [],
                'Hitter Averages': [],
                'Pitcher Averages': [],
                'Advanced Features': [],
                'Categorical': []
            }
            
            for feature in feature_names:
                if any(x in feature.lower() for x in ['plate_x', 'plate_z', 'zone', 'pitch_type', 'release_speed']):
                    feature_categories['Basic Statcast'].append(feature)
                elif any(x in feature.lower() for x in ['movement', 'break', 'spin']):
                    feature_categories['Pitch Movement'].append(feature)
                elif any(x in feature.lower() for x in ['balls', 'strikes', 'count']):
                    feature_categories['Count-based'].append(feature)
                elif any(x in feature.lower() for x in ['zone', 'location', 'plate']):
                    feature_categories['Zone/Location'].append(feature)
                elif 'acuna' in feature.lower():
                    feature_categories['Hitter Averages'].append(feature)
                elif any(x in feature.lower() for x in ['pitcher', 'sandy']):
                    feature_categories['Pitcher Averages'].append(feature)
                elif any(x in feature.lower() for x in ['babip', 'contact', 'advanced']):
                    feature_categories['Advanced Features'].append(feature)
                elif any(x in feature.lower() for x in ['_cat', 'encoded', 'dummy']):
                    feature_categories['Categorical'].append(feature)
                else:
                    feature_categories['Advanced Features'].append(feature)
            
            print("=== Features by Category ===")
            for category, features in feature_categories.items():
                if features:
                    print(f"\n{category} ({len(features)} features):")
                    for feature in sorted(features):
                        print(f"  - {feature}")
            
            # Save feature list to file
            with open('swing_model_features.txt', 'w') as f:
                f.write("=== Swing vs No Swing Model Features ===\n\n")
                f.write(f"Total features: {len(feature_names)}\n\n")
                f.write("Complete feature list:\n")
                for i, feature in enumerate(feature_names, 1):
                    f.write(f"{i:3d}. {feature}\n")
                f.write("\n")
                f.write("Features by category:\n")
                for category, features in feature_categories.items():
                    if features:
                        f.write(f"\n{category} ({len(features)} features):\n")
                        for feature in sorted(features):
                            f.write(f"  - {feature}\n")
            
            print(f"\nFeature list saved to: swing_model_features.txt")
            
            return feature_names
            
        else:
            print("Could not extract feature names from model or feature engineering")
            return None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def check_feature_engineering_alignment():
    """Check if our feature engineering module has all required features"""
    
    # Get required features
    required_features = analyze_swing_model_features()
    if not required_features:
        return
    
    # Import our feature engineering module
    try:
        from feature_engineering import get_model_features
        
        # Get required features directly
        required_features = get_model_features(model_type='swing_vs_noswing')
        
        # Check which required features are missing
        missing_features = []
        for feature in required_features:
            if feature not in features_df.columns:
                missing_features.append(feature)
        
        print(f"\n=== Feature Engineering Alignment Check ===")
        print(f"Features generated: {len(features_df.columns)}")
        print(f"Features required: {len(required_features)}")
        print(f"Missing features: {len(missing_features)}")
        
        if missing_features:
            print("\nMissing features:")
            for feature in missing_features:
                print(f"  - {feature}")
        else:
            print("\n✅ All required features are available!")
            
    except ImportError:
        print("Could not import feature_engineering module")
    except Exception as e:
        print(f"Error checking feature engineering: {e}")

if __name__ == "__main__":
    analyze_swing_model_features()
    check_feature_engineering_alignment() 
 
 
 
 
 
 
 
 
 
 
 
 
 