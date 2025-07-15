import joblib
import pandas as pd

def extract_swing_model_features():
    """Extract all features expected by the swing vs no swing model"""
    try:
        # Load the model
        models = joblib.load('sequential_models.pkl')
        
        # Get the swing features
        swing_features = models.get('swing_features', [])
        
        print("SWING VS NO SWING MODEL FEATURES:")
        print("=" * 50)
        print(f"Total features expected: {len(swing_features)}")
        print("\nAll expected features:")
        for i, feature in enumerate(swing_features, 1):
            print(f"{i:3d}. {feature}")
        
        # Also get the preprocessor features
        swing_preprocessor = models.get('swing_preprocessor')
        if swing_preprocessor:
            print(f"\nPreprocessor feature names:")
            if hasattr(swing_preprocessor, 'feature_names_in_'):
                for i, feature in enumerate(swing_preprocessor.feature_names_in_, 1):
                    print(f"{i:3d}. {feature}")
        
        return swing_features
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

if __name__ == "__main__":
    features = extract_swing_model_features() 
import pandas as pd

def extract_swing_model_features():
    """Extract all features expected by the swing vs no swing model"""
    try:
        # Load the model
        models = joblib.load('sequential_models.pkl')
        
        # Get the swing features
        swing_features = models.get('swing_features', [])
        
        print("SWING VS NO SWING MODEL FEATURES:")
        print("=" * 50)
        print(f"Total features expected: {len(swing_features)}")
        print("\nAll expected features:")
        for i, feature in enumerate(swing_features, 1):
            print(f"{i:3d}. {feature}")
        
        # Also get the preprocessor features
        swing_preprocessor = models.get('swing_preprocessor')
        if swing_preprocessor:
            print(f"\nPreprocessor feature names:")
            if hasattr(swing_preprocessor, 'feature_names_in_'):
                for i, feature in enumerate(swing_preprocessor.feature_names_in_, 1):
                    print(f"{i:3d}. {feature}")
        
        return swing_features
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

if __name__ == "__main__":
    features = extract_swing_model_features() 
 
 
 
 
 
 
 
 
import pandas as pd

def extract_swing_model_features():
    """Extract all features expected by the swing vs no swing model"""
    try:
        # Load the model
        models = joblib.load('sequential_models.pkl')
        
        # Get the swing features
        swing_features = models.get('swing_features', [])
        
        print("SWING VS NO SWING MODEL FEATURES:")
        print("=" * 50)
        print(f"Total features expected: {len(swing_features)}")
        print("\nAll expected features:")
        for i, feature in enumerate(swing_features, 1):
            print(f"{i:3d}. {feature}")
        
        # Also get the preprocessor features
        swing_preprocessor = models.get('swing_preprocessor')
        if swing_preprocessor:
            print(f"\nPreprocessor feature names:")
            if hasattr(swing_preprocessor, 'feature_names_in_'):
                for i, feature in enumerate(swing_preprocessor.feature_names_in_, 1):
                    print(f"{i:3d}. {feature}")
        
        return swing_features
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

if __name__ == "__main__":
    features = extract_swing_model_features() 
import pandas as pd

def extract_swing_model_features():
    """Extract all features expected by the swing vs no swing model"""
    try:
        # Load the model
        models = joblib.load('sequential_models.pkl')
        
        # Get the swing features
        swing_features = models.get('swing_features', [])
        
        print("SWING VS NO SWING MODEL FEATURES:")
        print("=" * 50)
        print(f"Total features expected: {len(swing_features)}")
        print("\nAll expected features:")
        for i, feature in enumerate(swing_features, 1):
            print(f"{i:3d}. {feature}")
        
        # Also get the preprocessor features
        swing_preprocessor = models.get('swing_preprocessor')
        if swing_preprocessor:
            print(f"\nPreprocessor feature names:")
            if hasattr(swing_preprocessor, 'feature_names_in_'):
                for i, feature in enumerate(swing_preprocessor.feature_names_in_, 1):
                    print(f"{i:3d}. {feature}")
        
        return swing_features
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

if __name__ == "__main__":
    features = extract_swing_model_features() 
 
 
 
 
 
 
 
 
 
 
 
 