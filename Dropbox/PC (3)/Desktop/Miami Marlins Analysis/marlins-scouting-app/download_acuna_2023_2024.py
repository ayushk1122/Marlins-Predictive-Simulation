import pybaseball as pb
import pandas as pd
from datetime import datetime
import warnings
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from feature_engineering import engineer_features_for_model
warnings.filterwarnings('ignore')

def download_acuna_2023_2024():
    """
    Download Ronald Acuña Jr.'s 2023-2024 Statcast data and test whiff vs contact model
    """
    print("🔍 Downloading Ronald Acuña Jr.'s 2023-2024 data...")
    
    # Ronald Acuña Jr.'s MLBAM ID
    acuna_id = 660670
    
    all_data = []
    
    for year in [2025, 2025]:
        print(f"📅 Downloading {year} season...")
        
        try:
            # Download Statcast data for Acuña for this year
            year_data = pb.statcast_batter(
                f"{year}-07-09", 
                f"{year}-07-14", 
                acuna_id
            )
            
            if not year_data.empty:
                print(f"✅ {year}: {len(year_data)} pitches downloaded")
                all_data.append(year_data)
            else:
                print(f"⚠️  {year}: No data found")
                
        except Exception as e:
            print(f"❌ Error downloading {year} data: {e}")
    
    if all_data:
        # Combine all years
        complete_data = pd.concat(all_data, ignore_index=True)
        
        print(f"\n📊 Downloaded data shape: {complete_data.shape}")
        print(f"📅 Date range: {complete_data['game_date'].min()} to {complete_data['game_date'].max()}")
        
        # Save the dataset
        filename = 'ronald_acuna_jr_2023_2024_statcast.csv'
        complete_data.to_csv(filename, index=False)
        print(f"💾 Data saved to: {filename}")
        
        return complete_data
    else:
        print("❌ No data was downloaded")
        return None

def prepare_test_features(df, model_type='whiff_vs_contact'):
    """Prepare features for model testing using the feature engineering module"""
    df = df.copy()
    
    # Create swing and whiff columns
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    whiff_events = ['swinging_strike', 'swinging_strike_blocked']
    
    df['is_swing'] = df['description'].isin(swing_events).astype(int)
    df['is_whiff'] = df['description'].isin(whiff_events).astype(int)
    
    if model_type == 'whiff_vs_contact':
        # Filter for swings only
        swing_df = df[df['is_swing'] == 1].copy()
        
        if len(swing_df) == 0:
            print("✗ No swing data found!")
            return None
        
        print(f"✓ Found {len(swing_df)} swings in 2023-2024 data")
        
        # Create whiff vs contact target
        swing_df['is_whiff_binary'] = swing_df['is_whiff'].astype(int)
        
        # Use the feature engineering module
        swing_df = engineer_features_for_model(swing_df, 'whiff_vs_contact', 'acuna', 'unknown')
        
        return swing_df
    
    elif model_type == 'swing_vs_noswing':
        # Use all pitches for swing vs no swing
        print(f"✓ Found {len(df)} total pitches in 2023-2024 data")
        print(f"  Swings: {df['is_swing'].sum()} ({df['is_swing'].mean()*100:.1f}%)")
        print(f"  No swings: {(df['is_swing'] == 0).sum()} ({(df['is_swing'] == 0).mean()*100:.1f}%)")
        
        # Create swing vs no swing target
        df['is_swing_binary'] = df['is_swing'].astype(int)
        
        # Use the feature engineering module
        df = engineer_features_for_model(df, 'swing_vs_noswing', 'acuna', 'unknown')
        
        return df
    
    else:
        print(f"✗ Unknown model type: {model_type}")
        return None

def test_whiff_vs_contact_model(df):
    """Test the whiff vs contact model on new 2023-2024 data"""
    print("\n" + "="*60)
    print("TESTING WHIFF VS CONTACT MODEL ON 2023-2024 DATA")
    print("="*60)
    
    # Load model
    try:
        model = joblib.load('whiff_vs_contact_model.pkl')
        preprocessor = joblib.load('whiff_vs_contact_preprocessor.pkl')
        print("✓ Loaded whiff vs contact model")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Prepare features
    swing_df = prepare_test_features(df, 'whiff_vs_contact')
    if swing_df is None:
        return
    
    # Prepare target
    y_true = swing_df['is_whiff_binary'].values
    
    print(f"\nDataset Analysis:")
    print(f"Total swings: {len(swing_df)}")
    print(f"  Contacts: {(y_true == 0).sum()} ({(y_true == 0).sum()/len(swing_df)*100:.1f}%)")
    print(f"  Whiffs: {(y_true == 1).sum()} ({(y_true == 1).sum()/len(swing_df)*100:.1f}%)")
    
    # Prepare features for prediction
    expected_features = preprocessor['num_features'] + preprocessor['cat_features']
    available_features = [f for f in expected_features if f in swing_df.columns]
    missing_features = [f for f in expected_features if f not in swing_df.columns]
    
    print(f"Expected features: {len(expected_features)}")
    print(f"Available features: {len(available_features)}")
    print(f"Missing features: {len(missing_features)}")
    if missing_features:
        print(f"Missing: {missing_features[:10]}...")  # Show first 10
    
    X = swing_df[available_features].copy()
    
    # Check for duplicate columns
    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Warning: Found duplicate columns: {duplicate_cols}")
        # Remove duplicates by keeping only the first occurrence
        X = X.loc[:, ~X.columns.duplicated()]
    
    # Handle categorical features
    cat_features = preprocessor['cat_features']
    for feat in cat_features:
        if feat in X.columns:
            X[feat] = X[feat].fillna('unknown').astype(str)
    
    # Handle numeric features
    num_features = preprocessor['num_features']
    for feat in num_features:
        if feat in X.columns:
            try:
                # Ensure we're working with a Series, not a DataFrame
                if isinstance(X[feat], pd.DataFrame):
                    print(f"Warning: {feat} is a DataFrame, taking first column")
                    X[feat] = X[feat].iloc[:, 0]
                X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(0)
            except Exception as e:
                print(f"Error processing feature {feat}: {e}")
                X[feat] = 0.0
    
    # Make predictions
    try:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature matrix dtypes: {X.dtypes.value_counts()}")
        print(f"Sample feature values:")
        for col in X.columns[:5]:
            print(f"  {col}: {X[col].iloc[0]} (dtype: {X[col].dtype})")
        
        y_proba = model.predict_proba(X)
        
        # Apply confidence difference threshold approach
        whiff_threshold = 0.35
        contact_threshold = 0.75
        confidence_diff_threshold = 0.15
        
        # Calculate probability differences
        contact_proba = y_proba[:, 0]
        whiff_proba = y_proba[:, 1]
        proba_diff = contact_proba - whiff_proba
        
        # Apply confidence difference threshold logic
        y_pred_threshold = np.where(
            whiff_proba >= whiff_threshold, 1,  # High whiff prob -> predict whiff
            np.where(
                contact_proba >= contact_threshold, 0,  # High contact prob -> predict contact
                np.where(
                    proba_diff >= confidence_diff_threshold, 0,  # Sufficient confidence difference -> predict contact
                    1  # Default to whiff for uncertain cases
                )
            )
        )
        
        # Standard predictions (no threshold)
        y_pred_standard = model.predict(X)
        
        print(f"\nPrediction Results:")
        print(f"Total swings: {len(swing_df)}")
        print(f"Actual whiffs: {y_true.sum()} ({(y_true.sum()/len(swing_df)*100):.1f}%)")
        print(f"Actual contacts: {len(swing_df) - y_true.sum()} ({((len(swing_df) - y_true.sum())/len(swing_df)*100):.1f}%)")
        print(f"Standard predicted whiffs: {y_pred_standard.sum()}")
        print(f"Confidence Difference predicted whiffs: {y_pred_threshold.sum()}")
        
        # Calculate uncertain cases
        uncertain_cases = ((whiff_proba < whiff_threshold) & 
                          (contact_proba < contact_threshold) & 
                          (proba_diff < confidence_diff_threshold)).sum()
        
        print(f"Uncertain cases defaulted to whiff: {uncertain_cases}")
        print(f"Confidence difference threshold: {confidence_diff_threshold*100:.0f}%")
        
        # Evaluate both approaches
        accuracy_standard = accuracy_score(y_true, y_pred_standard)
        accuracy_threshold = accuracy_score(y_true, y_pred_threshold)
        
        print(f"\nStandard Accuracy: {accuracy_standard:.3f} ({accuracy_standard*100:.1f}%)")
        print(f"Confidence Difference Accuracy: {accuracy_threshold:.3f} ({accuracy_threshold*100:.1f}%)")
        
        print("\nStandard Classification Report:")
        print(classification_report(y_true, y_pred_standard, target_names=['Contact', 'Whiff']))
        
        print(f"\nConfidence Difference Classification Report:")
        print(classification_report(y_true, y_pred_threshold, target_names=['Contact', 'Whiff']))
        
        print("\nStandard Confusion Matrix:")
        cm_standard = confusion_matrix(y_true, y_pred_standard)
        print("          Predicted")
        print("          Contact  Whiff")
        print(f"Actual Contact  {cm_standard[0,0]:6d}  {cm_standard[0,1]:6d}")
        print(f"      Whiff     {cm_standard[1,0]:6d}  {cm_standard[1,1]:6d}")
        
        print(f"\nConfidence Difference Confusion Matrix:")
        cm_threshold = confusion_matrix(y_true, y_pred_threshold)
        print("          Predicted")
        print("          Contact  Whiff")
        print(f"Actual Contact  {cm_threshold[0,0]:6d}  {cm_threshold[0,1]:6d}")
        print(f"      Whiff     {cm_threshold[1,0]:6d}  {cm_threshold[1,1]:6d}")
        
        # Pitch type analysis
        print(f"\nPITCH TYPE ANALYSIS:")
        print("-" * 40)
        
        for pitch_type in swing_df['pitch_type'].unique():
            pitch_mask = swing_df['pitch_type'] == pitch_type
            if pitch_mask.sum() > 5:  # Only analyze if enough samples
                pitch_accuracy = (y_true[pitch_mask] == y_pred_threshold[pitch_mask]).mean()
                pitch_count = pitch_mask.sum()
                print(f"  {pitch_type}: {pitch_accuracy:.3f} accuracy ({pitch_count} swings)")
        
        # Zone analysis
        print(f"\nZONE ANALYSIS:")
        print("-" * 40)
        
        for zone in sorted(swing_df['zone'].unique()):
            zone_mask = swing_df['zone'] == zone
            if zone_mask.sum() > 5:  # Only analyze if enough samples
                zone_accuracy = (y_true[zone_mask] == y_pred_threshold[zone_mask]).mean()
                zone_count = zone_mask.sum()
                print(f"  Zone {zone}: {zone_accuracy:.3f} accuracy ({zone_count} swings)")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    print("\n" + "="*60)
    print("WHIFF VS CONTACT MODEL TESTING COMPLETE")
    print("="*60)

def test_swing_vs_noswing_model(df):
    """Test the swing vs no swing model on new 2023-2024 data"""
    print("\n" + "="*60)
    print("TESTING SWING VS NO SWING MODEL ON 2023-2024 DATA")
    print("="*60)
    
    # Load model
    try:
        models = joblib.load('sequential_models.pkl')
        model = models['swing_model']
        preprocessor = models['swing_preprocessor']
        features = models['swing_features']
        print("✓ Loaded swing vs no swing model")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Prepare features
    all_df = prepare_test_features(df, 'swing_vs_noswing')
    if all_df is None:
        return
    
    # Prepare target
    y_true = all_df['is_swing_binary'].values
    
    print(f"\nDataset Analysis:")
    print(f"Total pitches: {len(all_df)}")
    print(f"  Swings: {(y_true == 1).sum()} ({(y_true == 1).sum()/len(all_df)*100:.1f}%)")
    print(f"  No swings: {(y_true == 0).sum()} ({(y_true == 0).sum()/len(all_df)*100:.1f}%)")
    
    # Prepare features for prediction
    # Use the features that the model was trained on
    available_features = [f for f in features if f in all_df.columns]
    missing_features = [f for f in features if f not in all_df.columns]
    
    if missing_features:
        print(f"⚠️ Missing features: {len(missing_features)}")
        print(f"Missing: {missing_features[:5]}...")  # Show first 5
    
    X = all_df[available_features].copy()
    
    # Check for duplicate columns
    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Warning: Found duplicate columns: {duplicate_cols}")
        # Remove duplicates by keeping only the first occurrence
        X = X.loc[:, ~X.columns.duplicated()]
    
    # Handle missing values
    for feat in available_features:
        try:
            # Ensure we're working with a Series, not a DataFrame
            if isinstance(X[feat], pd.DataFrame):
                print(f"Warning: {feat} is a DataFrame, taking first column")
                X[feat] = X[feat].iloc[:, 0]
            
            if X[feat].dtype == 'object':
                X[feat] = X[feat].fillna('unknown').astype(str)
            else:
                X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(0)
        except Exception as e:
            print(f"Error processing feature {feat}: {e}")
            X[feat] = 0.0
    
    # Make predictions
    try:
        y_proba = model.predict_proba(X)
        
        # Apply confidence difference threshold approach
        swing_threshold = 0.35
        noswing_threshold = 0.75
        confidence_diff_threshold = 0.15
        
        # Calculate probability differences
        noswing_proba = y_proba[:, 0]
        swing_proba = y_proba[:, 1]
        proba_diff = swing_proba - noswing_proba
        
        # Apply confidence difference threshold logic
        y_pred_threshold = np.where(
            swing_proba >= swing_threshold, 1,  # High swing prob -> predict swing
            np.where(
                noswing_proba >= noswing_threshold, 0,  # High no swing prob -> predict no swing
                np.where(
                    proba_diff >= confidence_diff_threshold, 1,  # Sufficient confidence difference -> predict swing
                    0  # Default to no swing for uncertain cases
                )
            )
        )
        
        # Standard predictions (no threshold)
        y_pred_standard = model.predict(X)
        
        print(f"\nPrediction Results:")
        print(f"Total pitches: {len(all_df)}")
        print(f"Actual swings: {y_true.sum()} ({(y_true.sum()/len(all_df)*100):.1f}%)")
        print(f"Actual no swings: {len(all_df) - y_true.sum()} ({((len(all_df) - y_true.sum())/len(all_df)*100):.1f}%)")
        print(f"Standard predicted swings: {y_pred_standard.sum()}")
        print(f"Confidence Difference predicted swings: {y_pred_threshold.sum()}")
        
        # Calculate uncertain cases
        uncertain_cases = ((swing_proba < swing_threshold) & 
                          (noswing_proba < noswing_threshold) & 
                          (proba_diff < confidence_diff_threshold)).sum()
        
        print(f"Uncertain cases defaulted to no swing: {uncertain_cases}")
        print(f"Confidence difference threshold: {confidence_diff_threshold*100:.0f}%")
        
        # Evaluate both approaches
        accuracy_standard = accuracy_score(y_true, y_pred_standard)
        accuracy_threshold = accuracy_score(y_true, y_pred_threshold)
        
        print(f"\nStandard Accuracy: {accuracy_standard:.3f} ({accuracy_standard*100:.1f}%)")
        print(f"Confidence Difference Accuracy: {accuracy_threshold:.3f} ({accuracy_threshold*100:.1f}%)")
        
        print("\nStandard Classification Report:")
        print(classification_report(y_true, y_pred_standard, target_names=['No Swing', 'Swing']))
        
        print(f"\nConfidence Difference Classification Report:")
        print(classification_report(y_true, y_pred_threshold, target_names=['No Swing', 'Swing']))
        
        print("\nStandard Confusion Matrix:")
        cm_standard = confusion_matrix(y_true, y_pred_standard)
        print("          Predicted")
        print("          No Swing  Swing")
        print(f"Actual No Swing  {cm_standard[0,0]:6d}  {cm_standard[0,1]:6d}")
        print(f"      Swing      {cm_standard[1,0]:6d}  {cm_standard[1,1]:6d}")
        
        print(f"\nConfidence Difference Confusion Matrix:")
        cm_threshold = confusion_matrix(y_true, y_pred_threshold)
        print("          Predicted")
        print("          No Swing  Swing")
        print(f"Actual No Swing  {cm_threshold[0,0]:6d}  {cm_threshold[0,1]:6d}")
        print(f"      Swing      {cm_threshold[1,0]:6d}  {cm_threshold[1,1]:6d}")
        
        # Pitch type analysis
        print(f"\nPITCH TYPE ANALYSIS:")
        print("-" * 40)
        
        for pitch_type in all_df['pitch_type'].unique():
            pitch_mask = all_df['pitch_type'] == pitch_type
            if pitch_mask.sum() > 5:  # Only analyze if enough samples
                pitch_accuracy = (y_true[pitch_mask] == y_pred_threshold[pitch_mask]).mean()
                pitch_count = pitch_mask.sum()
                print(f"  {pitch_type}: {pitch_accuracy:.3f} accuracy ({pitch_count} pitches)")
        
        # Zone analysis
        print(f"\nZONE ANALYSIS:")
        print("-" * 40)
        
        for zone in sorted(all_df['zone'].unique()):
            zone_mask = all_df['zone'] == zone
            if zone_mask.sum() > 5:  # Only analyze if enough samples
                zone_accuracy = (y_true[zone_mask] == y_pred_threshold[zone_mask]).mean()
                zone_count = zone_mask.sum()
                print(f"  Zone {zone}: {zone_accuracy:.3f} accuracy ({zone_count} pitches)")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    print("\n" + "="*60)
    print("SWING VS NO SWING MODEL TESTING COMPLETE")
    print("="*60)

def main():
    """Main function to download data and test both models"""
    print("🚀 RONALD ACUÑA JR. 2023-2024 DATA TESTER")
    print("=" * 60)
    
    # Download data
    df = download_acuna_2023_2024()
    if df is None:
        return
    
    # # Test both models on new data
    # test_whiff_vs_contact_model(df)
    # test_swing_vs_noswing_model(df)
    
    # print("\n" + "="*60)
    # print("ALL MODEL TESTING COMPLETE")
    # print("="*60)

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 