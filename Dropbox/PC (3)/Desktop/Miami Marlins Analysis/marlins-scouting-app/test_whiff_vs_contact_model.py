import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_holdout_data():
    """Load holdout data for testing"""
    try:
        df = pd.read_csv('ronald_acuna_jr_holdout_statcast.csv')
        print(f"✓ Loaded holdout data with {len(df)} pitches")
        return df
    except Exception as e:
        print(f"✗ Error loading holdout data: {e}")
        return None

def load_whiff_model():
    """Load the trained whiff vs contact model"""
    try:
        model = joblib.load('whiff_vs_contact_model.pkl')
        preprocessor = joblib.load('whiff_vs_contact_preprocessor.pkl')
        print("✓ Loaded whiff vs contact model")
        return model, preprocessor
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def prepare_test_features(df):
    """Prepare features for whiff vs contact testing"""
    df = df.copy()
    
    # Create swing and whiff columns
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    whiff_events = ['swinging_strike', 'swinging_strike_blocked']
    
    df['is_swing'] = df['description'].isin(swing_events).astype(int)
    df['is_whiff'] = df['description'].isin(whiff_events).astype(int)
    
    # Filter for swings only
    swing_df = df[df['is_swing'] == 1].copy()
    
    if len(swing_df) == 0:
        print("✗ No swing data found in holdout set!")
        return None
    
    print(f"✓ Found {len(swing_df)} swings in holdout data")
    
    # Create whiff vs contact target
    swing_df['is_whiff_binary'] = swing_df['is_whiff'].astype(int)
    
    # ENGINEERED FEATURES (same as training)
    
    # Zone distance
    swing_df['zone_center_x'] = 0
    swing_df['zone_center_z'] = (swing_df['sz_top'] + swing_df['sz_bot']) / 2
    swing_df['zone_distance'] = np.sqrt(
        (swing_df['plate_x'] - swing_df['zone_center_x'])**2 + 
        (swing_df['plate_z'] - swing_df['zone_center_z'])**2
    )
    
    # Movement features
    swing_df['horizontal_break'] = swing_df['api_break_x_batter_in'].fillna(0)
    swing_df['vertical_break'] = swing_df['api_break_z_with_gravity'].fillna(0)
    swing_df['movement_magnitude'] = np.sqrt(swing_df['horizontal_break']**2 + swing_df['vertical_break']**2)
    
    # Count features
    swing_df['count_pressure'] = swing_df['balls'] - swing_df['strikes']
    swing_df['count_total'] = swing_df['balls'] + swing_df['strikes']
    swing_df['behind_in_count'] = (swing_df['strikes'] > swing_df['balls']).astype(int)
    swing_df['ahead_in_count'] = (swing_df['balls'] > swing_df['strikes']).astype(int)
    swing_df['two_strikes'] = (swing_df['strikes'] >= 2).astype(int)
    swing_df['three_balls'] = (swing_df['balls'] >= 3).astype(int)
    
    # Zone features
    swing_df['in_strike_zone'] = ((swing_df['plate_x'] >= -0.85) & (swing_df['plate_x'] <= 0.85) & 
                                 (swing_df['plate_z'] >= swing_df['sz_bot']) & (swing_df['plate_z'] <= swing_df['sz_top'])).astype(int)
    swing_df['far_from_zone'] = (swing_df['zone_distance'] > 1.0).astype(int)
    swing_df['high_pitch'] = (swing_df['plate_z'] > swing_df['sz_top']).astype(int)
    swing_df['low_pitch'] = (swing_df['plate_z'] < swing_df['sz_bot']).astype(int)
    swing_df['inside_pitch'] = (swing_df['plate_x'] < -0.85).astype(int)
    swing_df['outside_pitch'] = (swing_df['plate_x'] > 0.85).astype(int)
    
    # Pitch type features
    swing_df['is_fastball'] = swing_df['pitch_type'].isin(['FF', 'SI', 'FC']).astype(int)
    swing_df['is_breaking_ball'] = swing_df['pitch_type'].isin(['SL', 'CU', 'KC']).astype(int)
    swing_df['is_offspeed'] = swing_df['pitch_type'].isin(['CH', 'FS']).astype(int)
    
    # Velocity features
    swing_df['high_velocity'] = (swing_df['release_speed'] > 95).astype(int)
    swing_df['low_velocity'] = (swing_df['release_speed'] < 85).astype(int)
    swing_df['velocity_movement_ratio'] = swing_df['release_speed'] / (swing_df['movement_magnitude'] + 0.1)
    
    # Movement features
    swing_df['high_movement'] = (swing_df['movement_magnitude'] > 6).astype(int)
    swing_df['low_movement'] = (swing_df['movement_magnitude'] < 2).astype(int)
    swing_df['movement_ratio'] = np.abs(swing_df['horizontal_break']) / (np.abs(swing_df['vertical_break']) + 0.1)
    
    # Interaction features
    swing_df['zone_distance_x_count_pressure'] = swing_df['zone_distance'] * swing_df['count_pressure']
    swing_df['movement_x_count_pressure'] = swing_df['movement_magnitude'] * swing_df['count_pressure']
    swing_df['in_zone_x_two_strikes'] = swing_df['in_strike_zone'] * swing_df['two_strikes']
    swing_df['far_from_zone_x_ahead'] = swing_df['far_from_zone'] * swing_df['ahead_in_count']
    
    # Advanced features
    swing_df['velocity_diff_from_avg'] = swing_df['release_speed'] - swing_df['release_speed'].mean()
    swing_df['movement_diff_from_avg'] = swing_df['movement_magnitude'] - swing_df['movement_magnitude'].mean()
    
    # Zone-specific features
    swing_df['zone_corner'] = ((swing_df['zone'] == 1) | (swing_df['zone'] == 3) | 
                               (swing_df['zone'] == 7) | (swing_df['zone'] == 9)).astype(int)
    swing_df['zone_heart'] = ((swing_df['zone'] == 2) | (swing_df['zone'] == 5) | 
                              (swing_df['zone'] == 8)).astype(int)
    swing_df['zone_shadow'] = ((swing_df['zone'] == 4) | (swing_df['zone'] == 6)).astype(int)
    
    # BABIP features (load from CSV)
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
        for idx, row in swing_df.iterrows():
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
        
        babip_df_features = pd.DataFrame(babip_features, index=swing_df.index)
        swing_df = pd.concat([swing_df, babip_df_features], axis=1)
        print("✓ Added BABIP features")
    except:
        # Add default BABIP features
        swing_df['batting_average_bip'] = 0.25
        swing_df['whiff_rate'] = 0.35
        swing_df['field_out_rate_bip'] = 0.40
        swing_df['balls_in_play'] = 0
        swing_df['total_swings'] = 0
        swing_df['total_whiffs'] = 0
        print("Added default BABIP features")
    
    return swing_df

def test_threshold_values(y_true, y_proba):
    """Test different threshold values to find optimal whiff prediction threshold"""
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("="*60)
    
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    results = []
    
    print(f"{'Threshold':<12} {'Whiff Pred':<10} {'Whiff Prec':<10} {'Whiff Recall':<12} {'Balanced Acc':<12}")
    print("-" * 60)
    
    for threshold in thresholds:
        y_pred = np.where(y_proba[:, 1] >= threshold, 1, 0)
        
        # Calculate metrics for imbalanced data
        whiff_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        whiff_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        whiff_predictions = y_pred.sum()
        
        results.append({
            'threshold': threshold,
            'whiff_pred': whiff_predictions,
            'whiff_precision': whiff_precision,
            'whiff_recall': whiff_recall,
            'balanced_acc': balanced_acc
        })
        
        print(f"{threshold:<12.2f} {whiff_predictions:<10d} {whiff_precision:<10.3f} {whiff_recall:<12.3f} {balanced_acc:<12.3f}")
    
    # Find best threshold for whiff precision (fewer false positives)
    best_precision = max(results, key=lambda x: x['whiff_precision'])
    print(f"\nBest threshold for whiff precision: {best_precision['threshold']:.2f} (precision: {best_precision['whiff_precision']:.3f})")
    
    # Find best threshold for whiff recall (catch more whiffs)
    best_recall = max(results, key=lambda x: x['whiff_recall'])
    print(f"Best threshold for whiff recall: {best_recall['threshold']:.2f} (recall: {best_recall['whiff_recall']:.3f})")
    
    # Find best threshold for balanced accuracy
    best_balanced = max(results, key=lambda x: x['balanced_acc'])
    print(f"Best threshold for balanced accuracy: {best_balanced['threshold']:.2f} (balanced acc: {best_balanced['balanced_acc']:.3f})")
    
    return results

def analyze_whiff_predictions(df, y_true, y_pred, probabilities):
    """Analyze whiff vs contact prediction results"""
    print("\n" + "="*60)
    print("WHIFF VS CONTACT PREDICTION ANALYSIS")
    print("="*60)
    
    # Overall accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class analysis
    print(f"\nPER-CLASS ANALYSIS:")
    print("-" * 40)
    
    # Contact class
    contact_mask = y_true == 0
    if contact_mask.sum() > 0:
        contact_accuracy = (y_true[contact_mask] == y_pred[contact_mask]).mean()
        contact_count = contact_mask.sum()
        print(f"\nCONTACT (n={contact_count}):")
        print(f"  Accuracy: {contact_accuracy:.3f} ({contact_accuracy*100:.1f}%)")
        
        # Misclassifications
        contact_misclassified = contact_mask & (y_true != y_pred)
        if contact_misclassified.sum() > 0:
            print(f"  Misclassified as whiff: {contact_misclassified.sum()} ({contact_misclassified.sum()/contact_count*100:.1f}%)")
    
    # Whiff class
    whiff_mask = y_true == 1
    if whiff_mask.sum() > 0:
        whiff_accuracy = (y_true[whiff_mask] == y_pred[whiff_mask]).mean()
        whiff_count = whiff_mask.sum()
        print(f"\nWHIFF (n={whiff_count}):")
        print(f"  Accuracy: {whiff_accuracy:.3f} ({whiff_accuracy*100:.1f}%)")
        
        # Misclassifications
        whiff_misclassified = whiff_mask & (y_true != y_pred)
        if whiff_misclassified.sum() > 0:
            print(f"  Misclassified as contact: {whiff_misclassified.sum()} ({whiff_misclassified.sum()/whiff_count*100:.1f}%)")
    
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
    
    # Detailed confidence analysis for misclassifications
    print(f"\nDETAILED CONFIDENCE ANALYSIS FOR MISCLASSIFICATIONS:")
    print("-" * 60)
    
    # Misclassified contacts (true contact, predicted whiff)
    contact_misclassified = (y_true == 0) & (y_pred == 1)
    if contact_misclassified.sum() > 0:
        print(f"\nMISCLASSIFIED CONTACTS (True Contact, Predicted Whiff):")
        print(f"  {'Pitch Type':<8} {'Zone':<4} {'Velocity':<8} {'Movement':<8} {'Count':<6} {'Confidence':<10}")
        print(f"  {'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
        
        misclassified_indices = np.where(contact_misclassified)[0]
        for idx in misclassified_indices:
            row = df.iloc[idx]
            max_prob = max_probs[idx]
            whiff_prob = probabilities[idx][1]  # Probability of whiff
            contact_prob = probabilities[idx][0]  # Probability of contact
            
            print(f"  {row['pitch_type']:<8} {row['zone']:<4.0f} {row['release_speed']:<8.1f} {row['movement_magnitude']:<8.2f} {row['balls']:.0f}-{row['strikes']:<3.0f} {max_prob:<10.1%}")
            print(f"    Whiff prob: {whiff_prob:.1%}, Contact prob: {contact_prob:.1%}")
    
    # Misclassified whiffs (true whiff, predicted contact)
    whiff_misclassified = (y_true == 1) & (y_pred == 0)
    if whiff_misclassified.sum() > 0:
        print(f"\nMISCLASSIFIED WHIFFS (True Whiff, Predicted Contact):")
        print(f"  {'Pitch Type':<8} {'Zone':<4} {'Velocity':<8} {'Movement':<8} {'Count':<6} {'Confidence':<10}")
        print(f"  {'-'*8} {'-'*4} {'-'*8} {'-'*8} {'-'*6} {'-'*10}")
        
        misclassified_indices = np.where(whiff_misclassified)[0]
        for idx in misclassified_indices:
            row = df.iloc[idx]
            max_prob = max_probs[idx]
            whiff_prob = probabilities[idx][1]  # Probability of whiff
            contact_prob = probabilities[idx][0]  # Probability of contact
            
            print(f"  {row['pitch_type']:<8} {row['zone']:<4.0f} {row['release_speed']:<8.1f} {row['movement_magnitude']:<8.2f} {row['balls']:.0f}-{row['strikes']:<3.0f} {max_prob:<10.1%}")
            print(f"    Whiff prob: {whiff_prob:.1%}, Contact prob: {contact_prob:.1%}")
    
    # Pitch type analysis
    print(f"\nPITCH TYPE ANALYSIS:")
    print("-" * 40)
    
    for pitch_type in df['pitch_type'].unique():
        pitch_mask = df['pitch_type'] == pitch_type
        if pitch_mask.sum() > 5:  # Only analyze if enough samples
            pitch_accuracy = (y_true[pitch_mask] == y_pred[pitch_mask]).mean()
            pitch_count = pitch_mask.sum()
            print(f"  {pitch_type}: {pitch_accuracy:.3f} accuracy ({pitch_count} swings)")
    
    # Zone analysis
    print(f"\nZONE ANALYSIS:")
    print("-" * 40)
    
    for zone in sorted(df['zone'].unique()):
        zone_mask = df['zone'] == zone
        if zone_mask.sum() > 5:  # Only analyze if enough samples
            zone_accuracy = (y_true[zone_mask] == y_pred[zone_mask]).mean()
            zone_count = zone_mask.sum()
            print(f"  Zone {zone}: {zone_accuracy:.3f} accuracy ({zone_count} swings)")

def main():
    """Main function to test whiff vs contact model"""
    print("Testing Whiff vs Contact Model on Holdout Data")
    print("=" * 60)
    
    # Load holdout data
    df = load_holdout_data()
    if df is None:
        return
    
    # Load model
    model, preprocessor = load_whiff_model()
    if model is None:
        return
    
    # Prepare features
    swing_df = prepare_test_features(df)
    if swing_df is None:
        return
    
    # Prepare target
    y_true = swing_df['is_whiff_binary'].values
    
    # Create balanced evaluation dataset
    print(f"\nCreating balanced evaluation dataset...")
    print(f"Original dataset: {len(swing_df)} total swings")
    print(f"  Contacts: {(y_true == 0).sum()} ({((y_true == 0).sum()/len(swing_df)*100):.1f}%)")
    print(f"  Whiffs: {(y_true == 1).sum()} ({((y_true == 1).sum()/len(swing_df)*100):.1f}%)")
    
    # Find indices for contacts and whiffs
    contact_indices = np.where(y_true == 0)[0]
    whiff_indices = np.where(y_true == 1)[0]
    
    # Determine the smaller class size
    min_class_size = min(len(contact_indices), len(whiff_indices))
    print(f"Balanced dataset will use {min_class_size} samples from each class")
    
    # Randomly sample equal numbers from each class
    np.random.seed(42)  # For reproducibility
    balanced_contact_indices = np.random.choice(contact_indices, size=min_class_size, replace=False)
    balanced_whiff_indices = np.random.choice(whiff_indices, size=min_class_size, replace=False)
    
    # Combine balanced indices
    balanced_indices = np.concatenate([balanced_contact_indices, balanced_whiff_indices])
    np.random.shuffle(balanced_indices)  # Shuffle to avoid class ordering
    
    # Create balanced dataset
    balanced_swing_df = swing_df.iloc[balanced_indices].copy()
    balanced_y_true = y_true[balanced_indices]
    
    print(f"Balanced dataset: {len(balanced_swing_df)} total swings")
    print(f"  Contacts: {(balanced_y_true == 0).sum()} (50.0%)")
    print(f"  Whiffs: {(balanced_y_true == 1).sum()} (50.0%)")
    
    # Prepare features for prediction
    X = balanced_swing_df[preprocessor['num_features'] + preprocessor['cat_features']].copy()
    
    # Handle categorical features
    cat_features = preprocessor['cat_features']
    for feat in cat_features:
        if feat in X.columns:
            X[feat] = X[feat].fillna('unknown').astype(str)
    
    # Handle numeric features
    num_features = preprocessor['num_features']
    for feat in num_features:
        if feat in X.columns:
            X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(0)
    
    # Make predictions
    try:
        y_proba = model.predict_proba(X)
        
        # Apply confidence threshold for whiff predictions
        whiff_threshold = 0.50  # 50% confidence required to predict whiff
        y_pred_threshold = np.where(y_proba[:, 1] >= whiff_threshold, 1, 0)  # 1 = whiff, 0 = contact
        
        # Standard predictions (no threshold)
        y_pred_standard = model.predict(X)
        
        print(f"\nPrediction Results (BALANCED DATASET):")
        print(f"Total swings: {len(balanced_swing_df)}")
        print(f"Actual whiffs: {balanced_y_true.sum()} (50.0%)")
        print(f"Actual contacts: {len(balanced_swing_df) - balanced_y_true.sum()} (50.0%)")
        print(f"Standard predicted whiffs: {y_pred_standard.sum()}")
        print(f"Threshold predicted whiffs (≥{whiff_threshold*100:.0f}%): {y_pred_threshold.sum()}")
        
        print(f"\nBalanced Dataset Analysis:")
        print(f"  Equal representation of contacts and whiffs")
        print(f"  Overall accuracy now reflects true model performance!")
        
        # Evaluate both approaches
        accuracy_standard = accuracy_score(balanced_y_true, y_pred_standard)
        accuracy_threshold = accuracy_score(balanced_y_true, y_pred_threshold)
        
        print(f"\nStandard Accuracy: {accuracy_standard:.3f} ({accuracy_standard*100:.1f}%)")
        print(f"Threshold Accuracy: {accuracy_threshold:.3f} ({accuracy_threshold*100:.1f}%)")
        
        print("\nStandard Classification Report:")
        print(classification_report(balanced_y_true, y_pred_standard, target_names=['Contact', 'Whiff']))
        
        print(f"\nThreshold Classification Report (≥{whiff_threshold*100:.0f}%):")
        print(classification_report(balanced_y_true, y_pred_threshold, target_names=['Contact', 'Whiff']))
        
        print("\nStandard Confusion Matrix:")
        cm_standard = confusion_matrix(balanced_y_true, y_pred_standard)
        print("          Predicted")
        print("          Contact  Whiff")
        print(f"Actual Contact  {cm_standard[0,0]:6d}  {cm_standard[0,1]:6d}")
        print(f"      Whiff     {cm_standard[1,0]:6d}  {cm_standard[1,1]:6d}")
        
        print(f"\nThreshold Confusion Matrix (≥{whiff_threshold*100:.0f}%):")
        cm_threshold = confusion_matrix(balanced_y_true, y_pred_threshold)
        print("          Predicted")
        print("          Contact  Whiff")
        print("          Contact  Whiff")
        print(f"Actual Contact  {cm_threshold[0,0]:6d}  {cm_threshold[0,1]:6d}")
        print(f"      Whiff     {cm_threshold[1,0]:6d}  {cm_threshold[1,1]:6d}")
        
        # Test different threshold values
        threshold_results = test_threshold_values(balanced_y_true, y_proba)
        
        # Detailed analysis with threshold predictions
        analyze_whiff_predictions(balanced_swing_df, balanced_y_true, y_pred_threshold, y_proba)
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    print("\n" + "="*60)
    print("WHIFF VS CONTACT MODEL TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 