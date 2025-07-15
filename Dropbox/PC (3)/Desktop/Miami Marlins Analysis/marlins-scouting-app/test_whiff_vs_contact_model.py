import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

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
    print("CONFIDENCE DIFFERENCE THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # Test confidence difference thresholds
    confidence_diff_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    whiff_threshold = 0.35
    contact_threshold = 0.75
    results = []
    
    print(f"{'Conf Diff Thr':<12} {'Whiff Pred':<10} {'Whiff Prec':<10} {'Whiff Recall':<12} {'Balanced Acc':<12} {'Uncertain':<10}")
    print("-" * 75)
    
    for conf_diff_thresh in confidence_diff_thresholds:
        # Calculate probability differences
        contact_proba = y_proba[:, 0]
        whiff_proba = y_proba[:, 1]
        proba_diff = contact_proba - whiff_proba
        
        # Apply confidence difference threshold logic
        y_pred = np.where(
            whiff_proba >= whiff_threshold, 1,  # High whiff prob -> predict whiff
            np.where(
                contact_proba >= contact_threshold, 0,  # High contact prob -> predict contact
                np.where(
                    proba_diff >= conf_diff_thresh, 0,  # Sufficient confidence difference -> predict contact
                    1  # Default to whiff for uncertain cases
                )
            )
        )
        
        # Calculate uncertain cases (defaulted to whiff)
        uncertain_cases = ((whiff_proba < whiff_threshold) & 
                          (contact_proba < contact_threshold) & 
                          (proba_diff < conf_diff_thresh)).sum()
        
        # Calculate metrics for imbalanced data
        whiff_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        whiff_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        whiff_predictions = y_pred.sum()
        
        print(f"{conf_diff_thresh:<12.2f} {whiff_predictions:<10d} {whiff_precision:<10.3f} {whiff_recall:<12.3f} {balanced_acc:<12.3f} {uncertain_cases:<10d}")
        
        results.append({
            'confidence_diff_threshold': conf_diff_thresh,
            'whiff_pred': whiff_predictions,
            'whiff_precision': whiff_precision,
            'whiff_recall': whiff_recall,
            'balanced_acc': balanced_acc,
            'uncertain_cases': uncertain_cases
        })
    
    # Find best threshold combination
    best_result = max(results, key=lambda x: x['balanced_acc'])
    print(f"\nBest confidence difference threshold:")
    print(f"  Confidence difference threshold: {best_result['confidence_diff_threshold']:.2f}")
    print(f"  Balanced accuracy: {best_result['balanced_acc']:.3f}")
    print(f"  Whiff precision: {best_result['whiff_precision']:.3f}")
    print(f"  Whiff recall: {best_result['whiff_recall']:.3f}")
    print(f"  Uncertain cases defaulted to whiff: {best_result['uncertain_cases']}")
    
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
    
    # DETAILED ANALYSIS OF MISCLASSIFIED WHIFFS
    print(f"\n" + "="*80)
    print("DETAILED ANALYSIS OF WHIFFS MISCLASSIFIED AS CONTACTS")
    print("="*80)
    
    # Find whiffs misclassified as contacts
    whiff_misclassified_as_contact = (y_true == 1) & (y_pred == 0)
    misclassified_whiffs = df[whiff_misclassified_as_contact].copy()
    
    if len(misclassified_whiffs) > 0:
        print(f"\nFound {len(misclassified_whiffs)} whiffs misclassified as contacts")
        print(f"This represents {len(misclassified_whiffs)/whiff_count*100:.1f}% of all actual whiffs")
        
        # Add prediction probabilities to the misclassified data
        misclassified_indices = np.where(whiff_misclassified_as_contact)[0]
        misclassified_whiffs['whiff_probability'] = [probabilities[idx][1] for idx in misclassified_indices]
        misclassified_whiffs['contact_probability'] = [probabilities[idx][0] for idx in misclassified_indices]
        misclassified_whiffs['confidence'] = [max(probabilities[idx]) for idx in misclassified_indices]
        
        # Basic statistics
        print(f"\nBASIC STATISTICS:")
        print(f"  Average whiff probability: {misclassified_whiffs['whiff_probability'].mean():.3f}")
        print(f"  Average contact probability: {misclassified_whiffs['contact_probability'].mean():.3f}")
        print(f"  Average confidence: {misclassified_whiffs['confidence'].mean():.3f}")
        print(f"  Median whiff probability: {misclassified_whiffs['whiff_probability'].median():.3f}")
        
        # Pitch type analysis
        print(f"\nPITCH TYPE ANALYSIS:")
        pitch_type_counts = misclassified_whiffs['pitch_type'].value_counts()
        for pitch_type, count in pitch_type_counts.head(10).items():
            avg_whiff_prob = misclassified_whiffs[misclassified_whiffs['pitch_type'] == pitch_type]['whiff_probability'].mean()
            print(f"  {pitch_type}: {count} misclassifications (avg whiff prob: {avg_whiff_prob:.3f})")
        
        # Zone analysis
        print(f"\nZONE ANALYSIS:")
        zone_counts = misclassified_whiffs['zone'].value_counts()
        for zone, count in zone_counts.head(10).items():
            avg_whiff_prob = misclassified_whiffs[misclassified_whiffs['zone'] == zone]['whiff_probability'].mean()
            print(f"  Zone {zone}: {count} misclassifications (avg whiff prob: {avg_whiff_prob:.3f})")
        
        # Velocity analysis
        print(f"\nVELOCITY ANALYSIS:")
        low_vel = misclassified_whiffs[misclassified_whiffs['release_speed'] < 85]
        high_vel = misclassified_whiffs[misclassified_whiffs['release_speed'] > 95]
        mid_vel = misclassified_whiffs[(misclassified_whiffs['release_speed'] >= 85) & (misclassified_whiffs['release_speed'] <= 95)]
        
        print(f"  Low velocity (<85 mph): {len(low_vel)} misclassifications (avg whiff prob: {low_vel['whiff_probability'].mean():.3f})")
        print(f"  Mid velocity (85-95 mph): {len(mid_vel)} misclassifications (avg whiff prob: {mid_vel['whiff_probability'].mean():.3f})")
        print(f"  High velocity (>95 mph): {len(high_vel)} misclassifications (avg whiff prob: {high_vel['whiff_probability'].mean():.3f})")
        
        # Movement analysis
        print(f"\nMOVEMENT ANALYSIS:")
        low_movement = misclassified_whiffs[misclassified_whiffs['movement_magnitude'] < 2]
        high_movement = misclassified_whiffs[misclassified_whiffs['movement_magnitude'] > 4]
        mid_movement = misclassified_whiffs[(misclassified_whiffs['movement_magnitude'] >= 2) & (misclassified_whiffs['movement_magnitude'] <= 4)]
        
        print(f"  Low movement (<2): {len(low_movement)} misclassifications (avg whiff prob: {low_movement['whiff_probability'].mean():.3f})")
        print(f"  Mid movement (2-4): {len(mid_movement)} misclassifications (avg whiff prob: {mid_movement['whiff_probability'].mean():.3f})")
        print(f"  High movement (>4): {len(high_movement)} misclassifications (avg whiff prob: {high_movement['whiff_probability'].mean():.3f})")
        
        # Count analysis
        print(f"\nCOUNT ANALYSIS:")
        early_count = misclassified_whiffs[(misclassified_whiffs['balls'] <= 1) & (misclassified_whiffs['strikes'] <= 1)]
        pressure_count = misclassified_whiffs[(misclassified_whiffs['strikes'] >= 2) | (misclassified_whiffs['balls'] >= 3)]
        other_count = misclassified_whiffs[~((misclassified_whiffs['balls'] <= 1) & (misclassified_whiffs['strikes'] <= 1)) & ~((misclassified_whiffs['strikes'] >= 2) | (misclassified_whiffs['balls'] >= 3))]
        
        print(f"  Early count (≤1 ball, ≤1 strike): {len(early_count)} misclassifications (avg whiff prob: {early_count['whiff_probability'].mean():.3f})")
        print(f"  Pressure count (≥2 strikes or ≥3 balls): {len(pressure_count)} misclassifications (avg whiff prob: {pressure_count['whiff_probability'].mean():.3f})")
        print(f"  Other counts: {len(other_count)} misclassifications (avg whiff prob: {other_count['whiff_probability'].mean():.3f})")
        
        # Confidence analysis
        print(f"\nCONFIDENCE ANALYSIS:")
        low_conf = misclassified_whiffs[misclassified_whiffs['confidence'] < 0.6]
        med_conf = misclassified_whiffs[(misclassified_whiffs['confidence'] >= 0.6) & (misclassified_whiffs['confidence'] < 0.8)]
        high_conf = misclassified_whiffs[misclassified_whiffs['confidence'] >= 0.8]
        
        print(f"  Low confidence (<60%): {len(low_conf)} misclassifications (avg whiff prob: {low_conf['whiff_probability'].mean():.3f})")
        print(f"  Medium confidence (60-80%): {len(med_conf)} misclassifications (avg whiff prob: {med_conf['whiff_probability'].mean():.3f})")
        print(f"  High confidence (≥80%): {len(high_conf)} misclassifications (avg whiff prob: {high_conf['whiff_probability'].mean():.3f})")
        
        # Show examples of misclassified whiffs
        print(f"\nEXAMPLES OF MISCLASSIFIED WHIFFS:")
        print(f"{'Pitch Type':<8} {'Zone':<4} {'Velocity':<8} {'Movement':<8} {'Count':<6} {'Whiff Prob':<10} {'Contact Prob':<12}")
        print("-" * 80)
        
        for i in range(min(10, len(misclassified_whiffs))):
            row = misclassified_whiffs.iloc[i]
            print(f"{row['pitch_type']:<8} {row['zone']:<4.0f} {row['release_speed']:<8.1f} {row['movement_magnitude']:<8.2f} {row['balls']:.0f}-{row['strikes']:<3.0f} {row['whiff_probability']:<10.3f} {row['contact_probability']:<12.3f}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS TO IMPROVE WHIFF DETECTION:")
        print("-" * 60)
        
        # Find patterns that could help
        high_whiff_prob_misclassified = misclassified_whiffs[misclassified_whiffs['whiff_probability'] > 0.4]
        if len(high_whiff_prob_misclassified) > 0:
            print(f"  {len(high_whiff_prob_misclassified)} whiffs had >40% whiff probability but were misclassified")
            print(f"  Consider lowering whiff threshold to catch these")
        
        # Identify problematic pitch types
        problematic_pitches = misclassified_whiffs.groupby('pitch_type')['whiff_probability'].mean().sort_values(ascending=False)
        print(f"  Problematic pitch types (high whiff prob but misclassified):")
        for pitch_type, avg_prob in problematic_pitches.head(5).items():
            count = len(misclassified_whiffs[misclassified_whiffs['pitch_type'] == pitch_type])
            print(f"    {pitch_type}: {count} misclassifications, avg whiff prob: {avg_prob:.3f}")
        
        # Identify problematic zones
        problematic_zones = misclassified_whiffs.groupby('zone')['whiff_probability'].mean().sort_values(ascending=False)
        print(f"  Problematic zones (high whiff prob but misclassified):")
        for zone, avg_prob in problematic_zones.head(5).items():
            count = len(misclassified_whiffs[misclassified_whiffs['zone'] == zone])
            print(f"    Zone {zone}: {count} misclassifications, avg whiff prob: {avg_prob:.3f}")
        
        print(f"\n  Consider adding features specific to these problematic areas")
        print(f"  Consider adjusting thresholds based on pitch type and zone")
        
    else:
        print(f"\nNo whiffs were misclassified as contacts!")

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
        
        # Apply confidence difference threshold approach
        whiff_threshold = 0.35  # Lower threshold for whiffs (minority class)
        contact_threshold = 0.75  # Higher threshold for contacts (majority class)
        confidence_diff_threshold = 0.15  # Minimum difference required to predict contact over whiff
        
        # Calculate probability differences
        contact_proba = y_proba[:, 0]
        whiff_proba = y_proba[:, 1]
        proba_diff = contact_proba - whiff_proba
        
        # Apply confidence difference threshold logic:
        # 1. If whiff prob >= 35%, predict whiff
        # 2. If contact prob >= 75%, predict contact
        # 3. If neither threshold met, require minimum confidence difference to predict contact
        # 4. Otherwise, predict whiff (safer default for uncertain cases)
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
        
        print(f"\nPrediction Results (BALANCED DATASET):")
        print(f"Total swings: {len(balanced_swing_df)}")
        print(f"Actual whiffs: {balanced_y_true.sum()} (50.0%)")
        print(f"Actual contacts: {len(balanced_swing_df) - balanced_y_true.sum()} (50.0%)")
        print(f"Standard predicted whiffs: {y_pred_standard.sum()}")
        print(f"Confidence Difference predicted whiffs: {y_pred_threshold.sum()}")
        
        # Calculate how many uncertain cases defaulted to whiff
        uncertain_cases = ((whiff_proba < whiff_threshold) & 
                          (contact_proba < contact_threshold) & 
                          (proba_diff < confidence_diff_threshold)).sum()
        
        print(f"Uncertain cases defaulted to whiff: {uncertain_cases}")
        print(f"Confidence difference threshold: {confidence_diff_threshold*100:.0f}%")
        
        print(f"\nBalanced Dataset Analysis:")
        print(f"  Equal representation of contacts and whiffs")
        print(f"  Overall accuracy now reflects true model performance!")
        print(f"  Confidence difference approach prioritizes whiff detection for uncertain cases")
        
        # Evaluate both approaches
        accuracy_standard = accuracy_score(balanced_y_true, y_pred_standard)
        accuracy_threshold = accuracy_score(balanced_y_true, y_pred_threshold)
        
        print(f"\nStandard Accuracy: {accuracy_standard:.3f} ({accuracy_standard*100:.1f}%)")
        print(f"Threshold Accuracy: {accuracy_threshold:.3f} ({accuracy_threshold*100:.1f}%)")
        
        print("\nStandard Classification Report:")
        print(classification_report(balanced_y_true, y_pred_standard, target_names=['Contact', 'Whiff']))
        
        print(f"\nConfidence Difference Classification Report:")
        print(classification_report(balanced_y_true, y_pred_threshold, target_names=['Contact', 'Whiff']))
        
        print("\nStandard Confusion Matrix:")
        cm_standard = confusion_matrix(balanced_y_true, y_pred_standard)
        print("          Predicted")
        print("          Contact  Whiff")
        print(f"Actual Contact  {cm_standard[0,0]:6d}  {cm_standard[0,1]:6d}")
        print(f"      Whiff     {cm_standard[1,0]:6d}  {cm_standard[1,1]:6d}")
        
        print(f"\nConfidence Difference Confusion Matrix:")
        cm_threshold = confusion_matrix(balanced_y_true, y_pred_threshold)
        print("          Predicted")
        print("          Contact  Whiff")
        print(f"Actual Contact  {cm_threshold[0,0]:6d}  {cm_threshold[0,1]:6d}")
        print(f"      Whiff     {cm_threshold[1,0]:6d}  {cm_threshold[1,1]:6d}")
        
        # Test different threshold values
        threshold_results = test_threshold_values(balanced_y_true, y_proba)
        
        # Detailed analysis with threshold predictions
        analyze_whiff_predictions(balanced_swing_df, balanced_y_true, y_pred_threshold, y_proba)
        
        # ============================================================================
        # COMPREHENSIVE ANALYSIS FOR WHITE PAPER
        # ============================================================================
        
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS FOR WHITE PAPER")
        print("="*60)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(balanced_y_true, y_pred_threshold)
        precision = precision_score(balanced_y_true, y_pred_threshold)
        recall = recall_score(balanced_y_true, y_pred_threshold)
        f1 = f1_score(balanced_y_true, y_pred_threshold)
        
        print(f"\nModel Performance Summary:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        
        # 1. CONFUSION MATRIX
        print("\n1. Creating Confusion Matrix...")
        cm = confusion_matrix(balanced_y_true, y_pred_threshold)
        print(f"Confusion Matrix Raw Values:")
        print(f"True Negatives (Contact → Contact): {cm[0,0]}")
        print(f"False Positives (Contact → Whiff): {cm[0,1]}")
        print(f"False Negatives (Whiff → Contact): {cm[1,0]}")
        print(f"True Positives (Whiff → Whiff): {cm[1,1]}")
        
        # Debug: Check if any values are zero
        print(f"\nConfusion Matrix Debug:")
        print(f"Matrix shape: {cm.shape}")
        print(f"All values: {cm.flatten()}")
        print(f"Any zero values: {np.any(cm == 0)}")
        print(f"Sum of all values: {np.sum(cm)}")
        print(f"Expected sum: {len(balanced_y_true)}")
        
        plt.figure(figsize=(10, 8))
        
        # Use a more robust colormap and ensure annotations are visible
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=['Contact', 'Whiff'], 
                    yticklabels=['Contact', 'Whiff'],
                    cbar_kws={'label': 'Count'})
        
        # Add explicit text annotations to ensure visibility
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                plt.text(j + 0.5, i + 0.5, str(value), 
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='black' if value > 0 else 'red')
        
        plt.title('Whiff vs Contact Classifier Confusion Matrix\nRonald Acuña Jr. Holdout Dataset', 
                  fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('whiff_contact_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. ROC CURVE
        print("\n2. Creating ROC Curve...")
        fpr, tpr, _ = roc_curve(balanced_y_true, y_proba[:, 1])  # Use whiff probability
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve\nWhiff vs Contact Classifier', 
                  fontsize=16, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('whiff_contact_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. PITCH TYPE ACCURACY BAR CHART
        print("\n3. Creating Pitch Type Accuracy Bar Chart...")
        
        # Calculate accuracy by pitch type
        pitch_type_accuracy = {}
        pitch_type_counts = {}
        
        for pitch_type in balanced_swing_df['pitch_type'].unique():
            if pd.isna(pitch_type):
                continue
            mask = balanced_swing_df['pitch_type'] == pitch_type
            if mask.sum() >= 5:  # Only include pitch types with at least 5 samples
                y_true_subset = balanced_y_true[mask]
                y_pred_subset = y_pred_threshold[mask]
                accuracy_subset = accuracy_score(y_true_subset, y_pred_subset)
                pitch_type_accuracy[pitch_type] = accuracy_subset
                pitch_type_counts[pitch_type] = mask.sum()
        
        # Sort by accuracy
        sorted_pitch_types = sorted(pitch_type_accuracy.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_pitch_types:
            pitch_types = [pt[0] for pt in sorted_pitch_types]
            accuracies = [pt[1] for pt in sorted_pitch_types]
            counts = [pitch_type_counts[pt] for pt in pitch_types]
            
            plt.figure(figsize=(12, 8))
            bars = plt.bar(range(len(pitch_types)), accuracies)
            plt.xlabel('Pitch Type', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Whiff vs Contact Classification Accuracy by Pitch Type\nRonald Acuña Jr. Holdout Dataset', 
                      fontsize=16, fontweight='bold')
            plt.xticks(range(len(pitch_types)), pitch_types)
            plt.ylim(0, 1)
            
            # Add count labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'n={count}', ha='center', va='bottom', fontsize=10)
            
            # Color bars by accuracy
            colors = plt.cm.viridis(accuracies)
            for i, (bar, color) in enumerate(zip(bars, colors)):
                bar.set_color(color)
            
            plt.tight_layout()
            plt.savefig('whiff_contact_pitch_type_accuracy.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print pitch type accuracy table
            print("\nPitch Type Accuracy Table:")
            print("=" * 50)
            print(f"{'Pitch Type':<10} {'Accuracy':<10} {'Count':<8}")
            print("-" * 50)
            for pitch_type, accuracy in sorted_pitch_types:
                count = pitch_type_counts[pitch_type]
                print(f"{pitch_type:<10} {accuracy:<10.3f} {count:<8}")
            print("=" * 50)
        else:
            print("No pitch types with sufficient samples for accuracy analysis.")
        
        # 4. DETAILED CLASSIFICATION REPORT
        print("\n4. Detailed Classification Report:")
        print("=" * 50)
        print(classification_report(balanced_y_true, y_pred_threshold, target_names=['Contact', 'Whiff']))
        
        # 5. SAVE COMPREHENSIVE RESULTS
        print("\n5. Saving Comprehensive Results...")
        results_summary = {
            'model_performance': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'dataset_info': {
                'total_swings': len(balanced_swing_df),
                'contact_count': (balanced_y_true == 0).sum(),
                'whiff_count': (balanced_y_true == 1).sum(),
                'contact_rate': (balanced_y_true == 0).sum() / len(balanced_y_true),
                'whiff_rate': (balanced_y_true == 1).sum() / len(balanced_y_true)
            },
            'pitch_type_accuracy': dict(sorted_pitch_types) if sorted_pitch_types else {}
        }
        
        # Save results to JSON
        import json
        with open('whiff_contact_analysis_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files for white paper:")
        print("- whiff_contact_confusion_matrix.png")
        print("- whiff_contact_roc_curve.png")
        print("- whiff_contact_pitch_type_accuracy.png")
        print("- whiff_contact_analysis_results.json")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    print("\n" + "="*60)
    print("WHIFF VS CONTACT MODEL TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

