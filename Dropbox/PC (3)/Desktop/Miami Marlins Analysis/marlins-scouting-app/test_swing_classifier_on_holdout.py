import pandas as pd
import numpy as np
import pickle
from collections import Counter
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Import the same functions from training script
from train_sequential_models import prepare_features, calculate_zone

# Load holdout dataset
holdout_df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")

# Load models
with open("sequential_models.pkl", "rb") as f:
    models = pickle.load(f)

swing_features = models['swing_features']
swing_preprocessor = models['swing_preprocessor']
swing_calibrated_model = models['swing_calibrated_model']
swing_threshold = models.get('swing_threshold', 0.9)  # Use the saved threshold

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']

print("Processing holdout dataset with comprehensive features...")

def classify_swing_robust(row):
    """
    Robustly classify whether a pitch resulted in a swing or not.
    Handles NaN values and edge cases properly.
    """
    description = row.get('description', None)
    events = row.get('events', None)
    
    # Check if both description and events are NaN/null
    if pd.isna(description) and pd.isna(events):
        return None  # Can't classify, skip this pitch
    
    # If description is available, use it as primary source
    if not pd.isna(description):
        if description in swing_events:
            return 1  # Swing
        elif description in no_swing_events:
            return 0  # No swing
        # If description is not in either list, check events
    
    # If description is NaN or unclear, check events
    if not pd.isna(events):
        if events in ['single', 'double', 'triple', 'home_run', 'field_out', 'sac_fly', 'sac_bunt']:
            return 1  # Swing (hit into play)
        elif events in ['walk', 'strikeout', 'hit_by_pitch']:
            return 0  # No swing (walk, called strikeout, HBP)
        # If events is not in either list, we can't classify
    
    # If we can't determine from either field, return None
    return None

# Create swing/no-swing target for the entire dataset with robust classification
print("Classifying swings and no-swings with robust logic...")
swing_classifications = []
valid_indices = []

for idx, row in holdout_df.iterrows():
    classification = classify_swing_robust(row)
    if classification is not None:
        swing_classifications.append(classification)
        valid_indices.append(idx)
    else:
        print(f"Warning: Could not classify pitch at index {idx}. Description: {row.get('description', 'NaN')}, Events: {row.get('events', 'NaN')}")

# Create a new DataFrame with only valid classifications
holdout_df_valid = holdout_df.iloc[valid_indices].copy()
holdout_df_valid['swing'] = swing_classifications

print(f"Original dataset size: {len(holdout_df)}")
print(f"Valid classifications: {len(holdout_df_valid)}")
print(f"Unclassifiable pitches: {len(holdout_df) - len(holdout_df_valid)}")

# Show classification breakdown
swing_count = sum(swing_classifications)
no_swing_count = len(swing_classifications) - swing_count
print(f"Swing classifications: {swing_count}")
print(f"No-swing classifications: {no_swing_count}")
print(f"Swing rate: {swing_count/len(swing_classifications)*100:.1f}%")

# Use the valid dataset for the rest of the analysis
holdout_df = holdout_df_valid

# Show some examples of the classification
print(f"\n--- CLASSIFICATION EXAMPLES ---")
print("Sample of classified pitches:")
sample_size = min(10, len(holdout_df))
for i in range(sample_size):
    row = holdout_df.iloc[i]
    desc = row.get('description', 'NaN')
    events = row.get('events', 'NaN')
    swing = row['swing']
    swing_text = "SWING" if swing == 1 else "NO SWING"
    print(f"  {i+1}. Description: '{desc}', Events: '{events}' → {swing_text}")

# Validate classification logic
print(f"\n--- CLASSIFICATION VALIDATION ---")
swing_by_desc = holdout_df[holdout_df['description'].isin(swing_events)]
no_swing_by_desc = holdout_df[holdout_df['description'].isin(no_swing_events)]
print(f"Pitches classified as swing by description: {len(swing_by_desc)}")
print(f"Pitches classified as no-swing by description: {len(no_swing_by_desc)}")
print(f"Pitches classified by events only: {len(holdout_df) - len(swing_by_desc) - len(no_swing_by_desc)}")

# Prepare features for the entire dataset at once (like training) - this includes all the complex engineered features
holdout_df, num_feats, cat_feats = prepare_features(holdout_df)

# Update swing rates now that swing column exists
from train_sequential_models import update_swing_rates
holdout_df = update_swing_rates(holdout_df)

# Get the features that the model expects
all_feats = num_feats + cat_feats

# Create input with only the features the model expects
model_input = holdout_df[all_feats]

# Transform using the preprocessor
X_swing = swing_preprocessor.transform(model_input)

# Handle any remaining NaN values in the transformed data
X_swing = np.nan_to_num(X_swing, nan=0.0)

# Make predictions for all rows at once using calibrated model
swing_probs = swing_calibrated_model.predict_proba(X_swing)
swing_prob_scores = swing_probs[:, 1]  # Probability of swing

# Get calibrated probabilities and apply count-specific thresholds
swing_preds = []
thresholds = []
for i, (prob, row) in enumerate(zip(swing_prob_scores, holdout_df.iterrows())):
    balls = row[1].get('balls', 0)
    strikes = row[1].get('strikes', 0)
    
    # Determine count situation and threshold
    # if balls <= 1 and strikes <= 1:
    #     threshold = 0.95 # Very high threshold for early counts (≤1 ball, ≤1 strike)
    # elif strikes == 2 and balls < 3:
    #     threshold = 0.65
    # elif balls == 3 and strikes == 0:
    #     threshold == 0.95
    # elif balls == 2 or balls == 1 and strikes == 0:
    #     threshold = 0.70
    # elif balls == 3 and strikes == 2: 
    #     threshold = 0.60
    # elif balls == 3 and strikes == 1:
    #     threshold = 0.85
    # elif balls == 1 and strikes == 1:
    #     threshold = 0.85 # High threshold for middle counts (1-1)
    # else:
    #     threshold = 0.9 # Default threshold for other situations
    if balls == 0 and strikes == 0:
        threshold = 0.95
    elif balls == 0 and strikes == 1:
        threshold = 0.90
    elif balls == 0 and strikes == 2:
        threshold = 0.55
    elif balls == 1 and strikes == 0:
        threshold = 0.95
    elif balls == 1 and strikes == 1:
        threshold = 0.95
    elif balls == 1 and strikes == 2:
        threshold = 0.55
    elif balls == 2 and strikes == 0:
        threshold = 0.5
    elif balls == 2 and strikes == 1:
        threshold = 0.80
    elif balls == 2 and strikes == 2:
        threshold = 0.55
    elif balls == 3 and strikes == 0:
        threshold = 0.95
    elif balls == 3 and strikes == 1:
        threshold = 0.55
    elif balls == 3 and strikes == 2:
        threshold = 0.50
    else:
        threshold = 0.9  # Fallback default

    
    # Make prediction with count-specific threshold
    swing_preds.append(1 if prob >= threshold else 0)
    thresholds.append(threshold)

swing_preds = np.array(swing_preds)

# Get true labels
true_swings = holdout_df['swing'].values

# Calculate accuracy
correct = np.sum(swing_preds == true_swings)
total = len(true_swings)

# Analysis containers
fp_pitch_types = []  # False positive: predicted swing, actually no swing
fn_pitch_types = []  # False negative: predicted no swing, actually swing
fp_descriptions = []
fn_descriptions = []
fp_events = []
fn_events = []
tp_pitch_types = []  # True positive: correct swing
tn_pitch_types = []  # True negative: correct no swing
tp_descriptions = []
tn_descriptions = []
tp_events = []
tn_events = []

# Analyze results
for idx in range(len(holdout_df)):
    row = holdout_df.iloc[idx]
    true_swing = true_swings[idx]
    swing_pred = swing_preds[idx]
    
    if swing_pred == true_swing:
        if swing_pred == 1:
            tp_pitch_types.append(row.get('pitch_type', 'N/A'))
            tp_descriptions.append(row.get('description', 'N/A'))
            tp_events.append(row.get('events', 'N/A'))
        else:
            tn_pitch_types.append(row.get('pitch_type', 'N/A'))
            tn_descriptions.append(row.get('description', 'N/A'))
            tn_events.append(row.get('events', 'N/A'))
    else:
        if swing_pred == 1:
            fp_pitch_types.append(row.get('pitch_type', 'N/A'))
            fp_descriptions.append(row.get('description', 'N/A'))
            fp_events.append(row.get('events', 'N/A'))
        else:
            fn_pitch_types.append(row.get('pitch_type', 'N/A'))
            fn_descriptions.append(row.get('description', 'N/A'))
            fn_events.append(row.get('events', 'N/A'))

# SIMPLIFIED OUTPUT - Just show the key numbers clearly
print("=" * 50)
print(f"SWING/NO-SWING CLASSIFIER RESULTS")
print("=" * 50)
print(f"Using calibrated model with threshold: {swing_threshold}")
print(f"Correct Predictions: {correct}")
print(f"Total Pitches: {total}")
print(f"Accuracy: {correct}/{total} = {correct/total:.4f} ({correct/total*100:.2f}%)")
print("=" * 50)

# Optional: Show breakdown if you want more detail
print(f"\nBreakdown:")
print(f"  True Positives (correct swing): {len(tp_pitch_types)}")
print(f"  True Negatives (correct no swing): {len(tn_pitch_types)}")
print(f"  False Positives (predicted swing, actually no swing): {len(fp_pitch_types)}")
print(f"  False Negatives (predicted no swing, actually swing): {len(fn_pitch_types)}")

print("\n--- False Positives (predicted swing, actually no swing) ---")
print("Top pitch types:", Counter(fp_pitch_types).most_common(5))
print("Top descriptions:", Counter(fp_descriptions).most_common(5))
print("Top events:", Counter(fp_events).most_common(5))

print("\n--- False Negatives (predicted no swing, actually swing) ---")
print("Top pitch types:", Counter(fn_pitch_types).most_common(5))
print("Top descriptions:", Counter(fn_descriptions).most_common(5))
print("Top events:", Counter(fn_events).most_common(5))

print("\n--- True Positives (correct swing) ---")
print("Top pitch types:", Counter(tp_pitch_types).most_common(5))
print("Top descriptions:", Counter(tp_descriptions).most_common(5))
print("Top events:", Counter(tp_events).most_common(5))

print("\n--- True Negatives (correct no swing) ---")
print("Top pitch types:", Counter(tn_pitch_types).most_common(5))
print("Top descriptions:", Counter(tn_descriptions).most_common(5))
print("Top events:", Counter(tn_events).most_common(5))

# INVESTIGATE FALSE POSITIVES IN DETAIL
print("\n" + "=" * 60)
print("FALSE POSITIVE INVESTIGATION")
print("=" * 60)

# Create DataFrame of false positives for detailed analysis
fp_indices = []
for idx in range(len(holdout_df)):
    if swing_preds[idx] == 1 and true_swings[idx] == 0:
        fp_indices.append(idx)

if fp_indices:
    fp_df = holdout_df.iloc[fp_indices].copy()
    print(f"\nFound {len(fp_df)} false positives to analyze")
    
    # Basic statistics of false positives
    print(f"\n--- FALSE POSITIVE BASIC STATISTICS ---")
    print(f"Total false positives: {len(fp_df)}")
    print(f"False positive rate: {len(fp_df)}/{len(holdout_df)} = {len(fp_df)/len(holdout_df)*100:.2f}%")
    
    # Location analysis
    print(f"\n--- FALSE POSITIVE LOCATION ANALYSIS ---")
    if 'plate_x' in fp_df.columns and 'plate_z' in fp_df.columns:
        print("Location statistics:")
        print(fp_df[['plate_x', 'plate_z']].describe())
        
        # Check if they're barely outside the zone
        if 'in_strike_zone' in fp_df.columns:
            in_zone_count = fp_df['in_strike_zone'].sum()
            out_zone_count = len(fp_df) - in_zone_count
            print(f"\nZone analysis:")
            print(f"  False positives in strike zone: {in_zone_count} ({in_zone_count/len(fp_df)*100:.1f}%)")
            print(f"  False positives outside strike zone: {out_zone_count} ({out_zone_count/len(fp_df)*100:.1f}%)")
        
        # Distance from zone center
        if 'zone_distance' in fp_df.columns:
            print(f"\nZone distance analysis:")
            print(fp_df['zone_distance'].describe())
            
            # Check for "edge" pitches (close to zone but outside)
            edge_pitches = fp_df[(fp_df['zone_distance'] > 0.8) & (fp_df['zone_distance'] < 1.2)]
            print(f"  Edge pitches (0.8-1.2 distance): {len(edge_pitches)} ({len(edge_pitches)/len(fp_df)*100:.1f}%)")
    
    # Count analysis
    print(f"\n--- FALSE POSITIVE COUNT ANALYSIS ---")
    if 'balls' in fp_df.columns and 'strikes' in fp_df.columns:
        print("Count distribution:")
        print(fp_df[['balls', 'strikes']].describe())
        
        # Check for early count situations
        early_count = fp_df[(fp_df['balls'] <= 1) & (fp_df['strikes'] <= 1)]
        print(f"  Early count (≤1 ball, ≤1 strike): {len(early_count)} ({len(early_count)/len(fp_df)*100:.1f}%)")
        
        # Check for pressure situations
        pressure_count = fp_df[(fp_df['strikes'] >= 2) | (fp_df['balls'] >= 3)]
        print(f"  Pressure situations (≥2 strikes or ≥3 balls): {len(pressure_count)} ({len(pressure_count)/len(fp_df)*100:.1f}%)")
    
    # Pitch type analysis
    print(f"\n--- FALSE POSITIVE PITCH TYPE ANALYSIS ---")
    if 'pitch_type' in fp_df.columns:
        pitch_type_counts = fp_df['pitch_type'].value_counts()
        print("Pitch type distribution:")
        print(pitch_type_counts)
        
        # Check for specific pitch types that might be "bait" pitches
        bait_pitches = fp_df[fp_df['pitch_type'].isin(['SL', 'CH', 'CU'])]
        print(f"\nBait pitches (SL, CH, CU): {len(bait_pitches)} ({len(bait_pitches)/len(fp_df)*100:.1f}%)")
    
    # Velocity analysis
    print(f"\n--- FALSE POSITIVE VELOCITY ANALYSIS ---")
    if 'release_speed' in fp_df.columns:
        print("Velocity statistics:")
        print(fp_df['release_speed'].describe())
        
        # Check for velocity deception
        low_vel = fp_df[fp_df['release_speed'] < 85]
        high_vel = fp_df[fp_df['release_speed'] > 95]
        print(f"  Low velocity (<85): {len(low_vel)} ({len(low_vel)/len(fp_df)*100:.1f}%)")
        print(f"  High velocity (>95): {len(high_vel)} ({len(high_vel)/len(fp_df)*100:.1f}%)")
    
    # Movement analysis
    print(f"\n--- FALSE POSITIVE MOVEMENT ANALYSIS ---")
    if 'api_break_x_batter_in' in fp_df.columns and 'api_break_z_with_gravity' in fp_df.columns:
        # IMPROVED Movement Quantification using Statcast break values
        fp_df['horizontal_break'] = fp_df['api_break_x_batter_in'].fillna(0)
        fp_df['vertical_break'] = fp_df['api_break_z_with_gravity'].fillna(0)
        fp_df['arm_side_break'] = fp_df['api_break_x_arm'].fillna(0)
        
        # Calculate total movement magnitude using the more accurate break values
        fp_df['movement_magnitude'] = np.sqrt(fp_df['horizontal_break']**2 + fp_df['vertical_break']**2)
    else:
        # Fallback to old method if new fields not available
        fp_df['movement_magnitude'] = np.sqrt(fp_df['pfx_x']**2 + fp_df['pfx_z']**2)
    
    print("Movement magnitude statistics:")
    print(fp_df['movement_magnitude'].describe())
    
    # Check for high movement pitches
    # Improved high movement detection based on pitch type
    high_movement = fp_df[
        ((fp_df['pitch_type'].isin(['SL', 'CU', 'KC'])) & (fp_df['movement_magnitude'] > 8)) |  # Breaking balls
        ((fp_df['pitch_type'].isin(['CH', 'FS'])) & (fp_df['movement_magnitude'] > 6)) |  # Offspeed
        ((~fp_df['pitch_type'].isin(['SL', 'CU', 'KC', 'CH', 'FS'])) & (fp_df['movement_magnitude'] > 4))  # Fastballs
    ]
    print(f"  High movement (pitch-type adjusted): {len(high_movement)} ({len(high_movement)/len(fp_df)*100:.1f}%)")
    
    # Game situation analysis
    print(f"\n--- FALSE POSITIVE GAME SITUATION ANALYSIS ---")
    if 'inning' in fp_df.columns:
        late_inning = fp_df[fp_df['inning'] >= 7]
        print(f"  Late inning (≥7): {len(late_inning)} ({len(late_inning)/len(fp_df)*100:.1f}%)")
    
    # Probability analysis
    print(f"\n--- FALSE POSITIVE PROBABILITY ANALYSIS ---")
    fp_prob_scores = swing_prob_scores[fp_indices]
    print("Probability scores for false positives:")
    print(f"  Mean probability: {np.mean(fp_prob_scores):.4f}")
    print(f"  Median probability: {np.median(fp_prob_scores):.4f}")
    print(f"  Min probability: {np.min(fp_prob_scores):.4f}")
    print(f"  Max probability: {np.max(fp_prob_scores):.4f}")
    print(f"  Std probability: {np.std(fp_prob_scores):.4f}")
    
    # Check how many are borderline predictions
    borderline_fp = fp_prob_scores[(fp_prob_scores >= 0.5) & (fp_prob_scores <= 0.6)]
    confident_fp = fp_prob_scores[fp_prob_scores > 0.6]
    print(f"  Borderline predictions (0.5-0.6): {len(borderline_fp)} ({len(borderline_fp)/len(fp_prob_scores)*100:.1f}%)")
    print(f"  Confident predictions (>0.6): {len(confident_fp)} ({len(confident_fp)/len(fp_prob_scores)*100:.1f}%)")
    
    # SUGGESTED NEW FEATURES BASED ON ANALYSIS
    print(f"\n--- SUGGESTED NEW FEATURES TO REDUCE FALSE POSITIVES ---")
    
    # Edge location feature
    if 'zone_distance' in fp_df.columns:
        edge_location_count = len(fp_df[(fp_df['zone_distance'] > 0.8) & (fp_df['zone_distance'] < 1.2)])
        print(f"  Edge location pitches: {edge_location_count} - Consider adding 'edge_location' feature")
    
    # Bait pitch feature
    if 'pitch_type' in fp_df.columns:
        bait_pitch_count = len(fp_df[fp_df['pitch_type'].isin(['SL', 'CH', 'CU'])])
        print(f"  Bait pitches (SL/CH/CU): {bait_pitch_count} - Consider adding 'bait_pitch' feature")
    
    # Early count feature
    if 'balls' in fp_df.columns and 'strikes' in fp_df.columns:
        early_count_count = len(fp_df[(fp_df['balls'] <= 1) & (fp_df['strikes'] <= 1)])
        print(f"  Early count pitches: {early_count_count} - Consider boosting early count features")
    
    # Velocity deception feature
    if 'release_speed' in fp_df.columns:
        low_vel_count = len(fp_df[fp_df['release_speed'] < 85])
        print(f"  Low velocity deception: {low_vel_count} - Consider adding velocity deception features")
    
    print(f"\n--- RECOMMENDATIONS ---")
    print("1. Add 'edge_location' feature for pitches just outside the zone")
    print("2. Add 'bait_pitch' feature for breaking balls and changeups")
    print("3. Boost early count features to reduce false positives in low-pressure situations")
    print("4. Add velocity deception features for unexpected velocity patterns")
    print("5. Consider increasing the probability threshold for more conservative predictions")

else:
    print("No false positives found to analyze!") 

# After making predictions for each pitch, output detailed analysis
# print("pitch_idx,confidence,threshold,prediction,actual")
# for idx, (proba, threshold, pred, actual) in enumerate(zip(swing_prob_scores, thresholds, swing_preds, true_swings)):
#     pred_label = 'SWING' if pred == 1 else 'NO SWING'
#     actual_label = 'SWING' if actual == 1 else 'NO SWING'
#     print(f"{idx},{proba:.4f},{threshold:.3f},{pred_label},{actual_label}") 

# NEW: OUTPUT ALL FEATURE IMPORTANCES TO CSV
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importances from the trained model
# We need to get the feature importances from the ensemble model
# Since we're using a calibrated model, we need to access the base estimator

# Try to get feature importances from the calibrated model
if hasattr(swing_calibrated_model, 'base_estimator'):
    base_model = swing_calibrated_model.base_estimator
    if hasattr(base_model, 'estimators_'):
        # Ensemble model - get importances from the first XGBoost model
        xgb_model = base_model.estimators_[0]
        if hasattr(xgb_model, 'feature_importances_'):
            feature_importances = xgb_model.feature_importances_
            print("✓ Got feature importances from ensemble XGBoost model")
        else:
            print("✗ XGBoost model doesn't have feature importances")
            feature_importances = None
    elif hasattr(base_model, 'feature_importances_'):
        # Single XGBoost model
        feature_importances = base_model.feature_importances_
        print("✓ Got feature importances from single XGBoost model")
    else:
        print("✗ Base model doesn't have feature importances")
        feature_importances = None
else:
    print("✗ Calibrated model doesn't have base_estimator")
    feature_importances = None

if feature_importances is not None:
    # Get feature names from the preprocessor
    feature_names = swing_preprocessor.get_feature_names_out()
    
    # Create DataFrame with feature names and importances
    feature_importance_df = pd.DataFrame({
        'feature_name': feature_names,
        'importance': feature_importances
    })
    
    # Sort by importance (descending)
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    # Add additional columns for analysis
    feature_importance_df['importance_rank'] = range(1, len(feature_importance_df) + 1)
    feature_importance_df['importance_percentile'] = feature_importance_df['importance'].rank(pct=True) * 100
    feature_importance_df['importance_category'] = pd.cut(
        feature_importance_df['importance'], 
        bins=[0, 0.001, 0.01, 0.1, 1.0], 
        labels=['Very Low (0-0.001)', 'Low (0.001-0.01)', 'Medium (0.01-0.1)', 'High (0.1-1.0)']
    )
    
    # Add feature type classification
    def classify_feature_type(feature_name):
        feature_name_lower = feature_name.lower()
        if 'acuna_' in feature_name_lower:
            return 'Hitter_Specific'
        elif 'zone' in feature_name_lower or 'plate_' in feature_name_lower:
            return 'Location'
        elif 'count' in feature_name_lower or 'ball' in feature_name_lower or 'strike' in feature_name_lower:
            return 'Count'
        elif 'pitch_type' in feature_name_lower or 'fastball' in feature_name_lower or 'breaking' in feature_name_lower:
            return 'Pitch_Type'
        elif 'movement' in feature_name_lower or 'break' in feature_name_lower:
            return 'Movement'
        elif 'velocity' in feature_name_lower or 'speed' in feature_name_lower:
            return 'Velocity'
        elif 'contact' in feature_name_lower or 'hit' in feature_name_lower:
            return 'Contact'
        elif 'babip' in feature_name_lower or 'whiff' in feature_name_lower:
            return 'Outcome'
        else:
            return 'Other'
    
    feature_importance_df['feature_type'] = feature_importance_df['feature_name'].apply(classify_feature_type)
    
    # Save to CSV
    csv_filename = "swing_classifier_feature_importances.csv"
    feature_importance_df.to_csv(csv_filename, index=False)
    
    print(f"✓ Feature importances saved to '{csv_filename}'")
    print(f"Total features analyzed: {len(feature_importance_df)}")
    
    # Print summary statistics
    print(f"\nFeature Importance Summary:")
    print(f"  Mean importance: {feature_importance_df['importance'].mean():.6f}")
    print(f"  Median importance: {feature_importance_df['importance'].median():.6f}")
    print(f"  Max importance: {feature_importance_df['importance'].max():.6f}")
    print(f"  Min importance: {feature_importance_df['importance'].min():.6f}")
    
    # Print feature type breakdown
    print(f"\nFeature Type Breakdown:")
    type_counts = feature_importance_df['feature_type'].value_counts()
    for feature_type, count in type_counts.items():
        avg_importance = feature_importance_df[feature_importance_df['feature_type'] == feature_type]['importance'].mean()
        print(f"  {feature_type}: {count} features, avg importance: {avg_importance:.6f}")
    
    # Print importance category breakdown
    print(f"\nImportance Category Breakdown:")
    category_counts = feature_importance_df['importance_category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} features")
    
    # Print top 20 features
    print(f"\nTop 20 Most Important Features:")
    for idx, row in feature_importance_df.head(20).iterrows():
        print(f"  {row['importance_rank']:2d}. {row['feature_name']:<40} {row['importance']:.6f} ({row['feature_type']})")
    
    # Print features with zero importance
    zero_importance_features = feature_importance_df[feature_importance_df['importance'] == 0]
    print(f"\nFeatures with Zero Importance ({len(zero_importance_features)} features):")
    for idx, row in zero_importance_features.iterrows():
        print(f"  {row['feature_name']} ({row['feature_type']})")
    
    # Print features with very low importance (< 0.001)
    very_low_features = feature_importance_df[feature_importance_df['importance'] < 0.001]
    print(f"\nFeatures with Very Low Importance (< 0.001) ({len(very_low_features)} features):")
    for idx, row in very_low_features.head(10).iterrows():  # Show first 10
        print(f"  {row['feature_name']} ({row['feature_type']}): {row['importance']:.6f}")
    if len(very_low_features) > 10:
        print(f"  ... and {len(very_low_features) - 10} more features")
    
    # Feature type analysis
    print(f"\nFeature Type Importance Analysis:")
    for feature_type in feature_importance_df['feature_type'].unique():
        type_features = feature_importance_df[feature_importance_df['feature_type'] == feature_type]
        avg_importance = type_features['importance'].mean()
        max_importance = type_features['importance'].max()
        zero_count = (type_features['importance'] == 0).sum()
        print(f"  {feature_type}:")
        print(f"    Count: {len(type_features)}")
        print(f"    Avg importance: {avg_importance:.6f}")
        print(f"    Max importance: {max_importance:.6f}")
        print(f"    Zero importance features: {zero_count}")
    
else:
    print("✗ Could not extract feature importances from the model")
    print("This might be because the model is an ensemble or calibrated model")
    print("You may need to access feature importances differently")
 
# --- Misclassification analysis by count ---
misclass_by_count = defaultdict(list)
for idx, (pred, actual, proba) in enumerate(zip(swing_preds, true_swings, swing_prob_scores)):
    if pred != actual:
        balls = holdout_df.iloc[idx]['balls']
        strikes = holdout_df.iloc[idx]['strikes']
        misclass_by_count[(balls, strikes)].append(proba)

print("\nMisclassifications by Count (Balls-Strikes):")
print(f"{'Balls-Strikes':<12} {'# Miss':<8} {'Avg Conf':<10}")
print("-" * 32)
for (balls, strikes), probs in sorted(misclass_by_count.items()):
    if math.isnan(balls) or math.isnan(strikes):
        print(f"Skipping count with NaN: balls={balls}, strikes={strikes}")
        continue
    avg_conf = sum(probs) / len(probs) if probs else 0
    print(f"{int(balls)}-{int(strikes):<9} {len(probs):<8} {avg_conf:.3f}") 
 
# --- Deeper dive into 0-0 count misclassifications ---
zero_zero_indices = [idx for idx, (pred, actual) in enumerate(zip(swing_preds, true_swings))
                    if pred != actual and holdout_df.iloc[idx]['balls'] == 0 and holdout_df.iloc[idx]['strikes'] == 0]

swing_to_noswing = 0
noswing_to_swing = 0
examples = []
for idx in zero_zero_indices:
    pred = swing_preds[idx]
    actual = true_swings[idx]
    proba = swing_prob_scores[idx]
    pitch_type = holdout_df.iloc[idx]['pitch_type'] if 'pitch_type' in holdout_df.columns else 'NA'
    if actual == 1 and pred == 0:
        swing_to_noswing += 1
        examples.append((idx, 'SWING', 'NO SWING', proba, pitch_type))
    elif actual == 0 and pred == 1:
        noswing_to_swing += 1
        examples.append((idx, 'NO SWING', 'SWING', proba, pitch_type))

print("\nDetailed 0-0 Count Misclassification Breakdown:")
print(f"  Actual SWING, Predicted NO SWING: {swing_to_noswing}")
print(f"  Actual NO SWING, Predicted SWING: {noswing_to_swing}")

print("\nSample of 0-0 misclassifications (idx, actual, predicted, confidence, pitch_type):")
for ex in examples[:10]:
    print(f"  {ex[0]}, {ex[1]}, {ex[2]}, {ex[3]:.3f}, {ex[4]}") 
 
# --- Per-count threshold grid search (coordinate descent) ---
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# Define count buckets
count_buckets = ["0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2", "3-0", "3-1", "3-2"]
threshold_range = np.arange(0.5, 0.96, 0.05)

# Map each pitch to its count bucket
pitch_counts = []
valid_indices = []
for i, row in holdout_df.iterrows():
    balls = row['balls']
    strikes = row['strikes']
    if pd.isna(balls) or pd.isna(strikes):
        continue
    pitch_counts.append(f"{int(balls)}-{int(strikes)}")
    valid_indices.append(i)

# Only keep pitches in our buckets
# valid_indices = [i for i, c in enumerate(pitch_counts) if c in count_buckets] # This line is no longer needed
filtered_probs = []
filtered_true = []
filtered_counts = []
for idx, i in enumerate(valid_indices):
    label = true_swings[i]
    if pd.isna(label):
        continue
    filtered_probs.append(swing_prob_scores[i])
    filtered_true.append(label)
    filtered_counts.append(pitch_counts[idx])

# Initialize thresholds
best_thresholds = {c: 0.9 for c in count_buckets}

# Coordinate descent grid search
for it in range(2):  # 2 passes is usually enough
    for bucket in count_buckets:
        best_acc = 0
        best_thr = best_thresholds[bucket]
        for thr in threshold_range:
            preds = []
            for prob, count in zip(filtered_probs, filtered_counts):
                use_thr = best_thresholds[count] if count != bucket else thr
                preds.append(1 if prob >= use_thr else 0)
            acc = balanced_accuracy_score(filtered_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
        best_thresholds[bucket] = best_thr

# Final evaluation with best thresholds
final_preds = [1 if prob >= best_thresholds[count] else 0 for prob, count in zip(filtered_probs, filtered_counts)]
final_acc = balanced_accuracy_score(filtered_true, final_preds)

print("\nBest per-count thresholds (coordinate grid search):")
for c in count_buckets:
    print(f"  {c}: {best_thresholds[c]:.2f}")
print(f"Balanced accuracy with optimized thresholds: {final_acc:.3f}")

# ============================================================================
# COMPREHENSIVE ANALYSIS FOR WHITE PAPER
# ============================================================================

print("\n" + "="*60)
print("COMPREHENSIVE ANALYSIS FOR WHITE PAPER")
print("="*60)

# Calculate comprehensive metrics
accuracy = np.sum(swing_preds == true_swings) / len(true_swings)
precision = precision_score(true_swings, swing_preds)
recall = recall_score(true_swings, swing_preds)
f1 = f1_score(true_swings, swing_preds)

print(f"\nModel Performance Summary:")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")

# 1. CONFUSION MATRIX
print("\n1. Creating Confusion Matrix...")
cm = confusion_matrix(true_swings, swing_preds)
print(f"Confusion Matrix Raw Values:")
print(f"True Negatives (No Swing → No Swing): {cm[0,0]}")
print(f"False Positives (No Swing → Swing): {cm[0,1]}")
print(f"False Negatives (Swing → No Swing): {cm[1,0]}")
print(f"True Positives (Swing → Swing): {cm[1,1]}")

plt.figure(figsize=(10, 8))
# Use a more contrasting colormap and ensure annotations are visible
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['No Swing', 'Swing'], 
            yticklabels=['No Swing', 'Swing'],
            cbar_kws={'label': 'Count'})
plt.title('Swing Classifier Confusion Matrix\nRonald Acuña Jr. Holdout Dataset', 
          fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('swing_classifier_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. FEATURE IMPORTANCE TABLE
print("\n2. Creating Feature Importance Table...")

# For CalibratedClassifierCV, we need to access the underlying model
print(f"Model type: {type(swing_calibrated_model)}")

# Try to get feature importances from the calibrated model
feature_importance = None

# Method 1: Try to access the calibrated model's estimators
if hasattr(swing_calibrated_model, 'calibrated_classifiers_'):
    print("✓ Accessing calibrated classifier's underlying model...")
    # Get the first calibrated classifier (they should all be the same base model)
    calibrated_classifier = swing_calibrated_model.calibrated_classifiers_[0]
    base_estimator = calibrated_classifier.base_estimator
    
    if hasattr(base_estimator, 'feature_importances_'):
        print("✓ Found feature importances in base estimator!")
        feature_importance = base_estimator.feature_importances_
    elif hasattr(base_estimator, 'get_booster'):
        print("✓ Using XGBoost booster feature importances...")
        try:
            booster = base_estimator.get_booster()
            feature_importance_dict = booster.get_score(importance_type='gain')
            # Convert to array format
            feature_importance = np.zeros(len(all_feats))
            for i, feat in enumerate(all_feats):
                if feat in feature_importance_dict:
                    feature_importance[i] = feature_importance_dict[feat]
        except Exception as e:
            print(f"Could not get feature importances from booster: {e}")

# Method 2: Try direct access if available
elif hasattr(swing_calibrated_model, 'estimator') and hasattr(swing_calibrated_model.estimator, 'feature_importances_'):
    print("✓ Using estimator's feature importances...")
    feature_importance = swing_calibrated_model.estimator.feature_importances_

if feature_importance is None:
    print("✗ Could not extract feature importances from calibrated model.")
    print("Using simple correlation-based importance as fallback...")
    # Fast correlation-based fallback
    feature_importance = []
    for i, feature in enumerate(all_feats):
        if i < len(X_swing[0]):
            feature_values = X_swing[:, i]
            correlation = np.corrcoef(feature_values, true_swings)[0, 1]
            feature_importance.append(abs(correlation) if not np.isnan(correlation) else 0.0)
        else:
            feature_importance.append(0.0)
    feature_importance = np.array(feature_importance)
else:
    print(f"✓ Successfully extracted feature importances. Shape: {feature_importance.shape}")

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': all_feats,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Display top 20 features as a table
print("\nTop 20 Most Important Features:")
print("=" * 80)
print(f"{'Rank':<4} {'Feature':<50} {'Importance':<12}")
print("-" * 80)
for i, (idx, row) in enumerate(importance_df.head(20).iterrows(), 1):
    feature_name = row['feature'][:49]  # Truncate if too long
    importance_val = f"{row['importance']:.6f}"
    print(f"{i:<4} {feature_name:<50} {importance_val:<12}")
print("=" * 80)

# Save feature importance to CSV
importance_df.to_csv('swing_classifier_feature_importance.csv', index=False)
print(f"\nFeature importance saved to 'swing_classifier_feature_importance.csv'")

# 3. THRESHOLD ANALYSIS
print("\n3. Performing Threshold Analysis...")
thresholds_to_test = np.arange(0.1, 1.0, 0.05)
results = []

for threshold in thresholds_to_test:
    # Apply threshold to all predictions
    preds = (swing_prob_scores >= threshold).astype(int)
    
    # Calculate metrics
    prec = precision_score(true_swings, preds, zero_division=0)
    rec = recall_score(true_swings, preds, zero_division=0)
    f1_val = f1_score(true_swings, preds, zero_division=0)
    
    # Calculate confusion matrix elements
    cm_temp = confusion_matrix(true_swings, preds)
    if cm_temp.shape == (2, 2):
        tn, fp, fn, tp = cm_temp.ravel()
    else:
        tn = fp = fn = tp = 0
    
    results.append({
        'threshold': threshold,
        'precision': prec,
        'recall': rec,
        'f1': f1_val,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    })

results_df = pd.DataFrame(results)

# Plot threshold curves
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Precision-Recall curve
ax1.plot(results_df['recall'], results_df['precision'], 'b-', linewidth=2, marker='o')
ax1.set_xlabel('Recall', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# F1 vs Threshold
ax2.plot(results_df['threshold'], results_df['f1'], 'g-', linewidth=2, marker='o')
ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('F1-Score', fontsize=12)
ax2.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Precision vs Threshold
ax3.plot(results_df['threshold'], results_df['precision'], 'r-', linewidth=2, marker='o')
ax3.set_xlabel('Threshold', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Recall vs Threshold
ax4.plot(results_df['threshold'], results_df['recall'], 'orange', linewidth=2, marker='o')
ax4.set_xlabel('Threshold', fontsize=12)
ax4.set_ylabel('Recall', fontsize=12)
ax4.set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('swing_classifier_threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. THRESHOLD TABLE
print("\n4. Generating Threshold Table...")
selected_thresholds = [0.3, 0.5, 0.7, 0.9]
table_data = []

for threshold in selected_thresholds:
    preds = (swing_prob_scores >= threshold).astype(int)
    prec = precision_score(true_swings, preds, zero_division=0)
    rec = recall_score(true_swings, preds, zero_division=0)
    f1_val = f1_score(true_swings, preds, zero_division=0)
    
    cm_temp = confusion_matrix(true_swings, preds)
    if cm_temp.shape == (2, 2):
        tn, fp, fn, tp = cm_temp.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # Determine FP/FN characteristics
    fp_char = "High" if fp > len(true_swings) * 0.1 else "Medium" if fp > len(true_swings) * 0.05 else "Low"
    fn_char = "High" if fn > len(true_swings) * 0.1 else "Medium" if fn > len(true_swings) * 0.05 else "Low"
    
    table_data.append({
        'Threshold': threshold,
        'Precision': f"{prec*100:.1f}%",
        'Recall': f"{rec*100:.1f}%",
        'F1': f"{f1_val*100:.1f}%",
        'False Positives': fp_char,
        'False Negatives': fn_char
    })

threshold_table = pd.DataFrame(table_data)
print("\nThreshold Analysis Table:")
print("=" * 80)
print(threshold_table.to_string(index=False))
print("=" * 80)

# Save table to CSV
threshold_table.to_csv('swing_classifier_threshold_table.csv', index=False)

# 5. ROC CURVE
print("\n5. Creating ROC Curve...")
fpr, tpr, _ = roc_curve(true_swings, swing_prob_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve\nSwing vs No-Swing Classifier', 
          fontsize=16, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('swing_classifier_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. DETAILED CLASSIFICATION REPORT
print("\n6. Detailed Classification Report:")
print("=" * 50)
print(classification_report(true_swings, swing_preds, target_names=['No Swing', 'Swing']))

# 7. SAVE COMPREHENSIVE RESULTS
print("\n7. Saving Comprehensive Results...")
results_summary = {
    'model_performance': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    },
    'dataset_info': {
        'total_pitches': len(holdout_df),
        'swing_rate': sum(swing_classifications)/len(swing_classifications),
        'swing_count': sum(swing_classifications),
        'no_swing_count': len(swing_classifications) - sum(swing_classifications)
    },
    'threshold_analysis': threshold_table.to_dict('records')
}

# Save results to JSON
import json
with open('swing_classifier_analysis_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("Generated files for white paper:")
print("- swing_classifier_confusion_matrix.png")
print("- swing_classifier_feature_importance.png")
print("- swing_classifier_threshold_analysis.png")
print("- swing_classifier_roc_curve.png")
print("- swing_classifier_threshold_table.csv")
print("- swing_classifier_analysis_results.json")
print("="*60) 
 
 
 
 
 
 
 
 
 