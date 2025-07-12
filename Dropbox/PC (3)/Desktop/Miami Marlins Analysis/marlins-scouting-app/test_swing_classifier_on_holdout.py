import pandas as pd
import numpy as np
import pickle
from collections import Counter

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
for i, (prob, row) in enumerate(zip(swing_prob_scores, holdout_df.iterrows())):
    balls = row[1].get('balls', 0)
    strikes = row[1].get('strikes', 0)
    
    # Determine count situation and threshold
    if balls <= 1 and strikes <= 1:
        threshold = 0.95 # Very high threshold for early counts (≤1 ball, ≤1 strike)
    elif balls == 1 and strikes == 1:
        threshold = 0.85 # High threshold for middle counts (1-1)
    elif strikes >= 2 or balls >= 3:
        threshold = 0.75 # Lower threshold for pressure situations (≥2 strikes or ≥3 balls)
    else:
        threshold = 0.9 # Default threshold for other situations
    
    # Make prediction with count-specific threshold
    swing_preds.append(1 if prob >= threshold else 0)

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