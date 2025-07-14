import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict

# Load holdout dataset
holdout_df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")

# Load models
with open("sequential_models.pkl", "rb") as f:
    models = pickle.load(f)

# Helper: Prepare features for a single row
# Only use features the model expects
swing_features = models['swing_features']
swing_outcome_features = models['swing_outcome_features']
no_swing_features = models['no_swing_features']

def get_expected_outcome(row):
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    if row.get('description') in swing_events:
        # Swing outcome
        if row.get('description') in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row.get('events') in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row.get('events') == 'field_out':
            return 'field_out'
        else:
            return 'field_out'
    elif row.get('description') in no_swing_events:
        # No-swing outcome
        if row.get('description') == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row.get('description') == 'called_strike':
            return 'strike'
        elif row.get('description') in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'
    else:
        return None

def fill_categoricals(df_row, features, preprocessor):
    # Get categorical feature names from the preprocessor
    cat_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'cat':
            cat_features = cols
    # Fill missing categoricals with 'Unknown'
    filled = {}
    for f in features:
        val = df_row.get(f, np.nan)
        if f in cat_features and (pd.isna(val) or val is None):
            filled[f] = 'Unknown'
        else:
            filled[f] = val
    return filled

# Use all pitches in the holdout set
sample_df = holdout_df  # no sampling

total = 0
correct = 0

# Analysis containers
missing_hyper_speed = []
correct_hyper_speed = []
wrong_hyper_speed = []
correct_pitch_types = []
wrong_pitch_types = []
correct_descriptions = []
wrong_descriptions = []
correct_events = []
wrong_events = []

for idx, row in sample_df.iterrows():
    expected = get_expected_outcome(row)
    model_final = None
    total += 1

    # Step 1: Swing/No-Swing
    swing_input = pd.DataFrame([fill_categoricals(row, swing_features, models['swing_preprocessor'])])
    X_swing = models['swing_preprocessor'].transform(swing_input)
    
    # Handle NaN values in the transformed data
    X_swing = np.nan_to_num(X_swing, nan=0.0)
    
    swing_probs = models['swing_model'].predict_proba(X_swing)[0]
    swing_pred = np.argmax(swing_probs)

    if swing_pred == 1:
        # Step 2: Swing Outcome
        swing_outcome_input = pd.DataFrame([fill_categoricals(row, swing_outcome_features, models['swing_outcome_preprocessor'])])
        X_swing_outcome = models['swing_outcome_preprocessor'].transform(swing_outcome_input)
        
        # Handle NaN values in the transformed data
        X_swing_outcome = np.nan_to_num(X_swing_outcome, nan=0.0)
        
        swing_outcome_probs = models['swing_outcome_model'].predict_proba(X_swing_outcome)[0]
        swing_outcome_pred_idx = np.argmax(swing_outcome_probs)
        swing_outcome_label = models['swing_outcome_le'].inverse_transform([swing_outcome_pred_idx])[0]
        model_final = swing_outcome_label
    else:
        # Step 3: No-Swing Outcome
        no_swing_input = pd.DataFrame([fill_categoricals(row, no_swing_features, models['no_swing_preprocessor'])])
        X_no_swing = models['no_swing_preprocessor'].transform(no_swing_input)
        
        # Handle NaN values in the transformed data
        X_no_swing = np.nan_to_num(X_no_swing, nan=0.0)
        
        no_swing_probs = models['no_swing_model'].predict_proba(X_no_swing)[0]
        no_swing_pred_idx = np.argmax(no_swing_probs)
        no_swing_label = models['no_swing_le'].inverse_transform([no_swing_pred_idx])[0]
        model_final = no_swing_label

    is_correct = expected == model_final
    if is_correct:
        correct += 1
        correct_hyper_speed.append(pd.isna(row.get('hyper_speed')))
        correct_pitch_types.append(row.get('pitch_type', 'N/A'))
        correct_descriptions.append(row.get('description', 'N/A'))
        correct_events.append(row.get('events', 'N/A'))
    else:
        wrong_hyper_speed.append(pd.isna(row.get('hyper_speed')))
        wrong_pitch_types.append(row.get('pitch_type', 'N/A'))
        wrong_descriptions.append(row.get('description', 'N/A'))
        wrong_events.append(row.get('events', 'N/A'))
        missing_hyper_speed.append(pd.isna(row.get('hyper_speed')))

# Accuracy for pitches with and without hyper_speed
num_with_hyper = sum([not x for x in missing_hyper_speed + correct_hyper_speed])
num_without_hyper = sum([x for x in missing_hyper_speed + correct_hyper_speed])
wrong_with_hyper = sum([not x for x in wrong_hyper_speed])
wrong_without_hyper = sum([x for x in wrong_hyper_speed])
correct_with_hyper = sum([not x for x in correct_hyper_speed])
correct_without_hyper = sum([x for x in correct_hyper_speed])

print(f"\n==============================")
print(f"Total correct predictions: {correct}")
print(f"Total predictions: {total}")
print(f"Overall accuracy: {correct / total:.4f}")
print(f"\n--- Hyper Speed Analysis ---")
print(f"Pitches WITH hyper_speed: {num_with_hyper}")
print(f"  Correct: {correct_with_hyper} | Wrong: {wrong_with_hyper} | Accuracy: {correct_with_hyper / num_with_hyper if num_with_hyper else 0:.4f}")
print(f"Pitches WITHOUT hyper_speed: {num_without_hyper}")
print(f"  Correct: {correct_without_hyper} | Wrong: {wrong_without_hyper} | Accuracy: {correct_without_hyper / num_without_hyper if num_without_hyper else 0:.4f}")

print(f"\n--- Top 10 Most Common Wrong Pitch Types ---")
for pitch, count in Counter(wrong_pitch_types).most_common(10):
    print(f"  {pitch}: {count}")

print(f"\n--- Top 10 Most Common Wrong Descriptions ---")
for desc, count in Counter(wrong_descriptions).most_common(10):
    print(f"  {desc}: {count}")

print(f"\n--- Top 10 Most Common Wrong Events ---")
for event, count in Counter(wrong_events).most_common(10):
    print(f"  {event}: {count}")

# Check if hyper_speed is the same as exit velocity (launch_speed)
if 'hyper_speed' in holdout_df.columns and 'launch_speed' in holdout_df.columns:
    both_present = holdout_df.dropna(subset=['hyper_speed', 'launch_speed'])
    if not both_present.empty:
        corr = both_present['hyper_speed'].corr(both_present['launch_speed'])
        print(f"\n--- hyper_speed vs. launch_speed correlation: {corr:.4f}")
        print("Sample values (hyper_speed, launch_speed):")
        print(both_present[['hyper_speed', 'launch_speed']].head(10))
        print("Difference stats:")
        print((both_present['hyper_speed'] - both_present['launch_speed']).describe())
    else:
        print("No rows with both hyper_speed and launch_speed present.")
else:
    print("hyper_speed or launch_speed column not found in holdout data.") 
 
 
 
 
 
 
 
 
 
 
 