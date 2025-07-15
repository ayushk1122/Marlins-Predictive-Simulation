import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
 
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
 
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
import numpy as np
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation.
    """
    # Calculate zones for all pitches
    df['zone'] = df.apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    
    # Define features
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'hyper_speed', 'age_pit'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return num_feats, cat_feats

def evaluate_swing_classifier(models, df):
    """
    Evaluate Model 1: Swing/No-Swing Classifier
    """
    print("\n=== EVALUATING MODEL 1: Swing/No-Swing Classifier ===")
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    print(f"Swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_model = models['swing_model']
    swing_preprocessor = models['swing_preprocessor']
    
    # Transform features
    X = swing_preprocessor.transform(df[all_feats])
    y_true = df['swing'].values
    
    # Predict
    y_pred = swing_model.predict(X)
    y_prob = swing_model.predict_proba(X)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Swing', 'Swing']))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_swing_outcome_classifier(models, df):
    """
    Evaluate Model 2: Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 2: Swing Outcome Classifier ===")
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    if len(df_swing) == 0:
        print("No swing events found in holdout dataset")
        return None
    
    # Create swing outcome target
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] == 'field_out':
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    swing_outcome_model = models['swing_outcome_model']
    swing_outcome_preprocessor = models['swing_outcome_preprocessor']
    swing_outcome_le = models['swing_outcome_le']
    
    # Transform features
    X = swing_outcome_preprocessor.transform(df_swing[all_feats])
    y_true = df_swing['swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = swing_outcome_le.transform(y_true)
    
    # Predict
    y_pred_encoded = swing_outcome_model.predict(X)
    y_prob = swing_outcome_model.predict_proba(X)
    
    # Decode predictions
    y_pred = swing_outcome_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def evaluate_no_swing_classifier(models, df):
    """
    Evaluate Model 3: No-Swing Outcome Classifier
    """
    print("\n=== EVALUATING MODEL 3: No-Swing Outcome Classifier ===")
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    if len(df_no_swing) == 0:
        print("No no-swing events found in holdout dataset")
        return None
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features
    num_feats, cat_feats = prepare_features(df_no_swing)
    all_feats = num_feats + cat_feats
    
    # Get model components
    no_swing_model = models['no_swing_model']
    no_swing_preprocessor = models['no_swing_preprocessor']
    no_swing_le = models['no_swing_le']
    
    # Transform features
    X = no_swing_preprocessor.transform(df_no_swing[all_feats])
    y_true = df_no_swing['no_swing_outcome'].values
    
    # Encode true labels
    y_true_encoded = no_swing_le.transform(y_true)
    
    # Predict
    y_pred_encoded = no_swing_model.predict(X)
    y_prob = no_swing_model.predict_proba(X)
    
    # Decode predictions
    y_pred = no_swing_le.inverse_transform(y_pred_encoded)
    
    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_prob,
        'true_labels': y_true
    }

def main():
    """
    Main evaluation function.
    """
    print("🎯 Evaluating Sequential Models on Holdout Dataset")
    print("=" * 60)
    
    # Load holdout dataset
    try:
        df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")
        print(f"Holdout dataset loaded: {len(df)} pitches")
    except FileNotFoundError:
        print("❌ Holdout dataset not found. Please run create_holdout_dataset.py first.")
        return
    
    # Load trained models
    try:
        with open("sequential_models.pkl", "rb") as f:
            models = pickle.load(f)
        print("✅ Trained models loaded successfully")
    except FileNotFoundError:
        print("❌ Trained models not found. Please run train_sequential_models.py first.")
        return
    
    # Evaluate each model
    results = {}
    
    # Model 1: Swing/No-Swing
    results['swing_classifier'] = evaluate_swing_classifier(models, df)
    
    # Model 2: Swing Outcomes
    results['swing_outcome_classifier'] = evaluate_swing_outcome_classifier(models, df)
    
    # Model 3: No-Swing Outcomes
    results['no_swing_classifier'] = evaluate_no_swing_classifier(models, df)
    
    # Save evaluation results
    evaluation_summary = {
        'holdout_dataset_size': len(df),
        'model_performance': {
            'swing_classifier': {
                'accuracy': results['swing_classifier']['accuracy'] if results['swing_classifier'] else None
            },
            'swing_outcome_classifier': {
                'accuracy': results['swing_outcome_classifier']['accuracy'] if results['swing_outcome_classifier'] else None
            },
            'no_swing_classifier': {
                'accuracy': results['no_swing_classifier']['accuracy'] if results['no_swing_classifier'] else None
            }
        }
    }
    
    with open("holdout_evaluation_results.json", "w") as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print("\n✅ Evaluation complete!")
    print("📁 Results saved to 'holdout_evaluation_results.json'")
    
    # Print summary
    print("\n📊 Model Performance Summary:")
    for model_name, result in results.items():
        if result:
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"  {model_name}: No data available for evaluation")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 