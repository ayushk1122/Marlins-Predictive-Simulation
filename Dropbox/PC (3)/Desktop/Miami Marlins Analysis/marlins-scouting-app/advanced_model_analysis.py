import pandas as pd
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_performance(model_file, holdout_file):
    """Comprehensive analysis of model performance"""
    
    # Load data
    holdout_df = pd.read_csv(holdout_file)
    with open(model_file, "rb") as f:
        models = pickle.load(f)
    
    swing_features = models['swing_features']
    swing_preprocessor = models['swing_preprocessor']
    swing_model = models['swing_model']
    
    # Define swing events
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    
    def fill_categoricals(df_row, features, preprocessor):
        cat_features = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'cat':
                cat_features = cols
        filled = {}
        for f in features:
            val = df_row.get(f, np.nan)
            if f in cat_features and (pd.isna(val) or val is None):
                filled[f] = 'Unknown'
            else:
                filled[f] = val
        return filled
    
    # Test predictions
    results = []
    for idx, row in holdout_df.iterrows():
        true_swing = 1 if row.get('description') in swing_events else 0
        swing_input = pd.DataFrame([fill_categoricals(row, swing_features, swing_preprocessor)])
        X_swing = swing_preprocessor.transform(swing_input)
        swing_probs = swing_model.predict_proba(X_swing)[0]
        swing_pred = np.argmax(swing_probs)
        swing_confidence = np.max(swing_probs)
        
        results.append({
            'true_swing': true_swing,
            'predicted_swing': swing_pred,
            'confidence': swing_confidence,
            'pitch_type': row.get('pitch_type', 'N/A'),
            'description': row.get('description', 'N/A'),
            'balls': row.get('balls', 0),
            'strikes': row.get('strikes', 0),
            'plate_x': row.get('plate_x', 0),
            'plate_z': row.get('plate_z', 0),
            'zone': row.get('zone', 0),
            'release_speed': row.get('release_speed', 0),
            'movement_magnitude': row.get('movement_magnitude', 0),
            'zone_distance': row.get('zone_distance', 0)
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    correct = (results_df['true_swing'] == results_df['predicted_swing']).sum()
    total = len(results_df)
    accuracy = correct / total
    
    # Confusion matrix
    tp = ((results_df['true_swing'] == 1) & (results_df['predicted_swing'] == 1)).sum()
    tn = ((results_df['true_swing'] == 0) & (results_df['predicted_swing'] == 0)).sum()
    fp = ((results_df['true_swing'] == 0) & (results_df['predicted_swing'] == 1)).sum()
    fn = ((results_df['true_swing'] == 1) & (results_df['predicted_swing'] == 0)).sum()
    
    precision_no_swing = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_no_swing = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_swing = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_swing = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision_no_swing': precision_no_swing,
        'recall_no_swing': recall_no_swing,
        'precision_swing': precision_swing,
        'recall_swing': recall_swing,
        'results_df': results_df
    }

def analyze_errors(results_df):
    """Analyze where the model makes errors"""
    
    # False positives (predicted swing, actually no swing)
    fp_df = results_df[(results_df['true_swing'] == 0) & (results_df['predicted_swing'] == 1)]
    
    # False negatives (predicted no swing, actually swing)
    fn_df = results_df[(results_df['true_swing'] == 1) & (results_df['predicted_swing'] == 0)]
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    print(f"\nFalse Positives (predicted swing, actually no swing): {len(fp_df)}")
    if len(fp_df) > 0:
        print("Top pitch types in false positives:")
        print(fp_df['pitch_type'].value_counts().head())
        print("\nTop descriptions in false positives:")
        print(fp_df['description'].value_counts().head())
        print(f"\nAverage confidence in false positives: {fp_df['confidence'].mean():.3f}")
        print(f"Average zone distance in false positives: {fp_df['zone_distance'].mean():.3f}")
    
    print(f"\nFalse Negatives (predicted no swing, actually swing): {len(fn_df)}")
    if len(fn_df) > 0:
        print("Top pitch types in false negatives:")
        print(fn_df['pitch_type'].value_counts().head())
        print("\nTop descriptions in false negatives:")
        print(fn_df['description'].value_counts().head())
        print(f"\nAverage confidence in false negatives: {fn_df['confidence'].mean():.3f}")
        print(f"Average zone distance in false negatives: {fn_df['zone_distance'].mean():.3f}")

def analyze_by_count(results_df):
    """Analyze performance by count situation"""
    
    print("\n" + "="*60)
    print("PERFORMANCE BY COUNT SITUATION")
    print("="*60)
    
    # Create count categories
    results_df['count_situation'] = results_df.apply(
        lambda row: f"{row['balls']}-{row['strikes']}", axis=1
    )
    
    count_performance = []
    for count in results_df['count_situation'].unique():
        count_df = results_df[results_df['count_situation'] == count]
        if len(count_df) >= 10:  # Only show counts with enough samples
            accuracy = (count_df['true_swing'] == count_df['predicted_swing']).mean()
            count_performance.append({
                'count': count,
                'samples': len(count_df),
                'accuracy': accuracy,
                'swing_rate': count_df['true_swing'].mean()
            })
    
    count_df = pd.DataFrame(count_performance).sort_values('samples', ascending=False)
    print(count_df.head(10))

def main():
    print("🔍 ADVANCED MODEL ANALYSIS")
    print("="*60)
    
    try:
        results = analyze_model_performance("sequential_models.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        
        print(f"Overall Accuracy: {results['correct']}/{results['total']} = {results['accuracy']:.4f}")
        print(f"No-Swing Precision: {results['precision_no_swing']:.4f}")
        print(f"No-Swing Recall: {results['recall_no_swing']:.4f}")
        print(f"Swing Precision: {results['precision_swing']:.4f}")
        print(f"Swing Recall: {results['recall_swing']:.4f}")
        
        # Analyze errors
        analyze_errors(results['results_df'])
        
        # Analyze by count
        analyze_by_count(results['results_df'])
        
    except FileNotFoundError:
        print("Model file not found. Please run train_sequential_models.py first.")

if __name__ == "__main__":
    main() 
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_model_performance(model_file, holdout_file):
    """Comprehensive analysis of model performance"""
    
    # Load data
    holdout_df = pd.read_csv(holdout_file)
    with open(model_file, "rb") as f:
        models = pickle.load(f)
    
    swing_features = models['swing_features']
    swing_preprocessor = models['swing_preprocessor']
    swing_model = models['swing_model']
    
    # Define swing events
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    
    def fill_categoricals(df_row, features, preprocessor):
        cat_features = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'cat':
                cat_features = cols
        filled = {}
        for f in features:
            val = df_row.get(f, np.nan)
            if f in cat_features and (pd.isna(val) or val is None):
                filled[f] = 'Unknown'
            else:
                filled[f] = val
        return filled
    
    # Test predictions
    results = []
    for idx, row in holdout_df.iterrows():
        true_swing = 1 if row.get('description') in swing_events else 0
        swing_input = pd.DataFrame([fill_categoricals(row, swing_features, swing_preprocessor)])
        X_swing = swing_preprocessor.transform(swing_input)
        swing_probs = swing_model.predict_proba(X_swing)[0]
        swing_pred = np.argmax(swing_probs)
        swing_confidence = np.max(swing_probs)
        
        results.append({
            'true_swing': true_swing,
            'predicted_swing': swing_pred,
            'confidence': swing_confidence,
            'pitch_type': row.get('pitch_type', 'N/A'),
            'description': row.get('description', 'N/A'),
            'balls': row.get('balls', 0),
            'strikes': row.get('strikes', 0),
            'plate_x': row.get('plate_x', 0),
            'plate_z': row.get('plate_z', 0),
            'zone': row.get('zone', 0),
            'release_speed': row.get('release_speed', 0),
            'movement_magnitude': row.get('movement_magnitude', 0),
            'zone_distance': row.get('zone_distance', 0)
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    correct = (results_df['true_swing'] == results_df['predicted_swing']).sum()
    total = len(results_df)
    accuracy = correct / total
    
    # Confusion matrix
    tp = ((results_df['true_swing'] == 1) & (results_df['predicted_swing'] == 1)).sum()
    tn = ((results_df['true_swing'] == 0) & (results_df['predicted_swing'] == 0)).sum()
    fp = ((results_df['true_swing'] == 0) & (results_df['predicted_swing'] == 1)).sum()
    fn = ((results_df['true_swing'] == 1) & (results_df['predicted_swing'] == 0)).sum()
    
    precision_no_swing = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_no_swing = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_swing = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_swing = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision_no_swing': precision_no_swing,
        'recall_no_swing': recall_no_swing,
        'precision_swing': precision_swing,
        'recall_swing': recall_swing,
        'results_df': results_df
    }

def analyze_errors(results_df):
    """Analyze where the model makes errors"""
    
    # False positives (predicted swing, actually no swing)
    fp_df = results_df[(results_df['true_swing'] == 0) & (results_df['predicted_swing'] == 1)]
    
    # False negatives (predicted no swing, actually swing)
    fn_df = results_df[(results_df['true_swing'] == 1) & (results_df['predicted_swing'] == 0)]
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    print(f"\nFalse Positives (predicted swing, actually no swing): {len(fp_df)}")
    if len(fp_df) > 0:
        print("Top pitch types in false positives:")
        print(fp_df['pitch_type'].value_counts().head())
        print("\nTop descriptions in false positives:")
        print(fp_df['description'].value_counts().head())
        print(f"\nAverage confidence in false positives: {fp_df['confidence'].mean():.3f}")
        print(f"Average zone distance in false positives: {fp_df['zone_distance'].mean():.3f}")
    
    print(f"\nFalse Negatives (predicted no swing, actually swing): {len(fn_df)}")
    if len(fn_df) > 0:
        print("Top pitch types in false negatives:")
        print(fn_df['pitch_type'].value_counts().head())
        print("\nTop descriptions in false negatives:")
        print(fn_df['description'].value_counts().head())
        print(f"\nAverage confidence in false negatives: {fn_df['confidence'].mean():.3f}")
        print(f"Average zone distance in false negatives: {fn_df['zone_distance'].mean():.3f}")

def analyze_by_count(results_df):
    """Analyze performance by count situation"""
    
    print("\n" + "="*60)
    print("PERFORMANCE BY COUNT SITUATION")
    print("="*60)
    
    # Create count categories
    results_df['count_situation'] = results_df.apply(
        lambda row: f"{row['balls']}-{row['strikes']}", axis=1
    )
    
    count_performance = []
    for count in results_df['count_situation'].unique():
        count_df = results_df[results_df['count_situation'] == count]
        if len(count_df) >= 10:  # Only show counts with enough samples
            accuracy = (count_df['true_swing'] == count_df['predicted_swing']).mean()
            count_performance.append({
                'count': count,
                'samples': len(count_df),
                'accuracy': accuracy,
                'swing_rate': count_df['true_swing'].mean()
            })
    
    count_df = pd.DataFrame(count_performance).sort_values('samples', ascending=False)
    print(count_df.head(10))

def main():
    print("🔍 ADVANCED MODEL ANALYSIS")
    print("="*60)
    
    try:
        results = analyze_model_performance("sequential_models.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        
        print(f"Overall Accuracy: {results['correct']}/{results['total']} = {results['accuracy']:.4f}")
        print(f"No-Swing Precision: {results['precision_no_swing']:.4f}")
        print(f"No-Swing Recall: {results['recall_no_swing']:.4f}")
        print(f"Swing Precision: {results['precision_swing']:.4f}")
        print(f"Swing Recall: {results['recall_swing']:.4f}")
        
        # Analyze errors
        analyze_errors(results['results_df'])
        
        # Analyze by count
        analyze_by_count(results['results_df'])
        
    except FileNotFoundError:
        print("Model file not found. Please run train_sequential_models.py first.")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
 
 
 
 