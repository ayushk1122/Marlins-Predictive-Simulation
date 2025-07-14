import pandas as pd
import numpy as np
import pickle
from collections import Counter

def test_swing_classifier(model_file, holdout_file):
    """Test a swing classifier and return detailed results"""
    
    # Load holdout dataset
    holdout_df = pd.read_csv(holdout_file)
    
    # Load models
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
    
    correct = 0
    total = 0
    
    # Analysis containers
    fp_pitch_types = []  # False positive: predicted swing, actually no swing
    fn_pitch_types = []  # False negative: predicted no swing, actually swing
    fp_descriptions = []
    fn_descriptions = []
    
    for idx, row in holdout_df.iterrows():
        # Ground truth: 1 if swing event, 0 otherwise
        true_swing = 1 if row.get('description') in swing_events else 0
        swing_input = pd.DataFrame([fill_categoricals(row, swing_features, swing_preprocessor)])
        X_swing = swing_preprocessor.transform(swing_input)
        swing_probs = swing_model.predict_proba(X_swing)[0]
        swing_pred = np.argmax(swing_probs)
        if swing_pred == true_swing:
            correct += 1
        else:
            if swing_pred == 1:
                fp_pitch_types.append(row.get('pitch_type', 'N/A'))
                fp_descriptions.append(row.get('description', 'N/A'))
            else:
                fn_pitch_types.append(row.get('pitch_type', 'N/A'))
                fn_descriptions.append(row.get('description', 'N/A'))
        total += 1
    
    accuracy = correct / total
    
    # Calculate precision and recall for no-swing (class 0)
    fp_count = len(fp_pitch_types)  # Predicted swing, actually no swing
    fn_count = len(fn_pitch_types)  # Predicted no swing, actually swing
    
    # Count actual swings and no-swings
    actual_swings = sum(1 for row in holdout_df.iterrows() if row[1].get('description') in swing_events)
    actual_no_swings = total - actual_swings
    
    precision_no_swing = (actual_no_swings - fp_count) / (actual_no_swings - fp_count + fn_count) if (actual_no_swings - fp_count + fn_count) > 0 else 0
    recall_no_swing = (actual_no_swings - fp_count) / actual_no_swings if actual_no_swings > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'precision_no_swing': precision_no_swing,
        'recall_no_swing': recall_no_swing,
        'fp_pitch_types': Counter(fp_pitch_types).most_common(5),
        'fp_descriptions': Counter(fp_descriptions).most_common(5)
    }

def main():
    print("🔄 COMPARING SWING CLASSIFIER MODELS")
    print("=" * 60)
    
    # Test old model (if it exists)
    try:
        old_results = test_swing_classifier("sequential_models_old.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("OLD MODEL RESULTS:")
        print(f"  Accuracy: {old_results['correct']}/{old_results['total']} = {old_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {old_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {old_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {old_results['fp_count']}")
        print(f"  False Negatives: {old_results['fn_count']}")
        print()
    except FileNotFoundError:
        print("No old model found for comparison")
        print()
    
    # Test new model
    try:
        new_results = test_swing_classifier("sequential_models.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("NEW MODEL RESULTS:")
        print(f"  Accuracy: {new_results['correct']}/{new_results['total']} = {new_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {new_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {new_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {new_results['fp_count']}")
        print(f"  False Negatives: {new_results['fn_count']}")
        print()
        
        print("TOP FALSE POSITIVE PITCH TYPES (predicted swing, actually no swing):")
        for pitch_type, count in new_results['fp_pitch_types']:
            print(f"  {pitch_type}: {count}")
        print()
        
        print("TOP FALSE POSITIVE DESCRIPTIONS:")
        for desc, count in new_results['fp_descriptions']:
            print(f"  {desc}: {count}")
        
    except FileNotFoundError:
        print("New model not found. Please run train_sequential_models.py first.")

if __name__ == "__main__":
    main() 
import numpy as np
import pickle
from collections import Counter

def test_swing_classifier(model_file, holdout_file):
    """Test a swing classifier and return detailed results"""
    
    # Load holdout dataset
    holdout_df = pd.read_csv(holdout_file)
    
    # Load models
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
    
    correct = 0
    total = 0
    
    # Analysis containers
    fp_pitch_types = []  # False positive: predicted swing, actually no swing
    fn_pitch_types = []  # False negative: predicted no swing, actually swing
    fp_descriptions = []
    fn_descriptions = []
    
    for idx, row in holdout_df.iterrows():
        # Ground truth: 1 if swing event, 0 otherwise
        true_swing = 1 if row.get('description') in swing_events else 0
        swing_input = pd.DataFrame([fill_categoricals(row, swing_features, swing_preprocessor)])
        X_swing = swing_preprocessor.transform(swing_input)
        swing_probs = swing_model.predict_proba(X_swing)[0]
        swing_pred = np.argmax(swing_probs)
        if swing_pred == true_swing:
            correct += 1
        else:
            if swing_pred == 1:
                fp_pitch_types.append(row.get('pitch_type', 'N/A'))
                fp_descriptions.append(row.get('description', 'N/A'))
            else:
                fn_pitch_types.append(row.get('pitch_type', 'N/A'))
                fn_descriptions.append(row.get('description', 'N/A'))
        total += 1
    
    accuracy = correct / total
    
    # Calculate precision and recall for no-swing (class 0)
    fp_count = len(fp_pitch_types)  # Predicted swing, actually no swing
    fn_count = len(fn_pitch_types)  # Predicted no swing, actually swing
    
    # Count actual swings and no-swings
    actual_swings = sum(1 for row in holdout_df.iterrows() if row[1].get('description') in swing_events)
    actual_no_swings = total - actual_swings
    
    precision_no_swing = (actual_no_swings - fp_count) / (actual_no_swings - fp_count + fn_count) if (actual_no_swings - fp_count + fn_count) > 0 else 0
    recall_no_swing = (actual_no_swings - fp_count) / actual_no_swings if actual_no_swings > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'precision_no_swing': precision_no_swing,
        'recall_no_swing': recall_no_swing,
        'fp_pitch_types': Counter(fp_pitch_types).most_common(5),
        'fp_descriptions': Counter(fp_descriptions).most_common(5)
    }

def main():
    print("🔄 COMPARING SWING CLASSIFIER MODELS")
    print("=" * 60)
    
    # Test old model (if it exists)
    try:
        old_results = test_swing_classifier("sequential_models_old.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("OLD MODEL RESULTS:")
        print(f"  Accuracy: {old_results['correct']}/{old_results['total']} = {old_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {old_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {old_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {old_results['fp_count']}")
        print(f"  False Negatives: {old_results['fn_count']}")
        print()
    except FileNotFoundError:
        print("No old model found for comparison")
        print()
    
    # Test new model
    try:
        new_results = test_swing_classifier("sequential_models.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("NEW MODEL RESULTS:")
        print(f"  Accuracy: {new_results['correct']}/{new_results['total']} = {new_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {new_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {new_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {new_results['fp_count']}")
        print(f"  False Negatives: {new_results['fn_count']}")
        print()
        
        print("TOP FALSE POSITIVE PITCH TYPES (predicted swing, actually no swing):")
        for pitch_type, count in new_results['fp_pitch_types']:
            print(f"  {pitch_type}: {count}")
        print()
        
        print("TOP FALSE POSITIVE DESCRIPTIONS:")
        for desc, count in new_results['fp_descriptions']:
            print(f"  {desc}: {count}")
        
    except FileNotFoundError:
        print("New model not found. Please run train_sequential_models.py first.")

if __name__ == "__main__":
    main() 
 
import numpy as np
import pickle
from collections import Counter

def test_swing_classifier(model_file, holdout_file):
    """Test a swing classifier and return detailed results"""
    
    # Load holdout dataset
    holdout_df = pd.read_csv(holdout_file)
    
    # Load models
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
    
    correct = 0
    total = 0
    
    # Analysis containers
    fp_pitch_types = []  # False positive: predicted swing, actually no swing
    fn_pitch_types = []  # False negative: predicted no swing, actually swing
    fp_descriptions = []
    fn_descriptions = []
    
    for idx, row in holdout_df.iterrows():
        # Ground truth: 1 if swing event, 0 otherwise
        true_swing = 1 if row.get('description') in swing_events else 0
        swing_input = pd.DataFrame([fill_categoricals(row, swing_features, swing_preprocessor)])
        X_swing = swing_preprocessor.transform(swing_input)
        swing_probs = swing_model.predict_proba(X_swing)[0]
        swing_pred = np.argmax(swing_probs)
        if swing_pred == true_swing:
            correct += 1
        else:
            if swing_pred == 1:
                fp_pitch_types.append(row.get('pitch_type', 'N/A'))
                fp_descriptions.append(row.get('description', 'N/A'))
            else:
                fn_pitch_types.append(row.get('pitch_type', 'N/A'))
                fn_descriptions.append(row.get('description', 'N/A'))
        total += 1
    
    accuracy = correct / total
    
    # Calculate precision and recall for no-swing (class 0)
    fp_count = len(fp_pitch_types)  # Predicted swing, actually no swing
    fn_count = len(fn_pitch_types)  # Predicted no swing, actually swing
    
    # Count actual swings and no-swings
    actual_swings = sum(1 for row in holdout_df.iterrows() if row[1].get('description') in swing_events)
    actual_no_swings = total - actual_swings
    
    precision_no_swing = (actual_no_swings - fp_count) / (actual_no_swings - fp_count + fn_count) if (actual_no_swings - fp_count + fn_count) > 0 else 0
    recall_no_swing = (actual_no_swings - fp_count) / actual_no_swings if actual_no_swings > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'precision_no_swing': precision_no_swing,
        'recall_no_swing': recall_no_swing,
        'fp_pitch_types': Counter(fp_pitch_types).most_common(5),
        'fp_descriptions': Counter(fp_descriptions).most_common(5)
    }

def main():
    print("🔄 COMPARING SWING CLASSIFIER MODELS")
    print("=" * 60)
    
    # Test old model (if it exists)
    try:
        old_results = test_swing_classifier("sequential_models_old.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("OLD MODEL RESULTS:")
        print(f"  Accuracy: {old_results['correct']}/{old_results['total']} = {old_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {old_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {old_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {old_results['fp_count']}")
        print(f"  False Negatives: {old_results['fn_count']}")
        print()
    except FileNotFoundError:
        print("No old model found for comparison")
        print()
    
    # Test new model
    try:
        new_results = test_swing_classifier("sequential_models.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("NEW MODEL RESULTS:")
        print(f"  Accuracy: {new_results['correct']}/{new_results['total']} = {new_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {new_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {new_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {new_results['fp_count']}")
        print(f"  False Negatives: {new_results['fn_count']}")
        print()
        
        print("TOP FALSE POSITIVE PITCH TYPES (predicted swing, actually no swing):")
        for pitch_type, count in new_results['fp_pitch_types']:
            print(f"  {pitch_type}: {count}")
        print()
        
        print("TOP FALSE POSITIVE DESCRIPTIONS:")
        for desc, count in new_results['fp_descriptions']:
            print(f"  {desc}: {count}")
        
    except FileNotFoundError:
        print("New model not found. Please run train_sequential_models.py first.")

if __name__ == "__main__":
    main() 
import numpy as np
import pickle
from collections import Counter

def test_swing_classifier(model_file, holdout_file):
    """Test a swing classifier and return detailed results"""
    
    # Load holdout dataset
    holdout_df = pd.read_csv(holdout_file)
    
    # Load models
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
    
    correct = 0
    total = 0
    
    # Analysis containers
    fp_pitch_types = []  # False positive: predicted swing, actually no swing
    fn_pitch_types = []  # False negative: predicted no swing, actually swing
    fp_descriptions = []
    fn_descriptions = []
    
    for idx, row in holdout_df.iterrows():
        # Ground truth: 1 if swing event, 0 otherwise
        true_swing = 1 if row.get('description') in swing_events else 0
        swing_input = pd.DataFrame([fill_categoricals(row, swing_features, swing_preprocessor)])
        X_swing = swing_preprocessor.transform(swing_input)
        swing_probs = swing_model.predict_proba(X_swing)[0]
        swing_pred = np.argmax(swing_probs)
        if swing_pred == true_swing:
            correct += 1
        else:
            if swing_pred == 1:
                fp_pitch_types.append(row.get('pitch_type', 'N/A'))
                fp_descriptions.append(row.get('description', 'N/A'))
            else:
                fn_pitch_types.append(row.get('pitch_type', 'N/A'))
                fn_descriptions.append(row.get('description', 'N/A'))
        total += 1
    
    accuracy = correct / total
    
    # Calculate precision and recall for no-swing (class 0)
    fp_count = len(fp_pitch_types)  # Predicted swing, actually no swing
    fn_count = len(fn_pitch_types)  # Predicted no swing, actually swing
    
    # Count actual swings and no-swings
    actual_swings = sum(1 for row in holdout_df.iterrows() if row[1].get('description') in swing_events)
    actual_no_swings = total - actual_swings
    
    precision_no_swing = (actual_no_swings - fp_count) / (actual_no_swings - fp_count + fn_count) if (actual_no_swings - fp_count + fn_count) > 0 else 0
    recall_no_swing = (actual_no_swings - fp_count) / actual_no_swings if actual_no_swings > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'fp_count': fp_count,
        'fn_count': fn_count,
        'precision_no_swing': precision_no_swing,
        'recall_no_swing': recall_no_swing,
        'fp_pitch_types': Counter(fp_pitch_types).most_common(5),
        'fp_descriptions': Counter(fp_descriptions).most_common(5)
    }

def main():
    print("🔄 COMPARING SWING CLASSIFIER MODELS")
    print("=" * 60)
    
    # Test old model (if it exists)
    try:
        old_results = test_swing_classifier("sequential_models_old.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("OLD MODEL RESULTS:")
        print(f"  Accuracy: {old_results['correct']}/{old_results['total']} = {old_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {old_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {old_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {old_results['fp_count']}")
        print(f"  False Negatives: {old_results['fn_count']}")
        print()
    except FileNotFoundError:
        print("No old model found for comparison")
        print()
    
    # Test new model
    try:
        new_results = test_swing_classifier("sequential_models.pkl", "ronald_acuna_jr_holdout_statcast.csv")
        print("NEW MODEL RESULTS:")
        print(f"  Accuracy: {new_results['correct']}/{new_results['total']} = {new_results['accuracy']:.4f}")
        print(f"  No-Swing Precision: {new_results['precision_no_swing']:.4f}")
        print(f"  No-Swing Recall: {new_results['recall_no_swing']:.4f}")
        print(f"  False Positives: {new_results['fp_count']}")
        print(f"  False Negatives: {new_results['fn_count']}")
        print()
        
        print("TOP FALSE POSITIVE PITCH TYPES (predicted swing, actually no swing):")
        for pitch_type, count in new_results['fp_pitch_types']:
            print(f"  {pitch_type}: {count}")
        print()
        
        print("TOP FALSE POSITIVE DESCRIPTIONS:")
        for desc, count in new_results['fp_descriptions']:
            print(f"  {desc}: {count}")
        
    except FileNotFoundError:
        print("New model not found. Please run train_sequential_models.py first.")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 