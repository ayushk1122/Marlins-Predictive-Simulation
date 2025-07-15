import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load holdout dataset
print("Loading holdout dataset...")
holdout_df = pd.read_csv("ronald_acuna_jr_holdout_statcast.csv")

# Load models
print("Loading trained models...")
with open("sequential_models.pkl", "rb") as f:
    models = pickle.load(f)

swing_features = models['swing_features']
swing_preprocessor = models['swing_preprocessor']
swing_calibrated_model = models['swing_calibrated_model']
swing_threshold = models.get('swing_threshold', 0.9)

# Import the same functions from training script
from train_sequential_models import prepare_features, calculate_zone

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']

def classify_swing_robust(row):
    """Robustly classify whether a pitch resulted in a swing or not."""
    description = row.get('description', None)
    events = row.get('events', None)
    
    if pd.isna(description) and pd.isna(events):
        return None
    
    if not pd.isna(description):
        if description in swing_events:
            return 1
        elif description in no_swing_events:
            return 0
    
    if not pd.isna(events):
        if events in ['single', 'double', 'triple', 'home_run', 'field_out', 'sac_fly', 'sac_bunt']:
            return 1
        elif events in ['walk', 'strikeout', 'hit_by_pitch']:
            return 0
    
    return None

# Create swing/no-swing target
print("Classifying swings and no-swings...")
swing_classifications = []
valid_indices = []

for idx, row in holdout_df.iterrows():
    classification = classify_swing_robust(row)
    if classification is not None:
        swing_classifications.append(classification)
        valid_indices.append(idx)

# Create valid dataset
holdout_df_valid = holdout_df.iloc[valid_indices].copy()
holdout_df_valid['swing'] = swing_classifications
holdout_df = holdout_df_valid

print(f"Dataset size: {len(holdout_df)}")
print(f"Swing rate: {sum(swing_classifications)/len(swing_classifications)*100:.1f}%")

# Prepare features
print("Preparing features...")
holdout_df, num_feats, cat_feats = prepare_features(holdout_df)

# Update swing rates
from train_sequential_models import update_swing_rates
holdout_df = update_swing_rates(holdout_df)

# Get features for model
all_feats = num_feats + cat_feats
model_input = holdout_df[all_feats]
X_swing = swing_preprocessor.transform(model_input)
X_swing = np.nan_to_num(X_swing, nan=0.0)

# Get predictions and probabilities
swing_probs = swing_calibrated_model.predict_proba(X_swing)
swing_prob_scores = swing_probs[:, 1]

# Apply count-specific thresholds
swing_preds = []
thresholds = []
for i, (prob, row) in enumerate(zip(swing_prob_scores, holdout_df.iterrows())):
    balls = row[1].get('balls', 0)
    strikes = row[1].get('strikes', 0)
    
    # Count-specific thresholds
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
        threshold = 0.9
    
    swing_preds.append(1 if prob >= threshold else 0)
    thresholds.append(threshold)

swing_preds = np.array(swing_preds)
true_swings = holdout_df['swing'].values

# Calculate metrics
accuracy = np.sum(swing_preds == true_swings) / len(true_swings)
precision = precision_score(true_swings, swing_preds)
recall = recall_score(true_swings, swing_preds)
f1 = f1_score(true_swings, swing_preds)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")

# 1. CONFUSION MATRIX
print("\nCreating confusion matrix...")
cm = confusion_matrix(true_swings, swing_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Swing', 'Swing'], 
            yticklabels=['No Swing', 'Swing'])
plt.title('Swing Classifier Confusion Matrix\nRonald Acuña Jr. Holdout Dataset', 
          fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('swing_classifier_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. FEATURE IMPORTANCE
print("\nExtracting feature importance...")
if hasattr(swing_calibrated_model, 'feature_importances_'):
    feature_importance = swing_calibrated_model.feature_importances_
else:
    # For models without feature_importances_, use permutation importance
    from sklearn.inspection import permutation_importance
    r = permutation_importance(swing_calibrated_model, X_swing, true_swings, n_repeats=10, random_state=42)
    feature_importance = r.importances_mean

# Get feature names
feature_names = all_feats

# Create feature importance DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
bars = plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 20 Most Important Features\nSwing vs No-Swing Classifier', 
          fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()

# Color bars by importance
colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())
for i, (bar, color) in enumerate(zip(bars, colors)):
    bar.set_color(color)

plt.tight_layout()
plt.savefig('swing_classifier_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. THRESHOLD ANALYSIS
print("\nPerforming threshold analysis...")
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
print("\nGenerating threshold table...")
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
print("\nCreating ROC curve...")
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
print("\nDetailed Classification Report:")
print("=" * 50)
print(classification_report(true_swings, swing_preds, target_names=['No Swing', 'Swing']))

# 7. SAVE COMPREHENSIVE RESULTS
print("\nSaving comprehensive results...")
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

print("\nAnalysis complete! Generated files:")
print("- swing_classifier_confusion_matrix.png")
print("- swing_classifier_feature_importance.png")
print("- swing_classifier_threshold_analysis.png")
print("- swing_classifier_roc_curve.png")
print("- swing_classifier_threshold_table.csv")
print("- swing_classifier_analysis_results.json") 