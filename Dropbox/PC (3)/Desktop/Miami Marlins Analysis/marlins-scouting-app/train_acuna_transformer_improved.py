# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from xgboost import XGBClassifier
# import pickle

# # Load dataset
# df = pd.read_csv("ronald_acuna_jr_statcast_pitches_sequential.csv")

# # Drop rows missing critical outcome
# df = df.dropna(subset=['description', 'events'])

# # Add whiff column
# df['whiff'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)

# # Map events to event_class (including whiff as separate class)
# def get_event_class(row):
#     if row['whiff'] == 1:
#         return 'whiff'
#     if pd.notna(row['events']):
#         return row['events']
#     return np.nan

# df['event_class'] = df.apply(get_event_class, axis=1)

# # Only keep rows where event_class is one of the 9 we're predicting
# valid_classes = ['field_out', 'walk', 'single', 'double', 'strikeout',
#                  'home_run', 'hit_by_pitch', 'triple', 'whiff']
# df = df[df['event_class'].isin(valid_classes)]

# # Calculate average swing metrics and fill missing
# swing_metrics = ['bat_speed', 'attack_angle', 'swing_path_tilt', 'swing_length']
# avg_metrics = df[swing_metrics].mean().to_dict()
# print("✅ Averages used to fill swing metrics:", avg_metrics)
# for metric, avg in avg_metrics.items():
#     df[metric] = avg

# # Define features
# num_feats = [
#     'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
#     'release_pos_x', 'release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
#     'plate_x', 'plate_z', 'api_break_z_with_gravity', 'api_break_x_batter_in',
#     'arm_angle', 'attack_angle', 'bat_speed', 'swing_path_tilt', 'swing_length',
#     'balls', 'strikes'  # ⬅️ NEW
# ]

# cat_feats = ['pitch_type', 'p_throws']
# all_feats = num_feats + cat_feats

# # Drop rows with missing feature values
# df = df.dropna(subset=all_feats)

# # Encode features and labels
# X_raw = df[all_feats]
# from sklearn.preprocessing import LabelEncoder

# # Encode string labels to integers
# le = LabelEncoder()
# y = le.fit_transform(df['event_class'])


# # Preprocessing
# pre = ColumnTransformer([
#     ('num', StandardScaler(), num_feats),
#     ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
# ])
# X = pre.fit_transform(X_raw)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# # Train classifier
# model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print("\n📊 Classification Report:")
# print(classification_report(y_test, y_pred))
# print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# # Save model
# with open("event_classifier_xgb.pkl", "wb") as f:
#     pickle.dump({'model': model, 'preprocessor': pre, 'label_encoder': le}, f)


# print("✅ Model + preprocessing pipeline saved to event_classifier_xgb.pkl")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pickle

# Load dataset
df = pd.read_csv("ronald_acuna_jr_statcast_pitches_sequential.csv")

# Drop rows missing critical outcome
df = df.dropna(subset=['description', 'events'])

# Add whiff column
df['whiff'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)

# Simplify event_class
def get_event_class(row):
    if row['whiff'] == 1:
        return 'whiff'
    if row['events'] in ['single', 'double', 'triple', 'home_run']:
        return 'hit_safely'
    if row['events'] in ['field_out', 'walk', 'hit_by_pitch']:
        return row['events']
    return np.nan

df['event_class'] = df.apply(get_event_class, axis=1)

# Keep valid classes only
valid_classes = ['field_out', 'walk', 'hit_safely', 'hit_by_pitch', 'whiff']
df = df[df['event_class'].isin(valid_classes)]

# Fill swing metrics
swing_metrics = ['bat_speed', 'attack_angle', 'swing_path_tilt', 'swing_length']
avg_metrics = df[swing_metrics].mean().to_dict()
print("✅ Averages used to fill swing metrics:", avg_metrics)
for metric, avg in avg_metrics.items():
    df[metric] = avg

# Define features
num_feats = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'sz_top', 'sz_bot', 'zone',
    'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
    'arm_angle', 'attack_angle', 'bat_speed', 'swing_path_tilt', 'swing_length',
    'balls', 'strikes'
]

cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment']
all_feats = num_feats + cat_feats

# Drop rows with missing feature values
df = df.dropna(subset=all_feats)

# Prepare features and labels
X_raw = df[all_feats]
le = LabelEncoder()
y = le.fit_transform(df['event_class'])

# Preprocessing pipeline
pre = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
])
X = pre.fit_transform(X_raw)

# Train/test split
X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
    X, y, X_raw, stratify=y, test_size=0.2, random_state=42
)

# Class weighting
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# Grid search parameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', class_weight='balanced')
grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train, sample_weight=sample_weights)

# Final model
model = grid.best_estimator_
print("✅ Best Parameters:", grid.best_params_)

# Evaluate performance
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("event_classifier_xgb.pkl", "wb") as f:
    pickle.dump({'model': model, 'preprocessor': pre, 'label_encoder': le}, f)
print("✅ Model + preprocessing pipeline saved to event_classifier_xgb.pkl")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Feature importance
importances = model.feature_importances_
feat_names = pre.get_feature_names_out()
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(20), importances[sorted_idx][:20])
plt.xticks(range(20), feat_names[sorted_idx][:20], rotation=90)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# 🔍 Test on 10 samples
print("\n🔍 Sample Predictions:")
X_test_sample = X_test[:10]
X_raw_sample = X_raw_test.iloc[:10]
y_true_sample = le.inverse_transform(y_test[:10])
y_pred_sample = le.inverse_transform(model.predict(X_test_sample))

for i in range(10):
    print(f"\n--- Sample {i+1} ---")
    print("Input features:")
    print(X_raw_sample.iloc[i].to_dict())
    print("Expected:", y_true_sample[i])
    print("Predicted:", y_pred_sample[i])

