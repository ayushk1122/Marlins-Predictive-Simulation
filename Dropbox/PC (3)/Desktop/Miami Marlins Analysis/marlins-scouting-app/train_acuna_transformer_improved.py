import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Create holdout dataset (10% for holdout, 90% for training)
print("📊 Creating holdout dataset...")
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.sort_values('game_date')

# Split chronologically - use last 10% of games for holdout
unique_games = df['game_date'].dt.date.unique()
split_idx = int(len(unique_games) * 0.9)  # 90% for training

train_games = unique_games[:split_idx]
holdout_games = unique_games[split_idx:]

train_mask = df['game_date'].dt.date.isin(train_games)
holdout_mask = df['game_date'].dt.date.isin(holdout_games)

df_train = df[train_mask].copy()
df_holdout = df[holdout_mask].copy()

print(f"Training set: {len(df_train)} pitches ({len(train_games)} games)")
print(f"Holdout set: {len(df_holdout)} pitches ({len(holdout_games)} games)")

# Save holdout dataset
df_holdout.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
print("✅ Holdout dataset saved to 'ronald_acuna_jr_holdout_statcast.csv'")

# Use training data for model training
df = df_train

# Drop missing outcomes
df = df.dropna(subset=['description', 'events'])

# Add whiff indicator
df['whiff'] = df['description'].isin(['swinging_strike', 'swinging_strike_blocked']).astype(int)
print("number of whiffs", len(df['whiff']))


# Updated event class logic to include strikeout
def get_event_class(row):
    if row['whiff'] == 1:
        return 'whiff'
    if row['events'] == 'strikeout':
        return 'strikeout'
    if row['events'] in ['single', 'double', 'triple', 'home_run', 'field_out']:
        return 'hit_out_bin'
    if row['events'] in ['walk', 'hit_by_pitch']:
        return row['events']
    return np.nan

df['event_class'] = df.apply(get_event_class, axis=1)
df = df[df['event_class'].notna()]


# Fill missing swing metrics
swing_metrics = ['bat_speed', 'attack_angle', 'swing_path_tilt', 'swing_length',
                 'babip_value', 'launch_speed_angle', 'attack_direction']
for metric in swing_metrics:
    if metric in df.columns:
        df[metric] = df[metric].fillna(df[metric].mean())

# Features
potential_num_feats = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'sz_top', 'sz_bot', 'zone',
    'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
    'arm_angle', 'attack_angle', 'bat_speed', 'swing_path_tilt', 'swing_length',
    'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
    'break_angle_deprecated', 'break_length_deprecated',
    'effective_speed', 'hyper_speed', 'age_pit', 'age_bat'
]
potential_cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
num_feats = [f for f in potential_num_feats if f in df.columns]
cat_feats = [f for f in potential_cat_feats if f in df.columns]
all_feats = num_feats + cat_feats

# Preprocess
X_raw = df[all_feats]
pre = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
])
X = pre.fit_transform(X_raw)

# Encode target
le = LabelEncoder()
y_all = le.fit_transform(df['event_class'])

# Remove hit_out_bin from original model
mask_non_hit_out = df['event_class'] != 'hit_out_bin'
X_orig = X[mask_non_hit_out]
y_orig = y_all[mask_non_hit_out]
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, stratify=y_orig, test_size=0.2, random_state=42)


# Re-label target only for new 4-class model (walk, HBP, whiff, strikeout)
df_4class = df[df['event_class'].isin(['walk', 'hit_by_pitch', 'whiff', 'strikeout'])].copy()
X_orig = pre.transform(df_4class[all_feats])
le_4 = LabelEncoder()
y_orig = le_4.fit_transform(df_4class['event_class'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, stratify=y_orig, test_size=0.2, random_state=42)

# Train the model
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
weights = compute_sample_weight("balanced", y_train)
clf.fit(X_train, y_train, sample_weight=weights)
y_pred = clf.predict(X_test)

print("\n✅ Updated Model (Includes strikeout):")
print(classification_report(y_test, y_pred, target_names=le_4.classes_))

# Save updated model
with open("event_classifier_xgb_with_strikeout.pkl", "wb") as f:
    pickle.dump({'model': clf, 'preprocessor': pre, 'label_encoder': le_4}, f)

# # -------------------------------------------------------
# # 🔁 Two-Stage Hit Outcome Model: Predict field_out vs hit
# # -------------------------------------------------------
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Drop rows without key event fields
df = df.dropna(subset=['description', 'events'])


# Use only fair batted ball events
fair_events = ['single', 'double', 'triple', 'home_run', 'field_out']
df_bin = df[df['events'].isin(fair_events)].copy()
df_bin['target'] = (df_bin['events'].isin(['single', 'double', 'triple', 'home_run'])).astype(int)

# Define impact and input features
impact_feats = ['launch_speed', 'launch_angle', 'hit_distance_sc']
df_bin = df_bin.dropna(subset=impact_feats)

num_feats = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
    'release_pos_x', 'release_pos_y', 'release_pos_z',
    'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'sz_top', 'sz_bot', 'zone',
    'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
    'arm_angle', 'attack_angle', 'bat_speed', 'swing_path_tilt', 'swing_length',
    'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
    'break_angle_deprecated', 'break_length_deprecated',
    'effective_speed', 'hyper_speed', 'age_pit', 'age_bat'
]
cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team']
num_feats = [f for f in num_feats if f in df_bin.columns]
cat_feats = [f for f in cat_feats if f in df_bin.columns]
all_feats = num_feats + cat_feats

# Preprocessing
pre = ColumnTransformer([
    ('num', StandardScaler(), num_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
])
X_raw = df_bin[all_feats]
X_all = pre.fit_transform(X_raw)
y = df_bin['target'].values

# Step 1: Predict launch metrics
predicted_feats = []
for feat in impact_feats:
    y_feat = df_bin[feat].values
    reg = XGBRegressor()
    reg.fit(X_all, y_feat)
    y_pred_full = reg.predict(X_all)
    predicted_feats.append(y_pred_full)

X_launch = np.vstack(predicted_feats).T
X_combined = np.hstack([X_all, X_launch])

# Train/test split
X_train_l, X_test_l, y_train, y_test = train_test_split(X_launch, y, stratify=y, test_size=0.2, random_state=42)
X_train_r, X_test_r, _, _ = train_test_split(X_all, y, stratify=y, test_size=0.2, random_state=42)
X_train_c, X_test_c, _, _ = train_test_split(X_combined, y, stratify=y, test_size=0.2, random_state=42)


# Combined model
clf_combined = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf_combined.fit(X_train_c, y_train)
y_pred_c = clf_combined.predict(X_test_c)

# Evaluation
print("\n📈 Combined Features Model Performance:")
print(classification_report(y_test, y_pred_c, target_names=["field_out", "hit_safely"]))

# Save all models
with open("model_combined.pkl", "wb") as f:
    pickle.dump({'model': clf_combined, 'preprocessor': pre, 'features': all_feats + impact_feats}, f)
