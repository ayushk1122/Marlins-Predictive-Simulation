import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap

# For Savant data
from pybaseball import statcast_batter

# Load your pitch-level data
df = pd.read_csv("ronald_acuna_jr_statcast_pitches_sequential.csv")
df = df.dropna(subset=['launch_angle'])

# Add Savant bat metrics (bat speed, attack angle, swing tilt, swing length)
batter_id = 660670
# Pull year-range you're training on:
sv = statcast_batter(start_dt='2023-01-01', end_dt='2025-06-30', player_id=batter_id)
sv = sv[['game_date','bat_speed','attack_angle','swing_path_tilt','swing_length']]
# Merge on date
df['game_date'] = pd.to_datetime(df['game_date'])
sv['game_date'] = pd.to_datetime(sv['game_date'])
df = df.merge(sv, on='game_date', how='left').dropna(subset=['bat_speed'])

# Define inputs + output
num_feats = [
  'release_speed','release_spin_rate','spin_axis','release_extension',
  'release_pos_x','release_pos_y','release_pos_z','pfx_x','pfx_z',
  'plate_x','plate_z','api_break_z_with_gravity','api_break_x_batter_in',
  'arm_angle','attack_angle','bat_speed','swing_path_tilt','swing_length'
]
cat_feats = ['pitch_type','p_throws']
X_raw = df[num_feats + cat_feats]
y = df['launch_angle']

# Preprocessing
pre = ColumnTransformer([
  ('num', StandardScaler(), num_feats),
  ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
])
X = pre.fit_transform(X_raw)

# Train/test
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RF regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Test MAE:", np.mean(np.abs(rf.predict(X_test) - y_test)))
print("Test R²:", rf.score(X_test, y_test))

# Feature importance
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test[:200])
shap.summary_plot(shap_values, features=pre.transformers_[0][2] + list(pre.named_transformers_['cat'].get_feature_names_out()), show=False)

# Save model + preprocessor
import pickle
pickle.dump({'model': rf, 'preprocessor': pre}, open("launch_angle_rf.pkl", "wb"))

print("\n✅ Model + preprocessing saved to launch_angle_rf.pkl")
