import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, classification_report, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

# Load the full dataset
csv_path = 'ronald_acuna_jr_statcast_pitches.csv'
data = pd.read_csv(csv_path)

# Features and targets
features = [
    'pitch_type', 'p_throws', 'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'api_break_z_with_gravity', 'api_break_x_batter_in',
    'release_pos_x', 'release_pos_y', 'release_pos_z', 'attack_angle', 'arm_angle', 'balls', 'strikes'
]
reg_targets = [
    'estimated_ba_using_speedangle', 'estimated_slg_using_speedangle', 'estimated_woba_using_speedangle',
    'launch_speed', 'launch_angle', 'hit_distance_sc'
]
clf_target = 'bb_type'

# Only use features that exist in the data
available_features = [f for f in features if f in data.columns]
all_needed = available_features + reg_targets + [clf_target]
data = data.dropna(subset=reg_targets + [clf_target])  # Only drop rows missing targets
X = data[available_features]
y_reg = data[reg_targets]
y_clf = data[clf_target]

# Encode bb_type as integers for XGBoost and MLP
le = LabelEncoder()
y_clf_encoded = le.fit_transform(y_clf)

# Preprocessing
categorical = ['pitch_type', 'p_throws']
numeric = [f for f in available_features if f not in categorical]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numeric)
])

# Train/test split
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, y_clf_train_enc, y_clf_test_enc = train_test_split(
    X, y_reg, y_clf, y_clf_encoded, test_size=0.2, random_state=42
)

# --- Model Comparison (Reference) ---
regressors = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
}
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
}

print('\n--- Regression Model Comparison (Reference) ---')
for name, reg in regressors.items():
    reg_pipe = Pipeline([
        ('pre', preprocessor),
        ('reg', reg)
    ])
    print(f'\nTraining {name}...')
    reg_pipe.fit(X_train, y_reg_train)
    y_reg_pred = reg_pipe.predict(X_test)
    print(f'{name} RMSE (all targets):', mean_squared_error(y_reg_test, y_reg_pred, squared=False))
    print(f'{name} Per-target regression metrics:')
    for i, target in enumerate(reg_targets):
        rmse = mean_squared_error(y_reg_test[target], y_reg_pred[:, i], squared=False)
        r2 = r2_score(y_reg_test[target], y_reg_pred[:, i])
        print(f"  {target}: RMSE={rmse:.4f}, R2={r2:.4f}")

print('\n--- Classification Model Comparison (Reference) ---')
for name, clf in classifiers.items():
    clf_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])
    print(f'\nTraining {name}...')
    clf_pipe.fit(X_train, y_clf_train)
    y_clf_pred = clf_pipe.predict(X_test)
    acc = accuracy_score(y_clf_test, y_clf_pred)
    print(f'{name} accuracy: {acc:.4f}')
    print(f'{name} classification report:')
    print(classification_report(y_clf_test, y_clf_pred))
    # Confusion matrix (string labels)
    cm = confusion_matrix(y_clf_test, y_clf_pred, labels=le.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation='vertical')
    plt.title(f'{name} Confusion Matrix (bb_type)')
    plt.show()

# --- Hyperparameter Tuning for RandomForest (Best Model) ---
print('\n--- Hyperparameter Tuning: RandomForest ---')
from scipy.stats import randint

# Regression tuning
tuned_reg_pipe = Pipeline([
    ('pre', preprocessor),
    ('reg', RandomForestRegressor(random_state=42))
])
reg_param_dist = {
    'reg__n_estimators': randint(100, 400),
    'reg__max_depth': randint(5, 30),
    'reg__min_samples_split': randint(2, 10),
    'reg__min_samples_leaf': randint(1, 8)
}
reg_search = RandomizedSearchCV(tuned_reg_pipe, reg_param_dist, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1)
reg_search.fit(X_train, y_reg_train)
print('Best RandomForestRegressor params:', reg_search.best_params_)
y_reg_pred = reg_search.predict(X_test)
print('Tuned RandomForest RMSE (all targets):', mean_squared_error(y_reg_test, y_reg_pred, squared=False))
print('Tuned RandomForest Per-target regression metrics:')
for i, target in enumerate(reg_targets):
    rmse = mean_squared_error(y_reg_test[target], y_reg_pred[:, i], squared=False)
    r2 = r2_score(y_reg_test[target], y_reg_pred[:, i])
    print(f"  {target}: RMSE={rmse:.4f}, R2={r2:.4f}")

# Classification tuning
tuned_clf_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])
clf_param_dist = {
    'clf__n_estimators': randint(100, 400),
    'clf__max_depth': randint(5, 30),
    'clf__min_samples_split': randint(2, 10),
    'clf__min_samples_leaf': randint(1, 8)
}
clf_search = RandomizedSearchCV(tuned_clf_pipe, clf_param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)
clf_search.fit(X_train, y_clf_train)
print('Best RandomForestClassifier params:', clf_search.best_params_)
y_clf_pred = clf_search.predict(X_test)
acc = accuracy_score(y_clf_test, y_clf_pred)
print('Tuned RandomForest accuracy:', f'{acc:.4f}')
print('Tuned RandomForest classification report:')
print(classification_report(y_clf_test, y_clf_pred))
cm = confusion_matrix(y_clf_test, y_clf_pred, labels=le.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation='vertical')
plt.title('Tuned RandomForest Confusion Matrix (bb_type)')
plt.show()