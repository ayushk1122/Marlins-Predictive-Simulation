import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import json

def optimize_swing_classifier(df):
    """
    Optimize the swing classifier hyperparameters for maximum accuracy
    """
    print("🔧 OPTIMIZING SWING CLASSIFIER FOR 70-80% ACCURACY")
    print("=" * 60)
    
    # Prepare features
    from train_sequential_models import prepare_features
    df, num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    # Preprocess
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    
    X = preprocessor.fit_transform(df[all_feats])
    y = df['swing'].values
    
    # Clean NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Calculate class weights
    swing_count = y_train.sum()
    no_swing_count = len(y_train) - swing_count
    scale_pos_weight = no_swing_count / swing_count
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class balance - Swings: {swing_count}, No-swings: {no_swing_count}")
    print(f"Scale pos weight: {scale_pos_weight:.3f}")
    
    # Define parameter grids for optimization
    xgb_param_grid = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0]
    }
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Optimize XGBoost
    print("\n🔍 Optimizing XGBoost...")
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_param_grid, n_iter=50, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"Best XGBoost accuracy: {xgb_search.best_score_:.4f}")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    
    # Optimize RandomForest
    print("\n🔍 Optimizing RandomForest...")
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )
    
    rf_search = RandomizedSearchCV(
        rf_model, rf_param_grid, n_iter=30, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    rf_search.fit(X_train, y_train)
    
    print(f"Best RandomForest accuracy: {rf_search.best_score_:.4f}")
    print(f"Best RandomForest params: {rf_search.best_params_}")
    
    # Create optimized ensemble
    best_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        **xgb_search.best_params_
    )
    
    best_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        **rf_search.best_params_
    )
    
    # Test individual models
    print("\n📊 Testing Individual Models:")
    
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"RandomForest Test Accuracy: {rf_acc:.4f}")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('rf', best_rf)
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
    # Detailed ensemble analysis
    from sklearn.metrics import classification_report
    print(f"\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['No Swing', 'Swing']))
    
    # Save optimized model
    optimized_models = {
        'xgb_model': best_xgb,
        'rf_model': best_rf,
        'ensemble_model': ensemble,
        'preprocessor': preprocessor,
        'features': all_feats,
        'xgb_params': xgb_search.best_params_,
        'rf_params': rf_search.best_params_,
        'test_accuracy': ensemble_acc
    }
    
    with open("optimized_sequential_models.pkl", "wb") as f:
        pickle.dump(optimized_models, f)
    
    print(f"\n✅ Optimized model saved with {ensemble_acc:.4f} accuracy")
    
    return optimized_models

def main():
    # Load dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    df = df.dropna(subset=['description', 'events'])
    
    # Optimize the model
    optimized_models = optimize_swing_classifier(df)
    
    print(f"\n🎯 Target achieved: {optimized_models['test_accuracy']:.4f} accuracy")
    if optimized_models['test_accuracy'] >= 0.70:
        print("🎉 SUCCESS: Model achieved 70%+ accuracy!")
    else:
        print("📈 Model improved but needs more optimization")

if __name__ == "__main__":
    main() 
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import json

def optimize_swing_classifier(df):
    """
    Optimize the swing classifier hyperparameters for maximum accuracy
    """
    print("🔧 OPTIMIZING SWING CLASSIFIER FOR 70-80% ACCURACY")
    print("=" * 60)
    
    # Prepare features
    from train_sequential_models import prepare_features
    df, num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    # Preprocess
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    
    X = preprocessor.fit_transform(df[all_feats])
    y = df['swing'].values
    
    # Clean NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Calculate class weights
    swing_count = y_train.sum()
    no_swing_count = len(y_train) - swing_count
    scale_pos_weight = no_swing_count / swing_count
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class balance - Swings: {swing_count}, No-swings: {no_swing_count}")
    print(f"Scale pos weight: {scale_pos_weight:.3f}")
    
    # Define parameter grids for optimization
    xgb_param_grid = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0]
    }
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Optimize XGBoost
    print("\n🔍 Optimizing XGBoost...")
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_param_grid, n_iter=50, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"Best XGBoost accuracy: {xgb_search.best_score_:.4f}")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    
    # Optimize RandomForest
    print("\n🔍 Optimizing RandomForest...")
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )
    
    rf_search = RandomizedSearchCV(
        rf_model, rf_param_grid, n_iter=30, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    rf_search.fit(X_train, y_train)
    
    print(f"Best RandomForest accuracy: {rf_search.best_score_:.4f}")
    print(f"Best RandomForest params: {rf_search.best_params_}")
    
    # Create optimized ensemble
    best_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        **xgb_search.best_params_
    )
    
    best_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        **rf_search.best_params_
    )
    
    # Test individual models
    print("\n📊 Testing Individual Models:")
    
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"RandomForest Test Accuracy: {rf_acc:.4f}")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('rf', best_rf)
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
    # Detailed ensemble analysis
    from sklearn.metrics import classification_report
    print(f"\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['No Swing', 'Swing']))
    
    # Save optimized model
    optimized_models = {
        'xgb_model': best_xgb,
        'rf_model': best_rf,
        'ensemble_model': ensemble,
        'preprocessor': preprocessor,
        'features': all_feats,
        'xgb_params': xgb_search.best_params_,
        'rf_params': rf_search.best_params_,
        'test_accuracy': ensemble_acc
    }
    
    with open("optimized_sequential_models.pkl", "wb") as f:
        pickle.dump(optimized_models, f)
    
    print(f"\n✅ Optimized model saved with {ensemble_acc:.4f} accuracy")
    
    return optimized_models

def main():
    # Load dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    df = df.dropna(subset=['description', 'events'])
    
    # Optimize the model
    optimized_models = optimize_swing_classifier(df)
    
    print(f"\n🎯 Target achieved: {optimized_models['test_accuracy']:.4f} accuracy")
    if optimized_models['test_accuracy'] >= 0.70:
        print("🎉 SUCCESS: Model achieved 70%+ accuracy!")
    else:
        print("📈 Model improved but needs more optimization")

if __name__ == "__main__":
    main() 
 
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import json

def optimize_swing_classifier(df):
    """
    Optimize the swing classifier hyperparameters for maximum accuracy
    """
    print("🔧 OPTIMIZING SWING CLASSIFIER FOR 70-80% ACCURACY")
    print("=" * 60)
    
    # Prepare features
    from train_sequential_models import prepare_features
    df, num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    # Preprocess
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    
    X = preprocessor.fit_transform(df[all_feats])
    y = df['swing'].values
    
    # Clean NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Calculate class weights
    swing_count = y_train.sum()
    no_swing_count = len(y_train) - swing_count
    scale_pos_weight = no_swing_count / swing_count
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class balance - Swings: {swing_count}, No-swings: {no_swing_count}")
    print(f"Scale pos weight: {scale_pos_weight:.3f}")
    
    # Define parameter grids for optimization
    xgb_param_grid = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0]
    }
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Optimize XGBoost
    print("\n🔍 Optimizing XGBoost...")
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_param_grid, n_iter=50, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"Best XGBoost accuracy: {xgb_search.best_score_:.4f}")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    
    # Optimize RandomForest
    print("\n🔍 Optimizing RandomForest...")
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )
    
    rf_search = RandomizedSearchCV(
        rf_model, rf_param_grid, n_iter=30, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    rf_search.fit(X_train, y_train)
    
    print(f"Best RandomForest accuracy: {rf_search.best_score_:.4f}")
    print(f"Best RandomForest params: {rf_search.best_params_}")
    
    # Create optimized ensemble
    best_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        **xgb_search.best_params_
    )
    
    best_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        **rf_search.best_params_
    )
    
    # Test individual models
    print("\n📊 Testing Individual Models:")
    
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"RandomForest Test Accuracy: {rf_acc:.4f}")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('rf', best_rf)
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
    # Detailed ensemble analysis
    from sklearn.metrics import classification_report
    print(f"\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['No Swing', 'Swing']))
    
    # Save optimized model
    optimized_models = {
        'xgb_model': best_xgb,
        'rf_model': best_rf,
        'ensemble_model': ensemble,
        'preprocessor': preprocessor,
        'features': all_feats,
        'xgb_params': xgb_search.best_params_,
        'rf_params': rf_search.best_params_,
        'test_accuracy': ensemble_acc
    }
    
    with open("optimized_sequential_models.pkl", "wb") as f:
        pickle.dump(optimized_models, f)
    
    print(f"\n✅ Optimized model saved with {ensemble_acc:.4f} accuracy")
    
    return optimized_models

def main():
    # Load dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    df = df.dropna(subset=['description', 'events'])
    
    # Optimize the model
    optimized_models = optimize_swing_classifier(df)
    
    print(f"\n🎯 Target achieved: {optimized_models['test_accuracy']:.4f} accuracy")
    if optimized_models['test_accuracy'] >= 0.70:
        print("🎉 SUCCESS: Model achieved 70%+ accuracy!")
    else:
        print("📈 Model improved but needs more optimization")

if __name__ == "__main__":
    main() 
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle
import json

def optimize_swing_classifier(df):
    """
    Optimize the swing classifier hyperparameters for maximum accuracy
    """
    print("🔧 OPTIMIZING SWING CLASSIFIER FOR 70-80% ACCURACY")
    print("=" * 60)
    
    # Prepare features
    from train_sequential_models import prepare_features
    df, num_feats, cat_feats = prepare_features(df)
    all_feats = num_feats + cat_feats
    
    # Create swing/no-swing target
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df['swing'] = df['description'].isin(swing_events).astype(int)
    
    # Preprocess
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    
    X = preprocessor.fit_transform(df[all_feats])
    y = df['swing'].values
    
    # Clean NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # Calculate class weights
    swing_count = y_train.sum()
    no_swing_count = len(y_train) - swing_count
    scale_pos_weight = no_swing_count / swing_count
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class balance - Swings: {swing_count}, No-swings: {no_swing_count}")
    print(f"Scale pos weight: {scale_pos_weight:.3f}")
    
    # Define parameter grids for optimization
    xgb_param_grid = {
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0, 5.0]
    }
    
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Optimize XGBoost
    print("\n🔍 Optimizing XGBoost...")
    xgb_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_model, xgb_param_grid, n_iter=50, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    xgb_search.fit(X_train, y_train)
    
    print(f"Best XGBoost accuracy: {xgb_search.best_score_:.4f}")
    print(f"Best XGBoost params: {xgb_search.best_params_}")
    
    # Optimize RandomForest
    print("\n🔍 Optimizing RandomForest...")
    rf_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    )
    
    rf_search = RandomizedSearchCV(
        rf_model, rf_param_grid, n_iter=30, cv=3, scoring='accuracy',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    rf_search.fit(X_train, y_train)
    
    print(f"Best RandomForest accuracy: {rf_search.best_score_:.4f}")
    print(f"Best RandomForest params: {rf_search.best_params_}")
    
    # Create optimized ensemble
    best_xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        **xgb_search.best_params_
    )
    
    best_rf = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        **rf_search.best_params_
    )
    
    # Test individual models
    print("\n📊 Testing Individual Models:")
    
    best_xgb.fit(X_train, y_train)
    xgb_pred = best_xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost Test Accuracy: {xgb_acc:.4f}")
    
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"RandomForest Test Accuracy: {rf_acc:.4f}")
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_xgb),
            ('rf', best_rf)
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")
    
    # Detailed ensemble analysis
    from sklearn.metrics import classification_report
    print(f"\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['No Swing', 'Swing']))
    
    # Save optimized model
    optimized_models = {
        'xgb_model': best_xgb,
        'rf_model': best_rf,
        'ensemble_model': ensemble,
        'preprocessor': preprocessor,
        'features': all_feats,
        'xgb_params': xgb_search.best_params_,
        'rf_params': rf_search.best_params_,
        'test_accuracy': ensemble_acc
    }
    
    with open("optimized_sequential_models.pkl", "wb") as f:
        pickle.dump(optimized_models, f)
    
    print(f"\n✅ Optimized model saved with {ensemble_acc:.4f} accuracy")
    
    return optimized_models

def main():
    # Load dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    df = df.dropna(subset=['description', 'events'])
    
    # Optimize the model
    optimized_models = optimize_swing_classifier(df)
    
    print(f"\n🎯 Target achieved: {optimized_models['test_accuracy']:.4f} accuracy")
    if optimized_models['test_accuracy'] >= 0.70:
        print("🎉 SUCCESS: Model achieved 70%+ accuracy!")
    else:
        print("📈 Model improved but needs more optimization")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 