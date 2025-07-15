import pandas as pd
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
 
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
 
 
 
 
 
 
 
 
 
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
 
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
import numpy as np
from collections import Counter

# Load dataset
# (use the full dataset for more context, or holdout for evaluation)
df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")

# Define swing events
swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
df['swing'] = df['description'].isin(swing_events).astype(int)

# Features to analyze
num_features = [
    'zone', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'effective_speed', 'release_extension', 'sz_top', 'sz_bot', 'balls', 'strikes'
]
cat_features = ['pitch_type', 'p_throws', 'stand', 'if_fielding_alignment', 'of_fielding_alignment']

print("\n=== Swing vs No Swing: Feature Strength Analysis ===")

for feat in num_features:
    if feat in df.columns:
        # Bin numericals by quantiles (quintiles if enough unique values, else quartiles)
        unique_vals = df[feat].nunique()
        if unique_vals > 10:
            bins = 5 if unique_vals > 20 else 4
            df[f'{feat}_bin'] = pd.qcut(df[feat], bins, duplicates='drop')
            group_col = f'{feat}_bin'
        else:
            group_col = feat
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(group_col)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top bins/values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top bins/values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3))

for feat in cat_features:
    if feat in df.columns:
        print(f"\nFeature: {feat}")
        swing_rates = df.groupby(feat)['swing'].agg(['count', 'sum', 'mean'])
        swing_rates = swing_rates.rename(columns={'sum': 'num_swings', 'mean': 'swing_rate'})
        print(swing_rates.sort_values('swing_rate', ascending=False))
        print("Top values for swings:")
        print(swing_rates.sort_values('swing_rate', ascending=False).head(3))
        print("Top values for no swings:")
        print(swing_rates.sort_values('swing_rate', ascending=True).head(3)) 
 
 
 
 
 
 
 
 
 
 
 