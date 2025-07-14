import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
 
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 
 
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
 
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

def create_holdout_dataset():
    """
    Create a holdout dataset for model evaluation.
    Splits data chronologically to maintain temporal order.
    """
    print("📊 Creating Holdout Dataset for Model Evaluation")
    print("=" * 50)
    
    # Load the complete dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    print(f"Original dataset size: {len(df)} pitches")
    
    # Sort by date to maintain chronological order
    df['game_date'] = pd.to_datetime(df['game_date'])
    df = df.sort_values('game_date')
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    # Remove rows with missing critical data
    df_clean = df.dropna(subset=['description', 'events', 'plate_x', 'plate_z'])
    print(f"After removing missing data: {len(df_clean)} pitches")
    
    # Split chronologically (80% training, 20% holdout)
    # Use the last 20% of games for holdout
    unique_games = df_clean['game_date'].dt.date.unique()
    split_idx = int(len(unique_games) * 0.8)
    
    train_games = unique_games[:split_idx]
    holdout_games = unique_games[split_idx:]
    
    print(f"Training games: {len(train_games)} (first {split_idx} games)")
    print(f"Holdout games: {len(holdout_games)} (last {len(unique_games) - split_idx} games)")
    
    # Split the data
    train_mask = df_clean['game_date'].dt.date.isin(train_games)
    holdout_mask = df_clean['game_date'].dt.date.isin(holdout_games)
    
    train_df = df_clean[train_mask].copy()
    holdout_df = df_clean[holdout_mask].copy()
    
    print(f"\nTraining set: {len(train_df)} pitches")
    print(f"Holdout set: {len(holdout_df)} pitches")
    
    # Save the datasets
    train_df.to_csv("ronald_acuna_jr_training_statcast.csv", index=False)
    holdout_df.to_csv("ronald_acuna_jr_holdout_statcast.csv", index=False)
    
    # Create summary statistics
    summary = {
        'total_pitches': len(df_clean),
        'training_pitches': len(train_df),
        'holdout_pitches': len(holdout_df),
        'training_games': len(train_games),
        'holdout_games': len(holdout_games),
        'date_range': {
            'training_start': train_df['game_date'].min().strftime('%Y-%m-%d'),
            'training_end': train_df['game_date'].max().strftime('%Y-%m-%d'),
            'holdout_start': holdout_df['game_date'].min().strftime('%Y-%m-%d'),
            'holdout_end': holdout_df['game_date'].max().strftime('%Y-%m-%d')
        },
        'description_distribution': {
            'training': train_df['description'].value_counts().to_dict(),
            'holdout': holdout_df['description'].value_counts().to_dict()
        },
        'events_distribution': {
            'training': train_df['events'].value_counts().to_dict(),
            'holdout': holdout_df['events'].value_counts().to_dict()
        }
    }
    
    # Save summary
    with open("holdout_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ Datasets created successfully!")
    print("📁 Files created:")
    print("  - ronald_acuna_jr_training_statcast.csv (training data)")
    print("  - ronald_acuna_jr_holdout_statcast.csv (holdout data)")
    print("  - holdout_dataset_summary.json (summary statistics)")
    
    # Print some key statistics
    print(f"\n📈 Key Statistics:")
    print(f"  Training set: {len(train_df)} pitches ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"  Holdout set: {len(holdout_df)} pitches ({len(holdout_df)/len(df_clean)*100:.1f}%)")
    
    # Show distribution of key events
    print(f"\n🎯 Swing/No-Swing Distribution:")
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    train_swings = train_df['description'].isin(swing_events).sum()
    train_no_swings = train_df['description'].isin(no_swing_events).sum()
    holdout_swings = holdout_df['description'].isin(swing_events).sum()
    holdout_no_swings = holdout_df['description'].isin(no_swing_events).sum()
    
    print(f"  Training - Swings: {train_swings}, No-swings: {train_no_swings}")
    print(f"  Holdout - Swings: {holdout_swings}, No-swings: {holdout_no_swings}")
    
    return train_df, holdout_df

def update_training_script():
    """
    Update the training script to use the new training dataset.
    """
    print("\n🔄 Updating training script to use holdout dataset...")
    
    # Read the current training script
    with open("train_sequential_models.py", "r") as f:
        content = f.read()
    
    # Replace the dataset loading line
    old_line = 'df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")'
    new_line = 'df = pd.read_csv("ronald_acuna_jr_training_statcast.csv")'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        
        # Write the updated script
        with open("train_sequential_models.py", "w") as f:
            f.write(content)
        
        print("✅ Training script updated to use training dataset")
    else:
        print("⚠️  Could not find the dataset loading line in training script")

def main():
    """
    Main function to create holdout dataset and update training script.
    """
    # Create the holdout dataset
    train_df, holdout_df = create_holdout_dataset()
    
    # Update the training script
    update_training_script()
    
    print("\n🎯 Next Steps:")
    print("1. Run 'python train_sequential_models.py' to train models on training data")
    print("2. Use 'ronald_acuna_jr_holdout_statcast.csv' for final model evaluation")
    print("3. Check 'holdout_dataset_summary.json' for detailed statistics")

if __name__ == "__main__":
    main() 
 
 
 
 
 
 
 
 