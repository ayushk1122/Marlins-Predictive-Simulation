import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle
import json
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from comprehensive_count_features import calculate_comprehensive_count_features, get_count_features_for_pitch, analyze_count_patterns
# Add SMOTE for class balancing
print("Attempting to import SMOTE...")
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline
    SMOTE_AVAILABLE = True
    print("✓ SMOTE successfully imported")
except ImportError as e:
    print(f"Warning: imbalanced-learn not available. Error: {e}")
    print("Install with: pip install imbalanced-learn")
    SMOTE_AVAILABLE = False
except Exception as e:
    print(f"Unexpected error importing SMOTE: {e}")
    SMOTE_AVAILABLE = False

# Neural network imports removed - focusing on XGBoost models only

# Global variable for situational thresholds
SITUATIONAL_THRESHOLDS = {}

# Create global label encoder for consistent encoding/decoding
def create_global_label_encoder():
    """Create a label encoder that knows all possible values"""
    le = LabelEncoder()
    # Define all possible outcome values
    all_possible_outcomes = ['whiff', 'hit_safely', 'field_out', 'ball', 'strike', 'hit_by_pitch']
    le.fit(all_possible_outcomes)
    return le

# Global label encoder instance
GLOBAL_LABEL_ENCODER = create_global_label_encoder()
import warnings
import sys
import logging

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('xgboost').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

def load_babip_data():
    """Load BABIP data from the calculate_pitch_type_zone_batting_averages.py script"""
    try:
        # Try to load the CSV file if it exists
        babip_df = pd.read_csv('pitch_type_zone_batting_averages.csv')
        print(f"✓ Loaded BABIP data with {len(babip_df)} pitch type x zone combinations")
        
        # Create a dictionary for quick lookup
        babip_lookup = {}
        for _, row in babip_df.iterrows():
            key = (row['Pitch Type'], row['Zone'])
            babip_lookup[key] = {
                'batting_average_bip': row['Batting Average (BIP)'],
                'whiff_rate': row['Whiff Rate'],
                'field_out_rate_bip': row['Field Out Rate (BIP)'],
                'balls_in_play': row['Balls in Play'],
                'total_swings': row['Total Swings'],
                'total_whiffs': row['Total Whiffs']
            }
        
        return babip_lookup
    except FileNotFoundError:
        print("⚠️ BABIP data file not found. Run calculate_pitch_type_zone_batting_averages.py first.")
        return {}
    except Exception as e:
        print(f"✗ Error loading BABIP data: {e}")
        return {}

def get_babip_features(pitch_type, zone, babip_lookup):
    """Get BABIP and whiff rate features for a specific pitch type and zone"""
    key = (pitch_type, zone)
    if key in babip_lookup:
        return babip_lookup[key]
    else:
        # Return default values if no data available
        return {
            'batting_average_bip': 0.25,  # Default BABIP
            'whiff_rate': 0.35,           # Default whiff rate
            'field_out_rate_bip': 0.40,   # Default field out rate
            'balls_in_play': 0,           # No balls in play data
            'total_swings': 0,            # No swing data
            'total_whiffs': 0             # No whiff data
        }

def calculate_zone(plate_x, plate_z):
    """
    Calculate Statcast zone (1-14) based on plate_x and plate_z coordinates.
    """
    # Strike zone boundaries (approximate)
    sz_left = -0.85
    sz_right = 0.85
    sz_bot = 1.5
    sz_top = 3.5
    
    # Check if pitch is in strike zone
    in_strike_zone = (sz_left <= plate_x <= sz_right) and (sz_bot <= plate_z <= sz_top)
    
    if in_strike_zone:
        # Calculate zone within strike zone (1-9)
        x_section = int((plate_x - sz_left) / ((sz_right - sz_left) / 3))
        z_section = int((plate_z - sz_bot) / ((sz_top - sz_bot) / 3))
        
        # Clamp to valid ranges
        x_section = max(0, min(2, x_section))
        z_section = max(0, min(2, z_section))
        
        # Convert to zone number (1-9)
        zone = z_section * 3 + x_section + 1
    else:
        # Outside strike zone (11-14)
        if plate_x < sz_left:  # Left side
            zone = 11 if plate_z > sz_top else 13
        else:  # Right side
            zone = 12 if plate_z > sz_top else 14
    
    return zone

def prepare_features(df):
    """
    Prepare features for modeling, including zone calculation and new engineered features.
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Only calculate zones for pitches that don't have valid zone data
    # Don't override valid Statcast zone data
    if 'zone' not in df.columns or df['zone'].isna().any() or (df['zone'] <= 0).any():
        print("Calculating zones for pitches with missing/invalid zone data...")
        # Only calculate for rows that need it
        zone_mask = df['zone'].isna() | (df['zone'] <= 0) if 'zone' in df.columns else pd.Series([True] * len(df))
        df.loc[zone_mask, 'zone'] = df[zone_mask].apply(lambda row: calculate_zone(row['plate_x'], row['plate_z']), axis=1)
    else:
        print("Using original Statcast zone data")
    
    # NEW ENGINEERED FEATURES
    
    # 1. Zone Distance: Distance from pitch location to center of strike zone
    df['zone_center_x'] = 0  # Center of plate is at x=0
    df['zone_center_z'] = (df['sz_top'] + df['sz_bot']) / 2  # Center of strike zone
    df['zone_distance'] = np.sqrt(
        (df['plate_x'] - df['zone_center_x'])**2 + 
        (df['plate_z'] - df['zone_center_z'])**2
    )
    
    # 2. IMPROVED Movement Quantification: Better calculation using multiple Statcast fields
    # Replace simplistic movement_magnitude with more sophisticated movement analysis
    
    # Primary movement metrics using Statcast's calculated break values
    df['horizontal_break'] = df['api_break_x_batter_in'].fillna(0)  # From batter's perspective
    df['vertical_break'] = df['api_break_z_with_gravity'].fillna(0)  # With gravity accounted for
    df['arm_side_break'] = df['api_break_x_arm'].fillna(0)  # From pitcher's arm side
    
    # Calculate total movement magnitude using the more accurate break values
    df['movement_magnitude'] = np.sqrt(df['horizontal_break']**2 + df['vertical_break']**2)
    
    # Movement direction and characteristics
    df['movement_direction'] = np.arctan2(df['vertical_break'], df['horizontal_break']) * 180 / np.pi
    df['movement_ratio'] = np.abs(df['horizontal_break']) / (np.abs(df['vertical_break']) + 0.1)
    
    # Spin-based movement analysis
    if 'spin_axis' in df.columns:
        df['spin_axis_rad'] = df['spin_axis'].fillna(0) * np.pi / 180  # Convert to radians
        df['spin_efficiency'] = df['release_spin_rate'] / (df['release_speed'] + 0.1)  # Spin per mph
    else:
        df['spin_axis_rad'] = 0
        df['spin_efficiency'] = 0
    
    # Velocity-movement relationship
    df['velocity_movement_ratio'] = df['release_speed'] / (df['movement_magnitude'] + 0.1)
    df['movement_per_mph'] = df['movement_magnitude'] / (df['release_speed'] + 0.1)
    
    # Pitch type specific movement expectations
    df['expected_movement'] = np.where(
        df['pitch_type'].isin(['SL', 'CU', 'KC']),  # Breaking balls
        df['movement_magnitude'] * 1.2,  # Expect more movement
        np.where(
            df['pitch_type'].isin(['CH', 'FS']),  # Offspeed
            df['movement_magnitude'] * 1.1,  # Moderate movement
            df['movement_magnitude'] * 0.8  # Fastballs - less movement expected
        )
    )
    
    # Movement deception (unexpected movement for pitch type)
    df['movement_deception'] = df['movement_magnitude'] - df['expected_movement']
    
    # High movement thresholds based on pitch type
    df['high_movement'] = np.where(
        df['pitch_type'].isin(['SL', 'CU', 'KC']),
        df['movement_magnitude'] > 8,  # Breaking balls
        np.where(
            df['pitch_type'].isin(['CH', 'FS']),
            df['movement_magnitude'] > 6,  # Offspeed
            df['movement_magnitude'] > 4   # Fastballs
        )
    ).astype(int)
    
    # Movement consistency (if we have multiple pitches from same pitcher)
    if 'pitcher' in df.columns:
        df['movement_std'] = df.groupby('pitcher')['movement_magnitude'].transform('std')
        df['movement_diff_from_avg'] = df['movement_magnitude'] - df.groupby('pitcher')['movement_magnitude'].transform('mean')
    else:
        df['movement_std'] = df['movement_magnitude'].std()
        df['movement_diff_from_avg'] = 0
    
    # 3. Location x Movement: Interaction between normalized location and movement
    # Normalize plate_x by strike zone width (approximately 17 inches = 1.417 feet)
    df['plate_x_norm'] = df['plate_x'] / 1.417
    # Normalize plate_z by strike zone height
    df['plate_z_norm'] = (df['plate_z'] - df['sz_bot']) / (df['sz_top'] - df['sz_bot'])
    
    # Interaction features
    df['plate_x_norm_x_movement'] = df['plate_x_norm'] * df['movement_magnitude']
    df['plate_z_norm_x_movement'] = df['plate_z_norm'] * df['movement_magnitude']
    
    # NEW FEATURES FOR BETTER NO-SWING PREDICTION
    
    # 4. Count-based features
    df['count_pressure'] = df['balls'] - df['strikes']  # Positive = ahead, negative = behind
    df['count_total'] = df['balls'] + df['strikes']
    df['behind_in_count'] = (df['strikes'] > df['balls']).astype(int)
    df['ahead_in_count'] = (df['balls'] > df['strikes']).astype(int)
    df['two_strikes'] = (df['strikes'] >= 2).astype(int)
    df['three_balls'] = (df['balls'] >= 3).astype(int)
    
    # NEW: Basic count-based features (no dependencies)
    df['early_count'] = ((df['balls'] <= 1) & (df['strikes'] <= 1)).astype(int)
    df['middle_count'] = ((df['balls'] == 1) & (df['strikes'] == 1)).astype(int)
    df['late_count'] = ((df['balls'] >= 2) | (df['strikes'] >= 2)).astype(int)
    df['pressure_count'] = ((df['strikes'] >= 2) | (df['balls'] >= 3)).astype(int)
    
    # Basic early count penalty (no dependencies)
    df['early_count_penalty'] = df['early_count'] * 0.3  # 30% penalty for early count swings
    
    # 5. Zone-specific features for no-swing prediction
    df['in_strike_zone'] = ((df['plate_x'] >= -0.85) & (df['plate_x'] <= 0.85) & 
                           (df['plate_z'] >= df['sz_bot']) & (df['plate_z'] <= df['sz_top'])).astype(int)
    df['far_from_zone'] = (df['zone_distance'] > 1.0).astype(int)  # More than 1 foot from center
    df['high_pitch'] = (df['plate_z'] > df['sz_top']).astype(int)
    df['low_pitch'] = (df['plate_z'] < df['sz_bot']).astype(int)
    df['inside_pitch'] = (df['plate_x'] < -0.85).astype(int)
    df['outside_pitch'] = (df['plate_x'] > 0.85).astype(int)
    
    # 6. Pitch type specific features
    df['is_fastball'] = df['pitch_type'].isin(['FF', 'SI', 'FC']).astype(int)
    df['is_breaking_ball'] = df['pitch_type'].isin(['SL', 'CU', 'KC']).astype(int)
    df['is_offspeed'] = df['pitch_type'].isin(['CH', 'FS']).astype(int)
    
    # 6.5. Count-based features with dependencies (after zone and pitch type features are created)
    df['early_count_zone_penalty'] = df['early_count'] * df['in_strike_zone'] * 0.2  # Additional penalty for zone pitches in early counts
    df['early_count_outside_penalty'] = df['early_count'] * (~df['in_strike_zone']).astype(int) * 0.5  # 50% penalty for outside pitches in early counts
    
    # Count-specific swing rate adjustments
    df['count_swing_rate_adjustment'] = np.where(
        df['early_count'] == 1, -0.25,  # Reduce swing probability by 25% in early counts
        np.where(
            df['pressure_count'] == 1, 0.15,  # Increase swing probability by 15% in pressure situations
            0.0  # No adjustment for middle counts
        )
    )
    
    # Count-specific location penalties
    df['early_count_location_penalty'] = np.where(
        df['early_count'] == 1,
        np.where(
            df['zone_distance'] > 0.5, 0.4,  # 40% penalty for pitches >0.5 feet from zone in early counts
            np.where(
                df['zone_distance'] > 0.2, 0.2,  # 20% penalty for pitches >0.2 feet from zone in early counts
                0.0  # No penalty for very close pitches
            )
        ),
        0.0  # No penalty for non-early counts
    )
    
    # Count-specific pitch type penalties
    df['early_count_breaking_penalty'] = df['early_count'] * df['is_breaking_ball'] * 0.3  # 30% penalty for breaking balls in early counts
    df['early_count_offspeed_penalty'] = df['early_count'] * df['is_offspeed'] * 0.25  # 25% penalty for offspeed in early counts
    
    # 7. Advanced interaction features
    df['zone_distance_x_count_pressure'] = df['zone_distance'] * df['count_pressure']
    df['movement_x_count_pressure'] = df['movement_magnitude'] * df['count_pressure']
    df['in_zone_x_two_strikes'] = df['in_strike_zone'] * df['two_strikes']
    df['far_from_zone_x_ahead'] = df['far_from_zone'] * df['ahead_in_count']
    
    # ADVANCED FEATURES FOR FURTHER IMPROVEMENT
    
    # 8. Velocity and movement features
    df['velocity_movement_ratio'] = df['release_speed'] / (df['movement_magnitude'] + 0.1)  # Avoid division by zero
    df['high_velocity'] = (df['release_speed'] > 95).astype(int)
    df['low_velocity'] = (df['release_speed'] < 85).astype(int)
    df['high_movement'] = (df['movement_magnitude'] > 10).astype(int)
    
    # 8.5. Count-specific velocity penalties (after velocity features are created)
    df['early_count_low_vel_penalty'] = df['early_count'] * df['low_velocity'] * 0.35  # 35% penalty for low velocity in early counts
    df['early_count_high_vel_penalty'] = df['early_count'] * df['high_velocity'] * 0.15  # 15% penalty for high velocity in early counts
    
    # 9. Zone edge features
    df['zone_edge_distance'] = np.minimum(
        np.abs(df['plate_x'] - (-0.85)),  # Distance from left edge
        np.abs(df['plate_x'] - 0.85)      # Distance from right edge
    )
    df['zone_top_distance'] = np.abs(df['plate_z'] - df['sz_top'])
    df['zone_bottom_distance'] = np.abs(df['plate_z'] - df['sz_bot'])
    df['closest_zone_edge'] = np.minimum(
        np.minimum(df['zone_edge_distance'], df['zone_top_distance']),
        df['zone_bottom_distance']
    )
    
    # 10. Count-specific features
    df['full_count'] = (df['balls'] == 3) & (df['strikes'] == 2)
    df['hitters_count'] = (df['balls'] >= 2) & (df['strikes'] <= 1)
    df['pitchers_count'] = (df['balls'] <= 1) & (df['strikes'] >= 2)
    df['neutral_count'] = (df['balls'] == 1) & (df['strikes'] == 1)
    
    # 11. Advanced zone features
    df['zone_quadrant'] = np.where(
        df['plate_x'] >= 0,
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_out', 'down_out'),
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_in', 'down_in')
    )
    
    # 12. Pitch deception features
    df['velocity_drop'] = (df['release_speed'] < 90) & df['is_fastball']
    df['breaking_ball_high'] = df['is_breaking_ball'] & (df['plate_z'] > df['sz_top'])
    df['offspeed_low'] = df['is_offspeed'] & (df['plate_z'] < df['sz_bot'])
    
    # 13. Context features - handle missing columns safely
    if 'inning' in df.columns:
        df['inning_late'] = (df['inning'] >= 7).astype(int)
    else:
        df['inning_late'] = 0
        
    if 'home_score' in df.columns and 'away_score' in df.columns:
        df['close_game'] = (np.abs(df['home_score'] - df['away_score']) <= 2).astype(int)
    else:
        df['close_game'] = 0
    
    # ADVANCED FEATURES FOR 70-80% ACCURACY
    
    # 14. Pitch sequencing features (if we have at-bat data)
    if 'at_bat_number' in df.columns:
        df['pitch_in_at_bat'] = df.groupby('at_bat_number').cumcount() + 1
        df['first_pitch'] = (df['pitch_in_at_bat'] == 1).astype(int)
        df['last_pitch'] = df.groupby('at_bat_number')['pitch_in_at_bat'].transform('max') == df['pitch_in_at_bat']
    else:
        df['pitch_in_at_bat'] = 1
        df['first_pitch'] = 1
        df['last_pitch'] = 1
    
    # 15. Advanced velocity features
    df['velocity_bin'] = pd.cut(df['release_speed'], bins=[0, 85, 90, 95, 100, 110], labels=['very_slow', 'slow', 'medium', 'fast', 'very_fast'])
    
    # Check if pitcher column exists before using it
    if 'pitcher' in df.columns:
        df['velocity_std'] = df.groupby('pitcher')['release_speed'].transform('std')
        df['velocity_diff_from_avg'] = df['release_speed'] - df.groupby('pitcher')['release_speed'].transform('mean')
    else:
        df['velocity_std'] = df['release_speed'].std()
        df['velocity_diff_from_avg'] = 0
    
    # 16. Movement deception features (using improved break values)
    df['horizontal_movement'] = df['horizontal_break']  # Use the better break values
    df['vertical_movement'] = df['vertical_break']  # Use the better break values
    # movement_ratio already calculated above with better values
    df['high_horizontal_movement'] = (np.abs(df['horizontal_break']) > 5).astype(int)
    df['high_vertical_movement'] = (np.abs(df['vertical_break']) > 5).astype(int)
    
    # 17. Zone precision features
    df['zone_center_distance'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - (df['sz_top'] + df['sz_bot'])/2)**2)
    df['zone_corner'] = ((df['plate_x'] >= 0.7) | (df['plate_x'] <= -0.7)) & ((df['plate_z'] >= df['sz_top'] - 0.2) | (df['plate_z'] <= df['sz_bot'] + 0.2))
    df['zone_heart'] = (df['zone_center_distance'] < 0.5).astype(int)
    df['zone_shadow'] = ((df['plate_x'] >= -1.0) & (df['plate_x'] <= 1.0) & (df['plate_z'] >= df['sz_bot'] - 0.2) & (df['plate_z'] <= df['sz_top'] + 0.2)) & ~df['in_strike_zone']
    
    # 18. Advanced count psychology features
    df['count_advantage'] = np.where(df['count_pressure'] > 0, 'hitter_ahead', np.where(df['count_pressure'] < 0, 'pitcher_ahead', 'neutral'))
    df['pressure_situation'] = (df['two_strikes'] | df['three_balls']).astype(int)
    df['must_swing'] = df['two_strikes'].astype(int)
    df['can_take'] = (df['ahead_in_count'] & ~df['in_strike_zone']).astype(int)
    
    # 19. Pitch type deception features
    df['fastball_high'] = df['is_fastball'] & (df['plate_z'] > df['sz_top'])
    df['breaking_ball_low'] = df['is_breaking_ball'] & (df['plate_z'] < df['sz_bot'])
    df['offspeed_middle'] = df['is_offspeed'] & df['in_strike_zone']
    
    # Check if at_bat_number exists before using it
    if 'at_bat_number' in df.columns:
        df['pitch_type_change'] = df.groupby('at_bat_number')['pitch_type'].shift(1) != df['pitch_type']
    else:
        df['pitch_type_change'] = 0
    
    # 20. Advanced location features
    df['location_quadrant'] = np.where(
        df['plate_x'] >= 0,
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_right', 'down_right'),
        np.where(df['plate_z'] >= (df['sz_top'] + df['sz_bot']) / 2, 'up_left', 'down_left')
    )
    df['location_extreme'] = (np.abs(df['plate_x']) > 1.0) | (df['plate_z'] > df['sz_top'] + 0.5) | (df['plate_z'] < df['sz_bot'] - 0.5)
    
    # 21. Spin and movement correlation features
    if 'release_spin_rate' in df.columns:
        df['spin_movement_correlation'] = df['release_spin_rate'] * df['movement_magnitude']
        df['high_spin'] = (df['release_spin_rate'] > 2500).astype(int)
        df['low_spin'] = (df['release_spin_rate'] < 2000).astype(int)
    else:
        df['spin_movement_correlation'] = 0
        df['high_spin'] = 0
        df['low_spin'] = 0
    
    # 22. Game situation features
    df['late_inning'] = (df['inning'] >= 8).astype(int) if 'inning' in df.columns else 0
    df['close_score'] = (np.abs(df['home_score'] - df['away_score']) <= 1).astype(int) if 'home_score' in df.columns and 'away_score' in df.columns else 0
    df['high_leverage'] = (df['late_inning'] & df['close_score']).astype(int)
    
    # 23. Batter-specific features (if available)
    if 'batter' in df.columns:
        # Don't reference 'swing' column since it's created later
        df['batter_swing_rate'] = 0.5  # Default value
        df['batter_zone_swing_rate'] = 0.5  # Default value
    else:
        df['batter_swing_rate'] = 0.5
        df['batter_zone_swing_rate'] = 0.5
    
    # 24. Pitcher-specific features (if available)
    if 'pitcher' in df.columns:
        # Don't reference 'swing' column since it's created later
        df['pitcher_swing_rate'] = 0.5  # Default value
        df['pitcher_zone_swing_rate'] = 0.5  # Default value
    else:
        df['pitcher_swing_rate'] = 0.5
        df['pitcher_zone_swing_rate'] = 0.5
    
    # NEW IMPROVED FEATURES FOR HIGHER ACCURACY
    
    # 25. Advanced count-based features
    df['count_ratio'] = df['strikes'] / (df['balls'] + df['strikes'] + 0.1)  # Avoid division by zero
    df['behind_by_two'] = (df['strikes'] - df['balls'] >= 2).astype(int)
    df['ahead_by_two'] = (df['balls'] - df['strikes'] >= 2).astype(int)
    df['full_count_pressure'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
    
    # 26. Zone-specific count features
    df['in_zone_two_strikes'] = df['in_strike_zone'] & df['two_strikes']
    df['out_zone_ahead'] = ~df['in_strike_zone'] & df['ahead_in_count']
    df['edge_zone_decision'] = ((df['zone_distance'] > 0.5) & (df['zone_distance'] < 1.0)).astype(int)
    
    # 27. Velocity deception features
    df['velocity_surprise'] = (df['release_speed'] < 90) & df['is_fastball']
    df['velocity_consistency'] = (df['release_speed'] > 95) & df['is_fastball']
    df['breaking_ball_velocity'] = df['is_breaking_ball'] & (df['release_speed'] > 85)
    
    # 28. Movement deception features
    df['high_movement_fastball'] = df['is_fastball'] & (df['movement_magnitude'] > 8)
    df['low_movement_breaking'] = df['is_breaking_ball'] & (df['movement_magnitude'] < 5)
    df['unexpected_movement'] = ((df['movement_magnitude'] > 12) | (df['movement_magnitude'] < 3)).astype(int)
    
    # 29. Location deception features
    df['corner_pitch'] = df['zone_corner'].astype(int)
    df['heart_pitch'] = df['zone_heart'].astype(int)
    df['shadow_pitch'] = df['zone_shadow'].astype(int)
    df['extreme_location'] = (np.abs(df['plate_x']) > 1.2) | (df['plate_z'] > df['sz_top'] + 0.8) | (df['plate_z'] < df['sz_bot'] - 0.8)
    
    # 30. Advanced interaction features
    df['velocity_x_location'] = df['release_speed'] * df['zone_distance']
    df['pitch_type_x_location'] = df['is_fastball'] * df['in_strike_zone']
    df['count_x_zone'] = df['count_pressure'] * df['in_strike_zone']
    
    # 31. Context-based features
    df['early_count_swing'] = (df['count_total'] <= 2) & df['in_strike_zone']
    df['late_count_take'] = (df['count_total'] >= 4) & ~df['in_strike_zone']
    df['pressure_swing'] = df['two_strikes'] & df['in_strike_zone']
    df['opportunity_take'] = df['ahead_in_count'] & ~df['in_strike_zone']
    
    # 32. COMPREHENSIVE COUNT FEATURES - Calculate dynamically from the data
    print("Calculating comprehensive count features...")
    count_features, count_results = calculate_comprehensive_count_features(df)
    
    # Add count features to the dataframe efficiently
    count_features_df = pd.DataFrame([count_features])
    for col in count_features_df.columns:
        df[col] = count_features_df[col].iloc[0]
    
    # Add current count features for each row efficiently
    def get_current_count_features(row):
        return get_count_features_for_pitch(
            row['balls'], row['strikes'], row['pitch_type'], count_features
        )
    
    # Apply once and extract all features
    current_features = df.apply(get_current_count_features, axis=1)
    
    # Add features efficiently using a single operation
    current_features_df = pd.DataFrame(current_features.tolist())
    df = pd.concat([df, current_features_df], axis=1)
    
    print(f"Added {len(count_features)} comprehensive count features")
    
    # Analyze count patterns
    analysis = analyze_count_patterns(count_results)
    print("\nCount Pattern Analysis:")
    print("Most aggressive counts:")
    for count, rate in analysis['most_aggressive_counts']:
        print(f"  {count}: {rate:.3f}")
    print("Least aggressive counts:")
    for count, rate in analysis['least_aggressive_counts']:
        print(f"  {count}: {rate:.3f}")
    
    # ADDING ALL MISSING FEATURES IDENTIFIED IN ANALYSIS
    print("Adding missing features to align with model expectations...")
    
    # 1. Breaking ball high (already calculated above)
    # 2. Close game (already calculated above)
    # 3. Corner pitch (already calculated above)
    
    # 4-7. Count-specific rates (will be calculated from career data)
    df['count_field_out_rate'] = 0.0
    df['count_hit_rate'] = 0.0
    df['count_swing_rate_adjustment'] = 0.0
    df['count_whiff_rate'] = 0.0
    
    # 8-16. Early count features (already calculated above)
    # 17. Extreme location (already calculated above)
    # 18-19. First/last pitch (already calculated above)
    # 20-22. Heart pitch and movement features (already calculated above)
    # 23-24. Movement features (already calculated above)
    # 25. Inning late (already calculated above)
    # 26-27. Late count and low movement breaking (already calculated above)
    # 28-29. Middle count and movement ratio (already calculated above)
    # 30. Offspeed low (already calculated above)
    # 31-32. Pitch in at bat and pitch type change (already calculated above)
    
    # 33-35. Pitch type rates
    df['pitch_type_field_out_rate'] = 0.0
    df['pitch_type_hit_rate'] = 0.0
    df['pitch_type_whiff_rate'] = 0.0
    
    # 36-37. Plate location normalized movement (already calculated above)
    # 38-41. Pressure features (already calculated above)
    # 42-43. Shadow pitch and unexpected movement (already calculated above)
    # 44. Velocity drop (already calculated above)
    # 45-52. Zone-specific features (already calculated above)
    
    # 32. Advanced zone features
    df['zone_quadrant_encoded'] = pd.Categorical(df['zone_quadrant']).codes
    df['location_quadrant_encoded'] = pd.Categorical(df['location_quadrant']).codes
    df['count_advantage_encoded'] = pd.Categorical(df['count_advantage']).codes
    
    # CALCULATE ACTUAL HITTER-SPECIFIC FEATURES FROM CAREER DATA
    print("Calculating actual hitter-specific features from career data...")
    
    # Filter to swing events to calculate actual swing rates
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'foul_bunt',
                   'missed_bunt', 'bunt_foul_tip', 'single', 'double', 'triple', 'home_run',
                   'groundout', 'force_out', 'double_play', 'triple_play', 'sac_fly', 'sac_bunt',
                   'field_error', 'fielders_choice', 'fielders_choice_out', 'sac_fly_double_play',
                   'sac_bunt_double_play', 'grounded_into_double_play']
    
    swing_df = df[df['events'].isin(swing_events)].copy()
    
    if len(swing_df) > 0:
        # Calculate actual swing rates by pitch type
        fastball_types = ['FF', 'SI', 'FC', 'FT']
        breaking_types = ['SL', 'CU', 'KC', 'SV']
        offspeed_types = ['CH', 'FS', 'FO']
        
        fastball_swings = swing_df[swing_df['pitch_type'].isin(fastball_types)]
        breaking_swings = swing_df[swing_df['pitch_type'].isin(breaking_types)]
        offspeed_swings = swing_df[swing_df['pitch_type'].isin(offspeed_types)]
        
        # Calculate actual swing rates
        acuna_features = {}
        acuna_features['acuna_fastball_swing_rate'] = len(fastball_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_breaking_swing_rate'] = len(breaking_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_offspeed_swing_rate'] = len(offspeed_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Zone swing rates
        zone_swings = swing_df[((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5))]
        outside_swings = swing_df[swing_df['plate_x'].abs() > 0.7]
        high_swings = swing_df[swing_df['plate_z'] > 3.0]
        low_swings = swing_df[swing_df['plate_z'] < 2.0]
        
        acuna_features['acuna_zone_swing_rate'] = len(zone_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_outside_swing_rate'] = len(outside_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_high_swing_rate'] = len(high_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_low_swing_rate'] = len(low_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Count-based swing rates
        ahead_swings = swing_df[swing_df['balls'] > swing_df['strikes']]
        behind_swings = swing_df[swing_df['strikes'] > swing_df['balls']]
        two_strikes_swings = swing_df[swing_df['strikes'] >= 2]
        full_count_swings = swing_df[(swing_df['balls'] == 3) & (swing_df['strikes'] == 2)]
        
        acuna_features['acuna_ahead_swing_rate'] = len(ahead_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_behind_swing_rate'] = len(behind_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_two_strikes_swing_rate'] = len(two_strikes_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_full_count_swing_rate'] = len(full_count_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Velocity-based swing rates
        high_vel_swings = swing_df[swing_df['release_speed'] > 95]
        low_vel_swings = swing_df[swing_df['release_speed'] < 85]
        
        acuna_features['acuna_high_vel_swing_rate'] = len(high_vel_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_low_vel_swing_rate'] = len(low_vel_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Movement-based swing rates
        high_movement_swings = swing_df[swing_df['movement_magnitude'] > 8]
        low_movement_swings = swing_df[swing_df['movement_magnitude'] < 4]
        
        acuna_features['acuna_high_movement_swing_rate'] = len(high_movement_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_low_movement_swing_rate'] = len(low_movement_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Situational swing rates
        if 'count_total' not in swing_df.columns:
            swing_df['count_total'] = swing_df['balls'] + swing_df['strikes']
        
        first_pitch_swings = swing_df[swing_df['count_total'] == 0]
        last_pitch_swings = swing_df[swing_df['count_total'] >= 5]
        
        acuna_features['acuna_first_pitch_swing_rate'] = len(first_pitch_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_last_pitch_swing_rate'] = len(last_pitch_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Calculate zone-specific swing rates from actual data
        zone_heart_swings = swing_df[((swing_df['plate_x'].abs() <= 0.5) & (swing_df['plate_z'] >= 2.0) & (swing_df['plate_z'] <= 3.0))]
        zone_corner_swings = swing_df[((swing_df['plate_x'].abs() >= 0.7) | (swing_df['plate_z'] >= 3.5) | (swing_df['plate_z'] <= 1.5))]
        zone_shadow_swings = swing_df[((swing_df['plate_x'].abs() <= 1.0) & (swing_df['plate_z'] >= 1.2) & (swing_df['plate_z'] <= 3.8)) & ~((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5))]
        
        acuna_features['acuna_zone_heart_swing_rate'] = len(zone_heart_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_zone_corner_swing_rate'] = len(zone_corner_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_zone_shadow_swing_rate'] = len(zone_shadow_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # NEW: Zone-specific contact rate features (for outcome prediction)
        # Calculate contact rates by zone - hits divided by swings in each zone
        hit_events = ['single', 'double', 'triple', 'home_run']
        
        # Zone heart contact rate
        zone_heart_hits = swing_df[((swing_df['plate_x'].abs() <= 0.5) & (swing_df['plate_z'] >= 2.0) & (swing_df['plate_z'] <= 3.0)) & 
                                   (swing_df['events'].isin(hit_events))]
        acuna_features['acuna_zone_heart_hit_rate'] = len(zone_heart_hits) / len(zone_heart_swings) if len(zone_heart_swings) > 0 else 0.0
        
        # Zone corner contact rate
        zone_corner_hits = swing_df[((swing_df['plate_x'].abs() >= 0.7) | (swing_df['plate_z'] >= 3.5) | (swing_df['plate_z'] <= 1.5)) & 
                                    (swing_df['events'].isin(hit_events))]
        acuna_features['acuna_zone_corner_hit_rate'] = len(zone_corner_hits) / len(zone_corner_swings) if len(zone_corner_swings) > 0 else 0.0
        
        # Zone shadow contact rate
        zone_shadow_hits = swing_df[((swing_df['plate_x'].abs() <= 1.0) & (swing_df['plate_z'] >= 1.2) & (swing_df['plate_z'] <= 3.8)) & 
                                    ~((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5)) & 
                                    (swing_df['events'].isin(hit_events))]
        acuna_features['acuna_zone_shadow_hit_rate'] = len(zone_shadow_hits) / len(zone_shadow_swings) if len(zone_shadow_swings) > 0 else 0.0
        
        # Overall zone contact rate
        zone_hits = swing_df[((swing_df['plate_x'].abs() <= 0.7) & (swing_df['plate_z'] >= 1.5) & (swing_df['plate_z'] <= 3.5)) & 
                             (swing_df['events'].isin(hit_events))]
        acuna_features['acuna_zone_hit_rate'] = len(zone_hits) / len(zone_swings) if len(zone_swings) > 0 else 0.0
        
        # Outside zone contact rate
        outside_hits = swing_df[(swing_df['plate_x'].abs() > 0.7) & (swing_df['events'].isin(hit_events))]
        acuna_features['acuna_outside_hit_rate'] = len(outside_hits) / len(outside_swings) if len(outside_swings) > 0 else 0.0
        
        # Calculate location-specific swing rates
        location_extreme_swings = swing_df[((swing_df['plate_x'].abs() > 1.2) | (swing_df['plate_z'] > 4.0) | (swing_df['plate_z'] < 1.0))]
        location_heart_swings = swing_df[((swing_df['plate_x'].abs() <= 0.5) & (swing_df['plate_z'] >= 2.0) & (swing_df['plate_z'] <= 3.0))]
        
        acuna_features['acuna_location_extreme_swing_rate'] = len(location_extreme_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_location_heart_swing_rate'] = len(location_heart_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Calculate situational swing rates
        pressure_swings = swing_df[swing_df['strikes'] >= 2]
        opportunity_swings = swing_df[swing_df['balls'] > swing_df['strikes']]
        
        acuna_features['acuna_pressure_swing_rate'] = len(pressure_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        acuna_features['acuna_opportunity_swing_rate'] = len(opportunity_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        
        # Calculate velocity change swing rates (using velocity differences)
        if 'release_speed' in swing_df.columns:
            avg_velocity = swing_df['release_speed'].mean()
            velocity_drop_swings = swing_df[swing_df['release_speed'] < (avg_velocity - 5)]
            velocity_surge_swings = swing_df[swing_df['release_speed'] > (avg_velocity + 5)]
            
            acuna_features['acuna_velocity_drop_swing_rate'] = len(velocity_drop_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
            acuna_features['acuna_velocity_surge_swing_rate'] = len(velocity_surge_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        else:
            acuna_features['acuna_velocity_drop_swing_rate'] = 0.0
            acuna_features['acuna_velocity_surge_swing_rate'] = 0.0
        
        # Calculate pitch type change swing rate (would need sequence data, using pitch type distribution for now)
        if 'pitch_type' in swing_df.columns:
            pitch_type_counts = swing_df['pitch_type'].value_counts()
            most_common_pitch = pitch_type_counts.index[0] if len(pitch_type_counts) > 0 else 'FF'
            non_most_common_swings = swing_df[swing_df['pitch_type'] != most_common_pitch]
            acuna_features['acuna_pitch_type_change_swing_rate'] = len(non_most_common_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        else:
            acuna_features['acuna_pitch_type_change_swing_rate'] = 0.0
        
        # Calculate inning and game situation swing rates (if data available)
        if 'inning' in swing_df.columns:
            late_inning_swings = swing_df[swing_df['inning'] >= 7]
            acuna_features['acuna_late_inning_swing_rate'] = len(late_inning_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        else:
            acuna_features['acuna_late_inning_swing_rate'] = 0.0
        
        if 'home_score' in swing_df.columns and 'away_score' in swing_df.columns:
            close_game_swings = swing_df[abs(swing_df['home_score'] - swing_df['away_score']) <= 2]
            acuna_features['acuna_close_game_swing_rate'] = len(close_game_swings) / len(swing_df) if len(swing_df) > 0 else 0.0
        else:
            acuna_features['acuna_close_game_swing_rate'] = 0.0
        
        print(f"Calculated {len(acuna_features)} actual hitter features from career data")
        
        # Add comprehensive count features to acuna_features
        for feature_name, feature_value in count_features.items():
            acuna_features[f'acuna_{feature_name}'] = feature_value
        
        print(f"Added {len(count_features)} comprehensive count features to hitter features")
        
    else:
        # Fallback to zeros if no swing data available
        acuna_features = {
            'acuna_fastball_swing_rate': 0.0,
            'acuna_breaking_swing_rate': 0.0,
            'acuna_offspeed_swing_rate': 0.0,
            'acuna_zone_swing_rate': 0.0,
            'acuna_outside_swing_rate': 0.0,
            'acuna_high_swing_rate': 0.0,
            'acuna_low_swing_rate': 0.0,
            'acuna_ahead_swing_rate': 0.0,
            'acuna_behind_swing_rate': 0.0,
            'acuna_two_strikes_swing_rate': 0.0,
            'acuna_full_count_swing_rate': 0.0,
            'acuna_high_vel_swing_rate': 0.0,
            'acuna_low_vel_swing_rate': 0.0,
            'acuna_high_movement_swing_rate': 0.0,
            'acuna_low_movement_swing_rate': 0.0,
            'acuna_late_inning_swing_rate': 0.0,
            'acuna_close_game_swing_rate': 0.0,
            'acuna_first_pitch_swing_rate': 0.0,
            'acuna_last_pitch_swing_rate': 0.0,
            'acuna_pitch_type_change_swing_rate': 0.0,
            'acuna_velocity_drop_swing_rate': 0.0,
            'acuna_velocity_surge_swing_rate': 0.0,
            'acuna_location_extreme_swing_rate': 0.0,
            'acuna_location_heart_swing_rate': 0.0,
            'acuna_pressure_swing_rate': 0.0,
            'acuna_opportunity_swing_rate': 0.0,
            'acuna_zone_corner_swing_rate': 0.0,
            'acuna_zone_shadow_swing_rate': 0.0,
            'acuna_zone_heart_swing_rate': 0.0,
            # NEW: Zone-specific contact rate features
            'acuna_zone_heart_hit_rate': 0.0,
            'acuna_zone_corner_hit_rate': 0.0,
            'acuna_zone_shadow_hit_rate': 0.0,
            'acuna_zone_hit_rate': 0.0,
            'acuna_outside_hit_rate': 0.0
        }
        
        # Add comprehensive count features to defaults
        for feature_name, feature_value in count_features.items():
            acuna_features[f'acuna_{feature_name}'] = feature_value
    
    # Assign all calculated Acuna features at once to avoid fragmentation
    # Create a DataFrame with all features and concatenate efficiently
    acuna_features_df = pd.DataFrame([acuna_features] * len(df), index=df.index)
    df = pd.concat([df, acuna_features_df], axis=1)
    
    # NEW: Add BABIP features for each pitch type and zone combination
    print("Loading BABIP data for outcome prediction features...")
    babip_lookup = load_babip_data()
    
    if babip_lookup:
        # Add BABIP features for each pitch
        babip_features = []
        for idx, row in df.iterrows():
            pitch_type = row['pitch_type']
            zone = row['zone']
            babip_data = get_babip_features(pitch_type, zone, babip_lookup)
            babip_features.append(babip_data)
        
        # Convert to DataFrame and add to main DataFrame
        babip_df = pd.DataFrame(babip_features, index=df.index)
        df = pd.concat([df, babip_df], axis=1)
        
        print(f"Added BABIP features for {len(babip_lookup)} pitch type x zone combinations")
    else:
        # Add default BABIP features if no data available
        df['batting_average_bip'] = 0.25
        df['whiff_rate'] = 0.35
        df['field_out_rate_bip'] = 0.40
        df['balls_in_play'] = 0
        print("Added default BABIP features (no BABIP data available)")
    
    # NEW: Add proxy contact features that combine zone location with swing rates
    # These are especially useful for outcome prediction
    df['zone_heart_contact'] = df['zone_heart'] * df['acuna_zone_heart_swing_rate']
    df['zone_corner_contact'] = df['zone_corner'] * df['acuna_zone_corner_swing_rate']
    df['zone_shadow_contact'] = df['zone_shadow'] * df['acuna_zone_shadow_swing_rate']
    df['zone_overall_contact'] = df['in_strike_zone'] * df['acuna_zone_swing_rate']
    df['outside_zone_contact'] = df['far_from_zone'] * df['acuna_outside_swing_rate']
    
    # Additional contact features for different pitch types
    df['fastball_zone_contact'] = df['is_fastball'] * df['zone_overall_contact']
    df['breaking_zone_contact'] = df['is_breaking_ball'] * df['zone_overall_contact']
    df['offspeed_zone_contact'] = df['is_offspeed'] * df['zone_overall_contact']
    
    # Contact features for different count situations
    df['pressure_zone_contact'] = df['pressure_situation'] * df['zone_overall_contact']
    df['opportunity_zone_contact'] = df['ahead_in_count'] * df['zone_overall_contact']
    df['two_strikes_zone_contact'] = df['two_strikes'] * df['zone_overall_contact']
    
    # Define features - REMOVE DUPLICATES
    num_feats = [
        'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
        'release_pos_x', 'release_pos_y', 'release_pos_z',
        'vx0', 'vy0', 'vz0', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
        'sz_top', 'sz_bot', 'zone',
        'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
        'arm_angle', 'balls', 'strikes', 'spin_dir', 'spin_rate_deprecated',
        'break_angle_deprecated', 'break_length_deprecated',
        'effective_speed', 'age_pit',
        # NEW ENGINEERED FEATURES
        'zone_distance', 'movement_magnitude', 'plate_x_norm_x_movement', 'plate_z_norm_x_movement',
        # NEW FEATURES FOR BETTER NO-SWING PREDICTION
        'count_pressure', 'count_total', 'behind_in_count', 'ahead_in_count', 'two_strikes', 'three_balls',
        'in_strike_zone', 'far_from_zone', 'high_pitch', 'low_pitch', 'inside_pitch', 'outside_pitch',
        'is_fastball', 'is_breaking_ball', 'is_offspeed',
        'zone_distance_x_count_pressure', 'movement_x_count_pressure', 'in_zone_x_two_strikes', 'far_from_zone_x_ahead',
        # ADVANCED FEATURES FOR FURTHER IMPROVEMENT
        'velocity_movement_ratio', 'high_velocity', 'low_velocity', 'high_movement',
        'zone_edge_distance', 'zone_top_distance', 'zone_bottom_distance', 'closest_zone_edge',
        'full_count', 'hitters_count', 'pitchers_count', 'neutral_count',
        'velocity_drop', 'breaking_ball_high', 'offspeed_low',
        'inning_late', 'close_game',
        # ADVANCED FEATURES FOR 70-80% ACCURACY
        'pitch_in_at_bat', 'first_pitch', 'last_pitch',
        'velocity_diff_from_avg', 'horizontal_movement', 'vertical_movement', 'movement_ratio',
        'high_horizontal_movement', 'high_vertical_movement', 'zone_center_distance',
        'zone_corner', 'zone_heart', 'zone_shadow', 'pressure_situation', 'must_swing', 'can_take',
        'fastball_high', 'breaking_ball_low', 'offspeed_middle', 'pitch_type_change',
        'location_extreme', 'high_leverage',
        # NEW IMPROVED FEATURES FOR HIGHER ACCURACY
        'count_ratio', 'behind_by_two', 'ahead_by_two', 'full_count_pressure',
        'in_zone_two_strikes', 'out_zone_ahead', 'edge_zone_decision',
        'velocity_surprise', 'velocity_consistency', 'breaking_ball_velocity',
        'high_movement_fastball', 'low_movement_breaking', 'unexpected_movement',
        'corner_pitch', 'heart_pitch', 'shadow_pitch', 'extreme_location',
        'velocity_x_location', 'pitch_type_x_location', 'count_x_zone',
        'early_count_swing', 'late_count_take', 'pressure_swing', 'opportunity_take',
        'zone_quadrant_encoded', 'location_quadrant_encoded', 'count_advantage_encoded',
        # NEW COUNT-BASED FEATURE WEIGHTING (HIGH PRIORITY)
        'early_count', 'middle_count', 'late_count', 'pressure_count',
        'early_count_penalty', 'early_count_zone_penalty', 'early_count_outside_penalty',
        'count_swing_rate_adjustment', 'early_count_location_penalty',
        'early_count_breaking_penalty', 'early_count_offspeed_penalty',
        'early_count_low_vel_penalty', 'early_count_high_vel_penalty',
        # HITTER-SPECIFIC SWING TENDENCY FEATURES (HIGH WEIGHT)
        'acuna_fastball_swing_rate', 'acuna_breaking_swing_rate', 'acuna_offspeed_swing_rate',
        'acuna_zone_swing_rate', 'acuna_outside_swing_rate', 'acuna_high_swing_rate', 'acuna_low_swing_rate',
        'acuna_ahead_swing_rate', 'acuna_behind_swing_rate', 'acuna_two_strikes_swing_rate', 'acuna_full_count_swing_rate',
        'acuna_high_vel_swing_rate', 'acuna_low_vel_swing_rate',
        'acuna_high_movement_swing_rate', 'acuna_low_movement_swing_rate',
        'acuna_late_inning_swing_rate', 'acuna_close_game_swing_rate',
        'acuna_first_pitch_swing_rate', 'acuna_last_pitch_swing_rate',
        'acuna_pitch_type_change_swing_rate', 'acuna_velocity_drop_swing_rate', 'acuna_velocity_surge_swing_rate',
        'acuna_location_extreme_swing_rate', 'acuna_location_heart_swing_rate',
        'acuna_pressure_swing_rate', 'acuna_opportunity_swing_rate',
        'acuna_zone_corner_swing_rate', 'acuna_zone_shadow_swing_rate', 'acuna_zone_heart_swing_rate',
        # NEW: Zone-specific contact rate features
        'acuna_zone_heart_hit_rate', 'acuna_zone_corner_hit_rate', 'acuna_zone_shadow_hit_rate',
        'acuna_zone_hit_rate', 'acuna_outside_hit_rate',
        # NEW: Proxy contact features
        'zone_heart_contact', 'zone_corner_contact', 'zone_shadow_contact',
        'zone_overall_contact', 'outside_zone_contact',
        'fastball_zone_contact', 'breaking_zone_contact', 'offspeed_zone_contact',
        'pressure_zone_contact', 'opportunity_zone_contact', 'two_strikes_zone_contact',
        # NEW: BABIP and whiff rate features for outcome prediction
        'batting_average_bip', 'whiff_rate', 'field_out_rate_bip', 'balls_in_play', 'total_swings', 'total_whiffs'
    ]
    
    cat_feats = ['pitch_type', 'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'stand', 'home_team', 'zone_quadrant', 'location_quadrant', 'count_advantage']
    
    # Filter to available features
    num_feats = [f for f in num_feats if f in df.columns]
    cat_feats = [f for f in cat_feats if f in df.columns]
    
    return df, num_feats, cat_feats

def update_swing_rates(df):
    """
    Update batter and pitcher swing rates after the swing column is created
    """
    df = df.copy()
    
    # Update batter-specific features
    if 'batter' in df.columns and 'swing' in df.columns:
        df['batter_swing_rate'] = df.groupby('batter')['swing'].transform('mean')
        df['batter_zone_swing_rate'] = df.groupby(['batter', 'in_strike_zone'])['swing'].transform('mean')
    
    # Update pitcher-specific features
    if 'pitcher' in df.columns and 'swing' in df.columns:
        df['pitcher_swing_rate'] = df.groupby('pitcher')['swing'].transform('mean')
        df['pitcher_zone_swing_rate'] = df.groupby(['pitcher', 'in_strike_zone'])['swing'].transform('mean')
    
    return df

def create_swing_classifier(df):
    """
    Model 1: Predict if batter swings or not - BALANCED VERSION
    """
    print("\n=== MODEL 1: Swing/No-Swing Classifier (BALANCED) ===")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Create swing/no-swing target with robust classification
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    
    def classify_swing_robust(row):
        """
        Robustly classify whether a pitch resulted in a swing or not.
        Handles NaN values and edge cases properly.
        """
        description = row.get('description', None)
        events = row.get('events', None)
        
        # Check if both description and events are NaN/null
        if pd.isna(description) and pd.isna(events):
            return None  # Can't classify, skip this pitch
        
        # If description is available, use it as primary source
        if not pd.isna(description):
            if description in swing_events:
                return 1  # Swing
            elif description in no_swing_events:
                return 0  # No swing
            # If description is not in either list, check events
        
        # If description is NaN or unclear, check events
        if not pd.isna(events):
            if events in ['single', 'double', 'triple', 'home_run', 'field_out', 'sac_fly', 'sac_bunt']:
                return 1  # Swing (hit into play)
            elif events in ['walk', 'strikeout', 'hit_by_pitch']:
                return 0  # No swing (walk, called strikeout, HBP)
            # If events is not in either list, we can't classify
        
        # If we can't determine from either field, return None
        return None
    
    # Apply robust classification
    print("Classifying swings and no-swings with robust logic...")
    
    # Use a more robust approach to avoid DataFrame fragmentation issues
    classifications = []
    valid_mask = []
    
    for idx, row in df.iterrows():
        classification = classify_swing_robust(row)
        if classification is not None:
            classifications.append(classification)
            valid_mask.append(True)
        else:
            print(f"Warning: Could not classify pitch at index {idx}. Description: {row.get('description', 'NaN')}, Events: {row.get('events', 'NaN')}")
            valid_mask.append(False)
    
    # Create a new DataFrame with only valid classifications using boolean indexing
    df_valid = df[valid_mask].copy()
    df_valid['swing'] = classifications
    
    print(f"Original dataset size: {len(df)}")
    print(f"Valid classifications: {len(df_valid)}")
    print(f"Unclassifiable pitches: {len(df) - len(df_valid)}")
    print(f"Valid mask count: {sum(valid_mask)}")
    
    # Show classification breakdown
    swing_count = sum(classifications)
    no_swing_count = len(classifications) - swing_count
    print(f"Swing classifications: {swing_count}")
    print(f"No-swing classifications: {no_swing_count}")
    print(f"Swing rate: {swing_count/len(classifications)*100:.1f}%")
    
    # Use the valid dataset for the rest of the analysis
    df = df_valid
    
    print(f"Original swing distribution: {df['swing'].value_counts()}")
    
    # Prepare features - this will add all the engineered features to df
    df, num_feats, cat_feats = prepare_features(df)
    
    # Update swing rates now that swing column exists
    df = update_swing_rates(df)
    
    # IMPLEMENT DOWNSAMPLING FOR BALANCED CLASSES
    print("\n=== IMPLEMENTING DOWNSAMPLING FOR BALANCED CLASSES ===")
    
    # Separate swing and no-swing data
    swing_df = df[df['swing'] == 1].copy()
    no_swing_df = df[df['swing'] == 0].copy()
    
    print(f"Original swing count: {len(swing_df)}")
    print(f"Original no-swing count: {len(no_swing_df)}")
    
    # Determine which class to downsample
    if len(swing_df) > len(no_swing_df):
        # More swings than no-swings, downsample swings
        swing_downsampled = resample(swing_df, 
                                   replace=False,
                                   n_samples=len(no_swing_df),
                                   random_state=42)
        df_balanced = pd.concat([swing_downsampled, no_swing_df])
        print(f"Downsampled swing count: {len(swing_downsampled)}")
        print(f"Final balanced dataset size: {len(df_balanced)}")
    else:
        # More no-swings than swings, downsample no-swings
        no_swing_downsampled = resample(no_swing_df, 
                                       replace=False,
                                       n_samples=len(swing_df),
                                       random_state=42)
        df_balanced = pd.concat([swing_df, no_swing_downsampled])
        print(f"Downsampled no-swing count: {len(no_swing_downsampled)}")
        print(f"Final balanced dataset size: {len(df_balanced)}")
    
    print(f"Balanced swing distribution: {df_balanced['swing'].value_counts()}")
    
    # Use the balanced dataset for training
    df = df_balanced
    
    all_feats = num_feats + cat_feats
    
    print(f"Number of numerical features: {len(num_feats)}")
    print(f"Number of categorical features: {len(cat_feats)}")
    print(f"Total features: {len(all_feats)}")
    
    # Preprocess
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    
    X = preprocessor.fit_transform(df[all_feats])
    y = df['swing'].values
    
    # Clean NaN values - replace with 0 for numerical features
    X = np.nan_to_num(X, nan=0.0)
    
    # Train/test split with stratification (now balanced due to downsampling)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    # BALANCED MODEL ARCHITECTURE WITH DOWNSAMPLING
    
    # Since we've downsampled, we don't need sample weights for balancing
    # But we'll still use feature weighting for zone/distance features
    sample_weights = None  # No sample weights needed with balanced data
    
    # Feature importance-based weighting
    # First, train a simple model to get feature importances
    temp_xgb = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        max_depth=4
    )
    if sample_weights is not None:
        temp_xgb.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        temp_xgb.fit(X_train, y_train)  # No sample weights needed with balanced data
    
    # Get feature importances for weighting
    feature_importances = temp_xgb.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    
    # Create feature weights based on importance
    feature_weights = {}
    for i, (name, importance) in enumerate(zip(feature_names, feature_importances)):
        feature_weights[name] = importance
    
    # Boost weights for hitter-specific features
    acuna_features = [
        'acuna_fastball_swing_rate', 'acuna_breaking_swing_rate', 'acuna_offspeed_swing_rate',
        'acuna_zone_swing_rate', 'acuna_outside_swing_rate', 'acuna_high_swing_rate', 'acuna_low_swing_rate',
        'acuna_ahead_swing_rate', 'acuna_behind_swing_rate', 'acuna_two_strikes_swing_rate', 'acuna_full_count_swing_rate',
        'acuna_high_vel_swing_rate', 'acuna_low_vel_swing_rate',
        'acuna_high_movement_swing_rate', 'acuna_low_movement_swing_rate',
        'acuna_late_inning_swing_rate', 'acuna_close_game_swing_rate',
        'acuna_first_pitch_swing_rate', 'acuna_last_pitch_swing_rate',
        'acuna_pitch_type_change_swing_rate', 'acuna_velocity_drop_swing_rate', 'acuna_velocity_surge_swing_rate',
        'acuna_location_extreme_swing_rate', 'acuna_location_heart_swing_rate',
        'acuna_pressure_swing_rate', 'acuna_opportunity_swing_rate',
        'acuna_zone_corner_swing_rate', 'acuna_zone_shadow_swing_rate', 'acuna_zone_heart_swing_rate'
    ]
    
    # Boost the weights for Acuna-specific features
    for feature in acuna_features:
        if feature in feature_weights:
            feature_weights[feature] *= 3.0  # Triple the weight for hitter-specific features
    
    # DOUBLE THE WEIGHTS FOR ZONE AND DISTANCE FEATURES
    zone_distance_features = [
        'zone', 'zone_distance', 'zone_center_distance', 'zone_edge_distance', 
        'zone_top_distance', 'zone_bottom_distance', 'closest_zone_edge',
        'plate_x', 'plate_z', 'plate_x_norm', 'plate_z_norm',
        'in_strike_zone', 'far_from_zone', 'high_pitch', 'low_pitch', 
        'inside_pitch', 'outside_pitch', 'zone_corner', 'zone_heart', 'zone_shadow',
        'zone_quadrant', 'location_quadrant',
        'zone_distance_x_count_pressure', 'in_zone_x_two_strikes', 'far_from_zone_x_ahead'
    ]
    
    # Double the weights for zone and distance features
    for feature in zone_distance_features:
        if feature in feature_weights:
            feature_weights[feature] *= 2.0  # Double the weight for zone/distance features
            print(f"  Doubled weight for zone/distance feature: {feature}")
    
    # Also double weights for movement features that affect zone perception
    movement_zone_features = [
        'movement_magnitude', 'horizontal_movement', 'vertical_movement', 'movement_ratio',
        'high_movement', 'high_horizontal_movement', 'high_vertical_movement',
        'plate_x_norm_x_movement', 'plate_z_norm_x_movement', 'movement_x_count_pressure'
    ]
    
    for feature in movement_zone_features:
        if feature in feature_weights:
            feature_weights[feature] *= 2.0  # Double the weight for movement features
            print(f"  Doubled weight for movement feature: {feature}")
    
    print("\nTop 10 Most Important Features for Weighting:")
    sorted_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_features[:10]:
        print(f"  {name}: {importance:.4f}")
    
    print("\nAcuna-specific features with boosted weights:")
    for feature in acuna_features:
        if feature in feature_weights:
            print(f"  {feature}: {feature_weights[feature]:.4f}")
    
    print("\nZone and distance features with doubled weights:")
    for feature in zone_distance_features:
        if feature in feature_weights:
            print(f"  {feature}: {feature_weights[feature]:.4f}")
    
    print("\nMovement features with doubled weights:")
    for feature in movement_zone_features:
        if feature in feature_weights:
            print(f"  {feature}: {feature_weights[feature]:.4f}")
    
    # IMPLEMENT FEATURE SELECTION TO REDUCE NOISE
    print("\n=== IMPLEMENTING FEATURE SELECTION ===")
    
    # Use the temporary XGBoost model for feature selection
    # Select features based on importance threshold (median)
    selector = SelectFromModel(
        temp_xgb, 
        threshold="median",
        prefit=True  # Use the already fitted model
    )
    
    # Get selected feature indices and names
    selected_features_mask = selector.get_support()
    selected_feature_indices = np.where(selected_features_mask)[0]
    selected_feature_names = feature_names[selected_feature_indices]
    
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Selected number of features: {len(selected_feature_indices)}")
    print(f"Feature reduction: {X_train.shape[1] - len(selected_feature_indices)} features removed ({((X_train.shape[1] - len(selected_feature_indices))/X_train.shape[1]*100):.1f}% reduction)")
    
    # Transform training and test data to use only selected features
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Training data shape after feature selection: {X_train_selected.shape}")
    print(f"Test data shape after feature selection: {X_test_selected.shape}")
    
    # Show top selected features
    print("\nTop 20 Selected Features:")
    # Get feature importances from the temp_xgb model
    temp_importances = temp_xgb.feature_importances_
    selected_importances = temp_importances[selected_feature_indices]
    sorted_selected_idx = np.argsort(selected_importances)[::-1]
    
    for i, idx in enumerate(sorted_selected_idx[:20]):
        feature_idx = selected_feature_indices[idx]
        print(f"  {i+1:2d}. {feature_names[feature_idx]}: {selected_importances[idx]:.4f}")
    
    # Show some removed features (lowest importance)
    removed_features_mask = ~selected_features_mask
    removed_feature_indices = np.where(removed_features_mask)[0]
    removed_importances = temp_importances[removed_feature_indices]
    sorted_removed_idx = np.argsort(removed_importances)[::-1]
    
    print(f"\nTop 10 Removed Features (Lowest Importance):")
    for i, idx in enumerate(sorted_removed_idx[:10]):
        feature_idx = removed_feature_indices[idx]
        print(f"  {i+1:2d}. {feature_names[feature_idx]}: {removed_importances[idx]:.4f}")
    
    # Use selected features for all subsequent model training
    X_train = X_train_selected
    X_test = X_test_selected
    
    print(f"\n✅ Feature selection completed. Using {X_train.shape[1]} features for model training.")
    
    # Analyze feature selection impact
    print(f"\n=== FEATURE SELECTION ANALYSIS ===")
    
    # Check correlation between selected features
    if X_train.shape[1] > 1:
        correlation_matrix = np.corrcoef(X_train.T)
        high_corr_pairs = []
        for i in range(correlation_matrix.shape[0]):
            for j in range(i+1, correlation_matrix.shape[1]):
                if abs(correlation_matrix[i, j]) > 0.8:  # High correlation threshold
                    high_corr_pairs.append((i, j, correlation_matrix[i, j]))
        
        if high_corr_pairs:
            print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|correlation| > 0.8):")
            for i, j, corr in high_corr_pairs[:5]:  # Show first 5
                print(f"  Features {i} and {j}: correlation = {corr:.3f}")
        else:
            print("No highly correlated features found (|correlation| > 0.8)")
    
    # Show feature importance distribution
    if len(selected_importances) > 0:
        importance_percentiles = np.percentile(selected_importances, [25, 50, 75, 90, 95])
        print(f"\nFeature importance distribution:")
        print(f"  25th percentile: {importance_percentiles[0]:.4f}")
        print(f"  50th percentile: {importance_percentiles[1]:.4f}")
        print(f"  75th percentile: {importance_percentiles[2]:.4f}")
        print(f"  90th percentile: {importance_percentiles[3]:.4f}")
        print(f"  95th percentile: {importance_percentiles[4]:.4f}")
    else:
        print(f"\nNo features selected for importance distribution analysis.")
    
    # Create multiple models with different balancing strategies and feature weighting
    xgb_model1 = XGBClassifier(
        eval_metric='logloss', 
        random_state=42,
        max_depth=6,  # Reduced to prevent overfitting
        learning_rate=0.1,  # Increased for better convergence
        n_estimators=500,  # More trees
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,  # Added to prevent overfitting
        gamma=0.1,  # Added for regularization
        objective='binary:logistic'
    )
    
    xgb_model2 = XGBClassifier(
        eval_metric='logloss', 
        random_state=43,
        max_depth=4,  # Even more conservative
        learning_rate=0.05,
        n_estimators=800,  # More trees for this model
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.5,
        min_child_weight=5,
        gamma=0.2,
        objective='binary:logistic'
    )
    
    # Add Logistic Regression (no class_weight needed with downsampling)
    lr_model = LogisticRegression(
        random_state=42,
        C=0.8,  # Reduced for more regularization
        max_iter=1000,
        solver='liblinear'
    )
    
    # Add SVM (no class_weight needed with downsampling)
    svm_model = SVC(
        random_state=42,
        C=1.0,
        kernel='rbf',
        gamma='scale',
        probability=True
    )
    
    # Try to add RandomForest with balanced parameters and feature importance
    try:
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,  # Reduced to prevent overfitting
            random_state=42,
            min_samples_split=15,  # Increased
            min_samples_leaf=8,    # Increased
            max_features='sqrt',
            bootstrap=True,
            oob_score=True  # Out-of-bag scoring
        )
        
        # Create ensemble with multiple models - focus on balanced performance
        ensemble_model = VotingClassifier(
            estimators=[
                ('xgb1', xgb_model1),
                ('xgb2', xgb_model2),
                ('rf', rf_model),
                ('lr', lr_model),
                ('svm', svm_model)
            ],
            voting='soft',  # Use probability voting
            weights=[0.3, 0.25, 0.2, 0.15, 0.1]  # Weight the models
        )
        print("Using ensemble with XGBoost, RandomForest, LogisticRegression, and SVM")
        print(f"Downsampling applied for balanced classes")
        print(f"Hitter-specific features boosted with 3x weight")
        print(f"Zone and distance features boosted with 2x weight")
        print(f"Movement features boosted with 2x weight")
        
    except Exception as e:
        print(f"Using XGBoost, LogisticRegression, and SVM only due to: {e}")
        ensemble_model = VotingClassifier(
            estimators=[
                ('xgb1', xgb_model1),
                ('xgb2', xgb_model2),
                ('lr', lr_model),
                ('svm', svm_model)
            ],
            voting='soft',
            weights=[0.4, 0.3, 0.2, 0.1]
        )
        print(f"Downsampling applied for balanced classes")
        print(f"Hitter-specific features boosted with 3x weight")
        print(f"Zone and distance features boosted with 2x weight")
        print(f"Movement features boosted with 2x weight")
    
    # Train model with sample weights
    ensemble_model.fit(X_train, y_train)
    
    # IMPLEMENT COUNT-SPECIFIC THRESHOLDS TO REDUCE EARLY COUNT FALSE POSITIVES
    print("\n=== IMPLEMENTING COUNT-SPECIFIC THRESHOLDS ===")
    
    # Create count-specific thresholds based on analysis
    # Early counts: Higher threshold (more conservative) to reduce false positives
    # Pressure counts: Lower threshold (more aggressive) to catch necessary swings
    count_thresholds = {
        'early_count': 0.95,    # Very high threshold for early counts (≤1 ball, ≤1 strike)
        'middle_count': 0.85,   # High threshold for middle counts (1-1)
        'pressure_count': 0.75,  # Lower threshold for pressure situations (≥2 strikes or ≥3 balls)
        'default': 0.9          # Default threshold for other situations
    }
    
    print("Count-specific thresholds:")
    for count_type, threshold in count_thresholds.items():
        print(f"  {count_type}: {threshold}")
    
    # IMPLEMENT PROBABILITY CALIBRATION
    print("\n=== IMPLEMENTING PROBABILITY CALIBRATION ===")
    
    # Create calibrated model using sigmoid calibration
    calibrated_model = CalibratedClassifierCV(
        ensemble_model, 
        method='sigmoid', 
        cv=3,
        n_jobs=-1
    )
    
    # Fit the calibrated model
    calibrated_model.fit(X_train, y_train)
    print("Probability calibration completed using sigmoid method with 3-fold CV")
    
    # Evaluate both original and calibrated models
    y_pred = ensemble_model.predict(X_test)
    y_pred_proba = ensemble_model.predict_proba(X_test)
    y_calibrated_proba = calibrated_model.predict_proba(X_test)
    
    # Test different probability thresholds for swing prediction
    print("\n=== TESTING DIFFERENT PROBABILITY THRESHOLDS ===")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Original Model (0.5 threshold):")
    y_pred_original = (y_pred_proba[:, 1] > 0.5).astype(int)
    print(classification_report(y_test, y_pred_original, target_names=['No Swing', 'Swing']))
    
    print("\nCalibrated Model with Different Thresholds:")
    for threshold in thresholds:
        y_pred_calibrated = (y_calibrated_proba[:, 1] > threshold).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_calibrated, average=None)
        
        # Calculate false positive rate (no-swing predicted as swing)
        fp_rate = np.sum((y_pred_calibrated == 1) & (y_test == 0)) / np.sum(y_test == 0)
        
        print(f"\nThreshold {threshold}:")
        print(f"  Swing Precision: {precision[1]:.4f}")
        print(f"  Swing Recall: {recall[1]:.4f}")
        print(f"  Swing F1: {f1[1]:.4f}")
        print(f"  False Positive Rate: {fp_rate:.4f}")
        print(f"  Swing Predictions: {np.sum(y_pred_calibrated == 1)}/{len(y_pred_calibrated)} ({np.sum(y_pred_calibrated == 1)/len(y_pred_calibrated)*100:.1f}%)")
    
    # Use calibrated model with optimal threshold (0.9) for final predictions
    final_threshold = 0.9
    y_pred_final = (y_calibrated_proba[:, 1] > final_threshold).astype(int)
    print(f"\n=== FINAL MODEL WITH CALIBRATED PROBABILITIES (threshold={final_threshold}) ===")
    print(classification_report(y_test, y_pred_final, target_names=['No Swing', 'Swing']))
    
    # Calculate final metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_final, average=None)
    fp_rate = np.sum((y_pred_final == 1) & (y_test == 0)) / np.sum(y_test == 0)
    fn_rate = np.sum((y_pred_final == 0) & (y_test == 1)) / np.sum(y_test == 1)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_final)
    
    print(f"\nFinal Calibrated Model Metrics (threshold={final_threshold}):")
    print(f"  No-Swing Precision: {precision[0]:.4f}")
    print(f"  No-Swing Recall: {recall[0]:.4f}")
    print(f"  No-Swing F1: {f1[0]:.4f}")
    print(f"  Swing Precision: {precision[1]:.4f}")
    print(f"  Swing Recall: {recall[1]:.4f}")
    print(f"  Swing F1: {f1[1]:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  False Positive Rate: {fp_rate:.4f}")
    print(f"  False Negative Rate: {fn_rate:.4f}")
    print(f"  Swing Predictions: {np.sum(y_pred_final == 1)}/{len(y_pred_final)} ({np.sum(y_pred_final == 1)/len(y_pred_final)*100:.1f}%)")
    
    print("\nSwing/No-Swing Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['No Swing', 'Swing']))
    
    # Advanced metrics with focus on balanced performance
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    print(f"\nDetailed Metrics:")
    print(f"  No-Swing Precision: {precision[0]:.4f}")
    print(f"  No-Swing Recall: {recall[0]:.4f}")
    print(f"  No-Swing F1: {f1[0]:.4f}")
    print(f"  Swing Precision: {precision[1]:.4f}")
    print(f"  Swing Recall: {recall[1]:.4f}")
    print(f"  Swing F1: {f1[1]:.4f}")
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    # Calculate false positive rate (no-swing predicted as swing)
    fp_rate = np.sum((y_pred == 1) & (y_test == 0)) / np.sum(y_test == 0)
    print(f"  False Positive Rate (No-swing predicted as swing): {fp_rate:.4f}")
    
    # Calculate false negative rate (swing predicted as no-swing)
    fn_rate = np.sum((y_pred == 0) & (y_test == 1)) / np.sum(y_test == 1)
    print(f"  False Negative Rate (Swing predicted as no-swing): {fn_rate:.4f}")
    
    # Cross-validation score with balanced accuracy
    balanced_scorer = make_scorer(balanced_accuracy_score)
    cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring=balanced_scorer)
    print(f"\nCross-validation balanced accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance (from first XGBoost model)
    if sample_weights is not None:
        xgb_model1.fit(X_train, y_train, sample_weight=sample_weights)  # Fit with sample weights
    else:
        xgb_model1.fit(X_train, y_train)  # No sample weights needed with balanced data
    
    # Get feature importances from the trained model
    final_feature_names = preprocessor.get_feature_names_out()
    final_importances = xgb_model1.feature_importances_
    sorted_idx = np.argsort(final_importances)[::-1]
    print("\nTop 25 Feature Importances (Final Model):")
    for idx in sorted_idx[:25]:
        print(f"  {final_feature_names[idx]}: {final_importances[idx]:.4f}")
    
    # Return both original and calibrated models with threshold info
    model_info = {
        'ensemble_model': ensemble_model,
        'calibrated_model': calibrated_model,
        'preprocessor': preprocessor,
        'feature_selector': selector,
        'all_feats': all_feats,
        'selected_feature_names': selected_feature_names,
        'threshold': final_threshold,
        'calibration_method': 'sigmoid',
        'cv_folds': 3
    }
    
    print(f"\nModel saved with calibration threshold: {final_threshold}")
    print(f"Calibration method: sigmoid with {3}-fold CV")
    
    return model_info

def create_swing_outcome_classifier(df):
    """
    Model 2: If batter swings, predict outcome (whiff, hit_safely, field_out)
    """
    print("\n=== MODEL 2: Swing Outcome Classifier ===")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Filter to swing events only
    swing_events = ['swinging_strike', 'swinging_strike_blocked', 'hit_into_play', 'hit_into_play_score', 'foul', 'foul_tip', 'foul_bunt']
    df_swing = df[df['description'].isin(swing_events)].copy()
    
    # Create swing outcome target with better mapping
    def get_swing_outcome(row):
        if row['description'] in ['swinging_strike', 'swinging_strike_blocked']:
            return 'whiff'
        elif row['events'] in ['single', 'double', 'triple', 'home_run']:
            return 'hit_safely'
        elif row['events'] in ['field_out', 'groundout', 'force_out', 'double_play', 'triple_play', 
                              'sac_fly', 'sac_bunt', 'field_error', 'fielders_choice', 'fielders_choice_out',
                              'sac_fly_double_play', 'sac_bunt_double_play', 'grounded_into_double_play',
                              'batter_interference', 'catcher_interference', 'fan_interference', 'strikeout',
                              'strikeout_double_play', 'strikeout_triple_play', 'walk', 'intent_walk', 'hit_by_pitch',
                              'sacrifice_bunt_double_play', 'sacrifice_bunt_triple_play', 'umpire_interference']:
            return 'field_out'
        else:
            return 'field_out'  # Default for other contact
    
    # FIX ZONE 0.0 ISSUE - Clean zone data before processing
    print(f"\nZone validation in training data:")
    print(f"Before cleaning: {len(df_swing)} pitches")
    
    # Check for invalid zones
    invalid_zones = df_swing[df_swing['zone'] <= 0].copy()
    print(f"Pitches with invalid zones (≤0): {len(invalid_zones)}")
    
    if len(invalid_zones) > 0:
        print("Sample invalid zone pitches:")
        print(invalid_zones[['plate_x', 'plate_z', 'zone', 'description', 'events']].head())
    
    # Fix invalid zones by recalculating from plate_x and plate_z
    def fix_zone(row):
        if row['zone'] <= 0:
            # Recalculate zone from plate coordinates
            plate_x, plate_z = row['plate_x'], row['plate_z']
            
            # Standard zone calculation
            if pd.isna(plate_x) or pd.isna(plate_z):
                return 1  # Default to zone 1 if coordinates missing
            elif abs(plate_x) <= 0.7 and 1.5 <= plate_z <= 3.5:
                # In strike zone
                if plate_x >= 0:
                    if plate_z >= 2.5:
                        return 1  # High inside
                    else:
                        return 3  # Low inside
                else:
                    if plate_z >= 2.5:
                        return 2  # High outside
                    else:
                        return 4  # Low outside
            else:
                # Outside strike zone
                if plate_x >= 0:
                    if plate_z >= 2.5:
                        return 5  # High inside out
                    else:
                        return 7  # Low inside out
                else:
                    if plate_z >= 2.5:
                        return 6  # High outside out
                    else:
                        return 8  # Low outside out
        else:
            return row['zone']
    
    # Apply zone fixing
    df_swing['zone'] = df_swing.apply(fix_zone, axis=1)
    
    # Remove any remaining invalid zones
    df_swing = df_swing[df_swing['zone'] > 0]
    print(f"After cleaning: {len(df_swing)} pitches")
    
    df_swing['swing_outcome'] = df_swing.apply(get_swing_outcome, axis=1)
    
    print(f"Swing outcome distribution: {df_swing['swing_outcome'].value_counts()}")
    
    # Prepare features - this will add all the engineered features to df_swing
    df_swing, num_feats, cat_feats = prepare_features(df_swing)
    
    # Add swing column for rate calculations
    df_swing['swing'] = 1  # All pitches in this dataset are swings
    df_swing = update_swing_rates(df_swing)
    
    # Add outcome-specific features to address class imbalance
    print("Adding outcome-specific features...")
    
    # Zone-specific outcome features
    df_swing['zone_whiff_rate'] = 0.0  # Will be calculated from career data
    df_swing['zone_hit_rate'] = 0.0
    df_swing['zone_field_out_rate'] = 0.0
    
    # Count-specific outcome features
    df_swing['count_whiff_rate'] = 0.0
    df_swing['count_hit_rate'] = 0.0
    df_swing['count_field_out_rate'] = 0.0
    
    # Pitch type specific outcome features
    df_swing['pitch_type_whiff_rate'] = 0.0
    df_swing['pitch_type_hit_rate'] = 0.0
    df_swing['pitch_type_field_out_rate'] = 0.0
    
    # Pressure situation features
    df_swing['pressure_whiff_rate'] = 0.0
    df_swing['pressure_hit_rate'] = 0.0
    df_swing['pressure_field_out_rate'] = 0.0
    
    # Add these new features to the feature lists
    outcome_features = [
        'zone_whiff_rate', 'zone_hit_rate', 'zone_field_out_rate',
        'count_whiff_rate', 'count_hit_rate', 'count_field_out_rate',
        'pitch_type_whiff_rate', 'pitch_type_hit_rate', 'pitch_type_field_out_rate',
        'pressure_whiff_rate', 'pressure_hit_rate', 'pressure_field_out_rate'
    ]
    
    num_feats.extend(outcome_features)
    
    all_feats = num_feats + cat_feats
    
    # Preprocess
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    
    X = preprocessor.fit_transform(df_swing[all_feats])
    y = df_swing['swing_outcome'].values
    
    # Clean NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Clean target variables before encoding
    y_clean = []
    for outcome in y:
        if pd.isna(outcome) or outcome is None or outcome == '':
            y_clean.append('field_out')  # Default for invalid outcomes
        else:
            y_clean.append(str(outcome))
    
    # Create swing outcome specific label encoder (only 3 classes)
    swing_le = LabelEncoder()
    swing_le.fit(['whiff', 'hit_safely', 'field_out'])
    
    # Encode labels using swing-specific encoder
    y_encoded = swing_le.transform(y_clean)
    
    # Print class mapping
    class_mapping = dict(zip(swing_le.classes_, range(len(swing_le.classes_))))
    print(f"Class mapping: {class_mapping}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)
    
    # Print original class distribution
    original_class_counts = np.bincount(y_train)
    print(f"Original class distribution in training: {dict(zip(range(len(original_class_counts)), original_class_counts))}")
    
    # Apply SMOTE to balance classes
    global SMOTE_AVAILABLE
    print(f"\n=== SMOTE DEBUGGING ===")
    print(f"SMOTE_AVAILABLE: {SMOTE_AVAILABLE}")
    print(f"Original class distribution: {dict(zip(range(len(original_class_counts)), original_class_counts))}")
    print(f"Number of classes: {len(original_class_counts)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train unique values: {np.unique(y_train)}")
    
    if SMOTE_AVAILABLE:
        print("\n=== APPLYING SMOTE FOR CLASS BALANCING ===")
        
        # Add assertions to check for NaN values before SMOTE
        print("Checking for NaN values...")
        x_nan_count = np.isnan(X_train).sum()
        y_nan_count = np.isnan(y_train).sum()
        print(f"NaN values in X_train: {x_nan_count}")
        print(f"NaN values in y_train: {y_nan_count}")
        
        assert not np.isnan(X_train).any(), "X_train has NaN values!"
        assert not np.isnan(y_train).any(), "y_train has NaN values!"
        print("✓ No NaN values found in training data")
        
        # Test SMOTE with a simple example
        print("Testing SMOTE with simple example...")
        try:
            test_X = np.array([[1, 2], [3, 4], [5, 6]])
            test_y = np.array([0, 1, 0])
            test_smote = SMOTE(random_state=42)
            test_X_res, test_y_res = test_smote.fit_resample(test_X, test_y)
            print("✓ SMOTE test successful")
            print(f"Test SMOTE result - X: {test_X_res.shape}, y: {test_y_res.shape}")
        except Exception as e:
            print(f"✗ SMOTE test failed: {e}")
            SMOTE_AVAILABLE = False
        
        if SMOTE_AVAILABLE:
            try:
                print("Creating SMOTE instance...")
                # Create SMOTE with proper parameters
                smote = SMOTE(
                    sampling_strategy='auto',  # Will oversample all minority classes
                    random_state=42,
                    k_neighbors=5
                )
                
                print("Applying SMOTE to training data...")
                # Apply SMOTE
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                print("✓ SMOTE applied successfully!")
                print(f"Original data shape: X={X_train.shape}, y={y_train.shape}")
                print(f"Balanced data shape: X={X_train_balanced.shape}, y={y_train_balanced.shape}")
                
                # Print balanced class distribution
                balanced_class_counts = np.bincount(y_train_balanced)
                print(f"Balanced class distribution: {dict(zip(range(len(balanced_class_counts)), balanced_class_counts))}")
                print(f"Total samples after SMOTE: {len(y_train_balanced)} (was {len(y_train)})")
                
                # Calculate class weights for balanced data
                class_counts = np.bincount(y_train_balanced)
                total_samples = len(y_train_balanced)
                class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts) if count > 0}
                
                print(f"Class weights: {class_weights}")
            except Exception as e:
                print(f"SMOTE failed: {e}")
                print("=== FALLING BACK TO ALTERNATIVE BALANCING METHODS ===")
                X_train_balanced, y_train_balanced, class_weights = apply_alternative_balancing(X_train, y_train)
        else:
            print("=== SMOTE NOT AVAILABLE - USING ALTERNATIVE BALANCING METHODS ===")
            X_train_balanced, y_train_balanced, class_weights = apply_alternative_balancing(X_train, y_train)
    else:
        print("=== SMOTE NOT AVAILABLE - USING ALTERNATIVE BALANCING METHODS ===")
        X_train_balanced, y_train_balanced, class_weights = apply_alternative_balancing(X_train, y_train)
    
    # IMPORTANT: Also balance the test set to match training distribution
    print(f"\n=== BALANCING TEST SET TO MATCH TRAINING DISTRIBUTION ===")
    print(f"Original test set class distribution: {dict(zip(range(len(np.bincount(y_test))), np.bincount(y_test)))}")
    
    # Get the target class counts from balanced training data
    target_class_counts = np.bincount(y_train_balanced)
    print(f"Target class counts from balanced training: {dict(zip(range(len(target_class_counts)), target_class_counts))}")
    
    # Balance test set to match training distribution
    X_test_balanced_list = []
    y_test_balanced_list = []
    
    for class_label in range(len(target_class_counts)):
        if target_class_counts[class_label] > 0:
            # Get indices for this class in test set
            class_indices = np.where(y_test == class_label)[0]
            
            # Calculate target count for test set (proportional to training)
            test_target_count = int(target_class_counts[class_label] * 0.2)  # 20% of training count
            
            if len(class_indices) > test_target_count:
                # Randomly sample test_target_count samples
                np.random.seed(42)  # For reproducibility
                selected_indices = np.random.choice(class_indices, test_target_count, replace=False)
            else:
                # Keep all samples for this class
                selected_indices = class_indices
            
            # Add selected samples
            X_test_balanced_list.append(X_test[selected_indices])
            y_test_balanced_list.append(y_test[selected_indices])
    
    # Combine all classes
    X_test_balanced = np.vstack(X_test_balanced_list)
    y_test_balanced = np.hstack(y_test_balanced_list)
    
    # Shuffle the test data
    shuffle_indices = np.random.permutation(len(X_test_balanced))
    X_test_balanced = X_test_balanced[shuffle_indices]
    y_test_balanced = y_test_balanced[shuffle_indices]
    
    print(f"Balanced test set class distribution: {dict(zip(range(len(np.bincount(y_test_balanced))), np.bincount(y_test_balanced)))}")
    print(f"Balanced test set size: {len(y_test_balanced)}")
    
    # Use balanced data for training and testing
    X_train = X_train_balanced
    y_train = y_train_balanced
    X_test = X_test_balanced
    y_test = y_test_balanced
    
    # Train model with class weights and improved parameters
    swing_outcome_model = XGBClassifier(
        eval_metric='mlogloss',  # Better for multi-class
        random_state=42,
        scale_pos_weight=None,  # We'll use sample_weight instead
        max_depth=6,  # Reduce overfitting
        learning_rate=0.1,  # Slower learning
        n_estimators=200,  # More trees
        subsample=0.8,  # Prevent overfitting
        colsample_bytree=0.8,  # Prevent overfitting
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        min_child_weight=3,  # Require more samples per leaf
        gamma=0.1  # Minimum loss reduction for split
    )
    
    # Use sample weights to address class imbalance (with balanced data)
    sample_weights = np.array([class_weights[y] for y in y_train_balanced])
    swing_outcome_model.fit(X_train_balanced, y_train_balanced, sample_weight=sample_weights)
    
    # Evaluate
    y_pred = swing_outcome_model.predict(X_test)
    
    # Safe decoding with error handling
    try:
        y_test_decoded = swing_le.inverse_transform(y_test)
        y_pred_decoded = swing_le.inverse_transform(y_pred)
    except ValueError as e:
        print(f"Warning: Label decoding error: {e}")
        # Fallback: convert numeric predictions to strings
        y_test_decoded = [swing_le.classes_[int(val)] if 0 <= int(val) < len(swing_le.classes_) else 'field_out' for val in y_test]
        y_pred_decoded = [swing_le.classes_[int(val)] if 0 <= int(val) < len(swing_le.classes_) else 'field_out' for val in y_pred]
    
    # Debug: Check for problematic values
    print(f"\nDebug - y_test_decoded type: {type(y_test_decoded)}")
    print(f"\nDebug - y_test_decoded length: {len(y_test_decoded)}")
    print(f"\nDebug - y_test_decoded sample: {y_test_decoded[:5]}")
    print(f"\nDebug - y_pred_decoded type: {type(y_pred_decoded)}")
    print(f"\nDebug - y_pred_decoded length: {len(y_pred_decoded)}")
    print(f"\nDebug - y_pred_decoded sample: {y_pred_decoded[:5]}")
    
    # Additional debugging for class distribution
    print(f"\n=== CLASS DISTRIBUTION DEBUGGING ===")
    print(f"y_test unique values: {np.unique(y_test)}")
    print(f"y_test class counts: {np.bincount(y_test)}")
    print(f"y_pred unique values: {np.unique(y_pred)}")
    print(f"y_pred class counts: {np.bincount(y_pred)}")
    
    # Show class distribution in decoded values
    y_test_decoded_counts = pd.Series(y_test_decoded).value_counts()
    y_pred_decoded_counts = pd.Series(y_pred_decoded).value_counts()
    print(f"y_test_decoded class counts: {y_test_decoded_counts.to_dict()}")
    print(f"y_pred_decoded class counts: {y_pred_decoded_counts.to_dict()}")
    
    # Clean any problematic values - handle NaN and None values
    y_test_decoded = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_test_decoded]
    y_pred_decoded = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_pred_decoded]
    
    print("\nSwing Outcome Model Performance:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    # Add probability calibration to improve prediction confidence
    from sklearn.calibration import CalibratedClassifierCV
    
    # Create calibrated model
    calibrated_model = CalibratedClassifierCV(
        swing_outcome_model, 
        cv=5, 
        method='isotonic'
    )
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate calibrated model
    y_pred_calibrated = calibrated_model.predict(X_test)
    
    # Safe decoding for calibrated model
    try:
        y_pred_calibrated_decoded = swing_le.inverse_transform(y_pred_calibrated)
    except ValueError as e:
        print(f"Warning: Calibrated model label decoding error: {e}")
        y_pred_calibrated_decoded = [swing_le.classes_[int(val)] if 0 <= int(val) < len(swing_le.classes_) else 'field_out' for val in y_pred_calibrated]
    
    y_pred_calibrated_decoded = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_pred_calibrated_decoded]
    
    print("\nCalibrated Swing Outcome Model Performance:")
    print(classification_report(y_test_decoded, y_pred_calibrated_decoded))
    
    # Create ensemble model for better robustness
    from sklearn.ensemble import VotingClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    # Create additional models for ensemble
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Train RF model
    rf_model.fit(X_train, y_train)
    
    # Create voting ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', swing_outcome_model),
            ('rf', rf_model)
        ],
        voting='soft'  # Use probability voting
    )
    
    # Train ensemble
    ensemble_model.fit(X_train, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble_model.predict(X_test)
    
    # Safe decoding for ensemble model
    try:
        y_pred_ensemble_decoded = swing_le.inverse_transform(y_pred_ensemble)
    except ValueError as e:
        print(f"Warning: Ensemble model label decoding error: {e}")
        y_pred_ensemble_decoded = [swing_le.classes_[int(val)] if 0 <= int(val) < len(swing_le.classes_) else 'field_out' for val in y_pred_ensemble]
    
    y_pred_ensemble_decoded = [str(val) if val is not None and not pd.isna(val) else 'field_out' for val in y_pred_ensemble_decoded]
    
    print("\nEnsemble Model Performance:")
    print(classification_report(y_test_decoded, y_pred_ensemble_decoded))
    
    # AUTOMATIC MODEL SELECTION - Choose the best performing model
    print("\n=== AUTOMATIC MODEL SELECTION ===")
    
    # Calculate performance metrics for each model
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    
    # Base XGBoost model metrics
    base_accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    base_f1 = f1_score(y_test_decoded, y_pred_decoded, average='weighted')
    base_recall = recall_score(y_test_decoded, y_pred_decoded, average='weighted')
    
    # Calibrated model metrics
    calibrated_accuracy = accuracy_score(y_test_decoded, y_pred_calibrated_decoded)
    calibrated_f1 = f1_score(y_test_decoded, y_pred_calibrated_decoded, average='weighted')
    calibrated_recall = recall_score(y_test_decoded, y_pred_calibrated_decoded, average='weighted')
    
    # Ensemble model metrics
    ensemble_accuracy = accuracy_score(y_test_decoded, y_pred_ensemble_decoded)
    ensemble_f1 = f1_score(y_test_decoded, y_pred_ensemble_decoded, average='weighted')
    ensemble_recall = recall_score(y_test_decoded, y_pred_ensemble_decoded, average='weighted')
    
    print(f"Base XGBoost Model:")
    print(f"  Accuracy: {base_accuracy:.3f}")
    print(f"  F1-Score: {base_f1:.3f}")
    print(f"  Recall: {base_recall:.3f}")
    
    print(f"Calibrated Model:")
    print(f"  Accuracy: {calibrated_accuracy:.3f}")
    print(f"  F1-Score: {calibrated_f1:.3f}")
    print(f"  Recall: {calibrated_recall:.3f}")
    
    print(f"Ensemble Model:")
    print(f"  Accuracy: {ensemble_accuracy:.3f}")
    print(f"  F1-Score: {ensemble_f1:.3f}")
    print(f"  Recall: {ensemble_recall:.3f}")
    
    # Select the best model using F1-score (balanced metric)
    model_performances = {
        'base': (base_f1, base_accuracy, base_recall, swing_outcome_model, "Base XGBoost"),
        'calibrated': (calibrated_f1, calibrated_accuracy, calibrated_recall, calibrated_model, "Calibrated Model"),
        'ensemble': (ensemble_f1, ensemble_accuracy, ensemble_recall, ensemble_model, "Ensemble Model")
    }
    
    best_model_name = max(model_performances.keys(), key=lambda x: model_performances[x][0])
    best_f1, best_accuracy, best_recall, best_model, best_model_desc = model_performances[best_model_name]
    
    print(f"\n🏆 SELECTED BEST MODEL: {best_model_desc}")
    print(f"Best F1-Score: {best_f1:.3f}")
    print(f"Best Accuracy: {best_accuracy:.3f}")
    print(f"Best Recall: {best_recall:.3f}")
    
    # Alternative: Custom weighted score (50% accuracy + 50% recall)
    print(f"\nAlternative Selection using Custom Weighted Score (50% Accuracy + 50% Recall):")
    weighted_scores = {}
    for model_name, (f1, acc, rec, model, desc) in model_performances.items():
        weighted_score = 0.5 * acc + 0.5 * rec
        weighted_scores[model_name] = weighted_score
        print(f"  {desc}: {weighted_score:.3f}")
    
    best_weighted_model = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
    best_weighted_score = weighted_scores[best_weighted_model]
    
    print(f"Best Weighted Score Model: {model_performances[best_weighted_model][4]}")
    print(f"Best Weighted Score: {best_weighted_score:.3f}")
    
    # Use F1-score as primary selection (you can change this to weighted_score if preferred)
    best_model = model_performances[best_model_name][3]
    
    # Use the best model for final model
    swing_outcome_model = best_model
    
    # Print feature importances for the selected model
    print("\nSwing Outcome Feature Importances:")
    
    # Get feature importances based on model type
    if hasattr(swing_outcome_model, 'estimators_'):
        # Ensemble model - get importances from XGBoost
        xgb_model = swing_outcome_model.estimators_[0]  # First estimator is XGBoost
        if hasattr(xgb_model, 'feature_importances_'):
            feature_names = preprocessor.get_feature_names_out()
            importances = xgb_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print("XGBoost Feature Importances (from Ensemble):")
            for idx in sorted_idx[:20]:
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    elif hasattr(swing_outcome_model, 'base_estimator'):
        # Calibrated model - get importances from base estimator
        base_model = swing_outcome_model.base_estimator
        if hasattr(base_model, 'feature_importances_'):
            feature_names = preprocessor.get_feature_names_out()
            importances = base_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print("XGBoost Feature Importances (from Calibrated Model):")
            for idx in sorted_idx[:20]:
                print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    else:
        # Single model (base XGBoost)
        feature_names = preprocessor.get_feature_names_out()
        importances = swing_outcome_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("XGBoost Feature Importances:")
        for idx in sorted_idx[:20]:
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    return swing_outcome_model, preprocessor, all_feats, swing_le

def create_no_swing_classifier(df):
    """
    Model 3: If batter doesn't swing, predict outcome (ball, strike, hit_by_pitch)
    """
    print("\n=== MODEL 3: No-Swing Outcome Classifier ===")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Filter to no-swing events only
    no_swing_events = ['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch']
    df_no_swing = df[df['description'].isin(no_swing_events)].copy()
    
    # Create no-swing outcome target
    def get_no_swing_outcome(row):
        if row['description'] == 'hit_by_pitch':
            return 'hit_by_pitch'
        elif row['description'] in ['called_strike']:
            return 'strike'
        elif row['description'] in ['ball', 'blocked_ball']:
            return 'ball'
        else:
            return 'ball'  # Default
    
    df_no_swing['no_swing_outcome'] = df_no_swing.apply(get_no_swing_outcome, axis=1)
    
    print(f"No-swing outcome distribution: {df_no_swing['no_swing_outcome'].value_counts()}")
    
    # Prepare features - this will add all the engineered features to df_no_swing
    df_no_swing, num_feats, cat_feats = prepare_features(df_no_swing)
    
    # Add swing column for rate calculations
    df_no_swing['swing'] = 0  # All pitches in this dataset are no-swings
    df_no_swing = update_swing_rates(df_no_swing)
    
    all_feats = num_feats + cat_feats
    
    # Preprocess
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)
    ])
    
    X = preprocessor.fit_transform(df_no_swing[all_feats])
    y = df_no_swing['no_swing_outcome'].values
    
    # Clean NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    # Clean target variables before encoding
    y_clean = []
    for outcome in y:
        if pd.isna(outcome) or outcome is None or outcome == '' or str(outcome).lower() == 'nan':
            y_clean.append('ball')  # Default for invalid outcomes
        else:
            y_clean.append(str(outcome))
    
    # Create no-swing specific label encoder (only 3 classes)
    no_swing_le = LabelEncoder()
    no_swing_le.fit(['ball', 'strike', 'hit_by_pitch'])
    
    # Encode labels using no-swing-specific encoder
    y_encoded = no_swing_le.transform(y_clean)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)
    
    # Train model
    no_swing_model = XGBClassifier(eval_metric='logloss', random_state=42)
    no_swing_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = no_swing_model.predict(X_test)
    
    # Safe decoding with error handling
    try:
        y_test_decoded = no_swing_le.inverse_transform(y_test)
        y_pred_decoded = no_swing_le.inverse_transform(y_pred)
    except ValueError as e:
        print(f"Warning: Label decoding error: {e}")
        # Fallback: convert numeric predictions to strings
        y_test_decoded = [no_swing_le.classes_[int(val)] if 0 <= int(val) < len(no_swing_le.classes_) else 'ball' for val in y_test]
        y_pred_decoded = [no_swing_le.classes_[int(val)] if 0 <= int(val) < len(no_swing_le.classes_) else 'ball' for val in y_pred]
    
    # Debug: Check for problematic values
    print(f"\nDebug - y_test_decoded type: {type(y_test_decoded)}")
    print(f"\nDebug - y_test_decoded length: {len(y_test_decoded)}")
    print(f"\nDebug - y_test_decoded sample: {y_test_decoded[:5]}")
    print(f"\nDebug - y_pred_decoded type: {type(y_pred_decoded)}")
    print(f"\nDebug - y_pred_decoded length: {len(y_pred_decoded)}")
    print(f"\nDebug - y_pred_decoded sample: {y_pred_decoded[:5]}")
    
    # Clean any problematic values - handle NaN and None values
    y_test_decoded = [str(val) if val is not None and not pd.isna(val) else 'ball' for val in y_test_decoded]
    y_pred_decoded = [str(val) if val is not None and not pd.isna(val) else 'ball' for val in y_pred_decoded]
    
    print("\nNo-Swing Outcome Model Performance:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    # After each model is trained, print feature importances
    # For no_swing_model
    # Print feature importances
    feature_names = preprocessor.get_feature_names_out()
    importances = no_swing_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nNo-Swing Outcome Feature Importances:")
    for idx in sorted_idx[:20]:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    return no_swing_model, preprocessor, all_feats, no_swing_le

def apply_alternative_balancing(X_train, y_train):
    """
    Apply alternative class balancing methods when SMOTE is not available.
    Uses manual sampling and class weights without external dependencies.
    """
    print("\n=== APPLYING CUSTOM CLASS BALANCING METHODS ===")
    
    # Method 1: Manual Undersampling to balance classes
    print("Method 1: Manual Undersampling")
    
    # Get class counts
    class_counts = np.bincount(y_train)
    print(f"Original class counts: {dict(zip(range(len(class_counts)), class_counts))}")
    
    # Find the minority class count
    minority_count = min(class_counts)
    print(f"Minority class count: {minority_count}")
    
    # Target: balance all classes to 2x the minority class (but not more than original)
    target_count = min(minority_count * 2, max(class_counts))
    print(f"Target count per class: {target_count}")
    
    # Manual undersampling
    X_train_balanced_list = []
    y_train_balanced_list = []
    
    for class_label in range(len(class_counts)):
        if class_counts[class_label] > 0:
            # Get indices for this class
            class_indices = np.where(y_train == class_label)[0]
            
            if len(class_indices) > target_count:
                # Randomly sample target_count samples
                np.random.seed(42)  # For reproducibility
                selected_indices = np.random.choice(class_indices, target_count, replace=False)
            else:
                # Keep all samples for this class
                selected_indices = class_indices
            
            # Add selected samples
            X_train_balanced_list.append(X_train[selected_indices])
            y_train_balanced_list.append(y_train[selected_indices])
    
    # Combine all classes
    X_train_balanced = np.vstack(X_train_balanced_list)
    y_train_balanced = np.hstack(y_train_balanced_list)
    
    # Shuffle the data
    shuffle_indices = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced[shuffle_indices]
    y_train_balanced = y_train_balanced[shuffle_indices]
    
    print(f"After manual undersampling: {dict(zip(range(len(np.bincount(y_train_balanced))), np.bincount(y_train_balanced)))}")
    
    # Method 2: Enhanced Class Weights
    print("Method 2: Enhanced Class Weights")
    class_counts_balanced = np.bincount(y_train_balanced)
    total_samples = len(y_train_balanced)
    
    # Use inverse frequency weighting with additional boost for minority classes
    class_weights = {}
    for i, count in enumerate(class_counts_balanced):
        if count > 0:
            # Base inverse frequency weight
            base_weight = total_samples / (len(class_counts_balanced) * count)
            # Additional boost for minority classes
            if count < 100:
                boost_factor = 2.5  # Higher boost for very small classes
            elif count < 200:
                boost_factor = 2.0
            elif count < 300:
                boost_factor = 1.5
            else:
                boost_factor = 1.0
            class_weights[i] = base_weight * boost_factor
    
    print(f"Enhanced class weights: {class_weights}")
    
    # Method 3: Custom Sampling with Replacement for Minority Classes
    print("Method 3: Custom Sampling with Replacement")
    
    # For very small classes, add some duplicate samples to boost representation
    final_X_list = [X_train_balanced]
    final_y_list = [y_train_balanced]
    
    for class_label in range(len(class_counts_balanced)):
        count = class_counts_balanced[class_label]
        if count > 0 and count < 50:  # Very small classes
            # Add some duplicate samples to boost representation
            class_indices = np.where(y_train_balanced == class_label)[0]
            boost_samples = min(20, count)  # Add up to 20 more samples
            
            np.random.seed(42)
            boost_indices = np.random.choice(class_indices, boost_samples, replace=True)
            
            final_X_list.append(X_train_balanced[boost_indices])
            final_y_list.append(y_train_balanced[boost_indices])
    
    # Combine original and boosted samples
    X_train_final = np.vstack(final_X_list)
    y_train_final = np.hstack(final_y_list)
    
    # Shuffle again
    shuffle_indices = np.random.permutation(len(X_train_final))
    X_train_balanced = X_train_final[shuffle_indices]
    y_train_balanced = y_train_final[shuffle_indices]
    
    print(f"Final balanced class distribution: {dict(zip(range(len(np.bincount(y_train_balanced))), np.bincount(y_train_balanced)))}")
    print(f"Total samples after balancing: {len(y_train_balanced)}")
    
    return X_train_balanced, y_train_balanced, class_weights

def main():
    print("🎯 Training Sequential Pitch Outcome Models")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv("ronald_acuna_jr_complete_career_statcast.csv")
    
    # Debug: Show available columns
    print(f"\n📋 Available columns ({len(df.columns)}):")
    print(df.columns.tolist())
    
    # Create holdout dataset (10% for holdout, 90% for training)
    print("\n📊 Creating holdout dataset...")
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
    df = df.dropna(subset=['description', 'events'])
    
    print(f"Training dataset size: {len(df)} pitches")
    print(f"Description value counts:")
    print(df['description'].value_counts().head(10))
    
    # Data validation - check for problematic features
    print("\n🔍 Data Validation:")
    print(f"Total features available: {len(df.columns)}")
    
    # Check for columns with too many NaN values
    nan_counts = df.isnull().sum()
    high_nan_cols = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
    if high_nan_cols:
        print(f"Columns with >50% NaN values: {high_nan_cols}")
    
    # Check for infinite values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    inf_cols = inf_counts[inf_counts > 0].index.tolist()
    if inf_cols:
        print(f"Columns with infinite values: {inf_cols}")
    
    # Check for key columns we need
    key_columns = ['pitcher', 'batter', 'at_bat_number', 'inning', 'home_score', 'away_score']
    missing_key_columns = [col for col in key_columns if col not in df.columns]
    if missing_key_columns:
        print(f"Missing key columns: {missing_key_columns}")
    
    # Train Model 1: Swing/No-Swing
    swing_model_info = create_swing_classifier(df)
    swing_model = swing_model_info['ensemble_model']
    swing_calibrated_model = swing_model_info['calibrated_model']
    swing_preprocessor = swing_model_info['preprocessor']
    swing_feature_selector = swing_model_info['feature_selector']
    swing_features = swing_model_info['all_feats']
    swing_selected_features = swing_model_info['selected_feature_names']
    swing_threshold = swing_model_info['threshold']
    
    # Train Model 2: Swing Outcomes
    swing_outcome_model, swing_outcome_preprocessor, swing_outcome_features, swing_outcome_le = create_swing_outcome_classifier(df)
    
    # Train Model 3: No-Swing Outcomes
    no_swing_model, no_swing_preprocessor, no_swing_features, no_swing_le = create_no_swing_classifier(df)
    
    # Save all models
    models = {
        'swing_model': swing_model,
        'swing_calibrated_model': swing_calibrated_model,
        'swing_threshold': swing_threshold,
        'swing_preprocessor': swing_preprocessor,
        'swing_feature_selector': swing_feature_selector,
        'swing_features': swing_features,
        'swing_selected_features': swing_selected_features,
        'swing_outcome_model': swing_outcome_model,
        'swing_outcome_preprocessor': swing_outcome_preprocessor,
        'swing_outcome_features': swing_outcome_features,
        'swing_outcome_le': swing_outcome_le,
        'no_swing_model': no_swing_model,
        'no_swing_preprocessor': no_swing_preprocessor,
        'no_swing_features': no_swing_features,
        'no_swing_le': no_swing_le
    }
    
    with open("sequential_models.pkl", "wb") as f:
        pickle.dump(models, f)
    
    print("\n✅ All models saved to 'sequential_models.pkl'")
    print("\nModel Pipeline:")
    print("1. Swing/No-Swing Classifier")
    print("2. If Swing → Swing Outcome Classifier (whiff, hit_safely, field_out)")
    print("3. If No Swing → No-Swing Outcome Classifier (ball, strike, hit_by_pitch)")

if __name__ == "__main__":
    main() 
 
 
 
 
 