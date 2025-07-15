import json
import os
import pandas as pd
import numpy as np
import joblib
import pickle
from test_pitch_json_classifier import load_career_data, calculate_hitter_features, load_average_metrics, prepare_pitch_features
import warnings
warnings.filterwarnings('ignore')
import traceback

class AtBatSimulation:
    def __init__(self, hitter_name='acuna', pitcher_name='unknown'):
        """Initialize at-bat simulation with hitter and pitcher"""
        self.hitter_name = hitter_name
        self.pitcher_name = pitcher_name
        self.current_count = {'balls': 0, 'strikes': 0}
        self.at_bat_outcome = None
        self.pitch_sequence = []
        
        # Load pitcher averages
        self.pitcher_averages = self.load_pitcher_averages()
        
        # Load hitter averages CSV for advanced stats
        try:
            averages_csv = "ronald_acuna_jr_averages.csv"
            if os.path.exists(averages_csv):
                self.hitter_averages = pd.read_csv(averages_csv)
                print(f"✓ Loaded hitter averages from {averages_csv}")
            else:
                self.hitter_averages = None
                print(f"⚠️ Hitter averages CSV not found: {averages_csv}")
        except Exception as e:
            print(f"✗ Error loading hitter averages CSV: {e}")
            self.hitter_averages = None

        # Load models
        try:
            with open('sequential_models.pkl', 'rb') as f:
                self.models = pickle.load(f)
            
            self.swing_classifier = self.models.get('swing_calibrated_model')
            self.swing_preprocessor = self.models.get('swing_preprocessor')
            self.swing_threshold = self.models.get('swing_threshold', 0.9)
            
            if self.swing_classifier is None:
                print("✗ Swing classifier not found in model file")
                return
            
            print("✓ Loaded swing vs no swing model")
            
            # Load career data for feature calculation
            self.hitter_df, self.pitcher_df = load_career_data()
            if self.hitter_df is not None:
                self.hitter_features = calculate_hitter_features(self.hitter_df)
                print(f"✓ Loaded {len(self.hitter_features)} hitter features from career data")
            else:
                self.hitter_features = {}
                print("⚠️ Using default hitter features")
            
            # Load average metrics
            self.averages = load_average_metrics()
            print("✓ Loaded average metrics")
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            self.swing_classifier = None
            self.swing_preprocessor = None
            self.hitter_features = {}
            self.averages = {}
        # Load whiff vs contact model
        try:
            self.whiff_model = joblib.load('whiff_vs_contact_model.pkl')
            self.whiff_preprocessor = joblib.load('whiff_vs_contact_preprocessor.pkl')
            print("✓ Loaded whiff vs contact model")
        except Exception as e:
            print(f"✗ Error loading whiff vs contact model: {e}")
            self.whiff_model = None
            self.whiff_preprocessor = None
    
    def load_pitcher_averages(self):
        """Load pitcher-specific pitch averages from CSV file"""
        try:
            # Try to load the specific pitcher's averages
            csv_filename = f"{self.pitcher_name.lower().replace(' ', '_')}_pitch_averages.csv"
            if os.path.exists(csv_filename):
                pitcher_df = pd.read_csv(csv_filename)
                print(f"✓ Loaded pitcher averages from {csv_filename}")
                return pitcher_df
            else:
                # Fallback to Sandy Alcantara's averages if specific pitcher not found
                fallback_filename = "sandy_alcantara_pitch_averages.csv"
                if os.path.exists(fallback_filename):
                    pitcher_df = pd.read_csv(fallback_filename)
                    print(f"✓ Loaded fallback pitcher averages from {fallback_filename}")
                    return pitcher_df
                else:
                    print("⚠️ No pitcher averages file found")
                    return None
        except Exception as e:
            print(f"✗ Error loading pitcher averages: {e}")
            return None
    
    def get_pitcher_averages_for_pitch_type(self, pitch_type):
        """Get pitcher-specific averages for a given pitch type"""
        if self.pitcher_averages is None:
            return {}
        
        # Filter to the specific pitch type
        pitch_averages = self.pitcher_averages[self.pitcher_averages['pitch_type'] == pitch_type]
        
        if len(pitch_averages) == 0:
            print(f"⚠️ No averages found for pitch type {pitch_type}")
            return {}
        
        # Return the first row as a dictionary
        return pitch_averages.iloc[0].to_dict()
    
    def reset_at_bat(self):
        """Reset the at-bat to 0-0 count, but do NOT clear pitch_sequence here"""
        self.current_count = {'balls': 0, 'strikes': 0}
        self.at_bat_outcome = None
        # self.pitch_sequence = []  # Do NOT clear here; let frontend clear on new at-bat
        print("At-bat reset to 0-0 count")
    
    def get_pitch_for_count(self, balls, strikes):
        """Get the pitch JSON file that matches the current count"""
        pitches_dir = 'Pitchers/pitches'
        if not os.path.exists(pitches_dir):
            print(f"✗ Pitches directory not found: {pitches_dir}")
            return None
        
        # Look for pitch files with matching count
        for filename in os.listdir(pitches_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(pitches_dir, filename), 'r') as f:
                        pitch_data = json.load(f)
                    
                    # Check if this pitch matches the count
                    if (pitch_data.get('strikes') == strikes and 
                        pitch_data.get('balls') == balls):
                        print(f"✓ Found pitch for count {balls}-{strikes}: {filename}")
                        return pitch_data
                except Exception as e:
                    print(f"Error reading pitch file {filename}: {e}")
                    continue
        
        print(f"✗ No pitch found for count {balls}-{strikes}")
        return None
    
    def generate_features_for_pitch(self, pitch_data):
        """Generate all features needed for the swing vs no swing model"""
        # FIX ZONE 0.0 ISSUE - Validate zone data
        if 'zone' in pitch_data and pitch_data['zone'] <= 0:
            print(f"⚠️  Invalid zone detected: {pitch_data['zone']}")
            
            # Recalculate zone from plate coordinates
            plate_x = pitch_data.get('plate_x', 0)
            plate_z = pitch_data.get('plate_z', 2.5)
            
            if pd.isna(plate_x) or pd.isna(plate_z):
                pitch_data['zone'] = 1  # Default to zone 1
            elif abs(plate_x) <= 0.7 and 1.5 <= plate_z <= 3.5:
                # In strike zone
                if plate_x >= 0:
                    pitch_data['zone'] = 1 if plate_z >= 2.5 else 3
                else:
                    pitch_data['zone'] = 2 if plate_z >= 2.5 else 4
            else:
                # Outside strike zone
                if plate_x >= 0:
                    pitch_data['zone'] = 5 if plate_z >= 2.5 else 7
                else:
                    pitch_data['zone'] = 6 if plate_z >= 2.5 else 8
            
            print(f"✅ Fixed zone to: {pitch_data['zone']}")
        
        # Get pitcher averages for this pitch type
        pitch_type = pitch_data.get('pitch_type', 'FF')
        pitcher_averages = self.get_pitcher_averages_for_pitch_type(pitch_type)
        
        # Merge pitcher averages with pitch data
        if pitcher_averages:
            # Override pitch data with pitcher averages for key metrics
            for key, value in pitcher_averages.items():
                if key not in ['pitch_type'] and not pd.isna(value):  # Skip pitch_type and NaN values
                    pitch_data[key] = value
            print(f"✓ Applied pitcher averages for {pitch_type}")

        print(pitch_data)
        
        # Prepare features using the same approach as the test script
        print("🔧 Generating features for swing vs no swing model...")
        
        # Use the same feature preparation as the test script
        features = prepare_pitch_features(pitch_data, self.averages)
        
        # Update features with actual career data
        print(self.hitter_features)
        if self.hitter_features:
            for feature_name, value in self.hitter_features.items():
                if feature_name in features:
                    features[feature_name] = value
            print(f"✓ Applied {len(self.hitter_features)} hitter-specific features")
        
        # Create DataFrame with features
        df = pd.DataFrame([features])
        
        print(f"✓ Generated {len(df.columns)} features")
        return df
    
    def predict_swing_vs_noswing(self, pitch_data):
        """Predict swing vs no swing for a given pitch"""
        if self.swing_classifier is None:
            print("✗ No swing model loaded")
            return None
        
        # Generate features
        df = self.generate_features_for_pitch(pitch_data)
        
        # Transform using the preprocessor (like the test script)
        try:
            # Filter features to only include what the preprocessor expects
            available_feats = [f for f in self.swing_preprocessor.feature_names_in_ if f in df.columns]
            missing_feats = [f for f in self.swing_preprocessor.feature_names_in_ if f not in df.columns]
            
            if missing_feats:
                # Add default values for missing features
                for feat in missing_feats:
                    df[feat] = 0.0
            
            # Transform the features using only what the preprocessor expects
            X_swing = self.swing_preprocessor.transform(df[self.swing_preprocessor.feature_names_in_])
            
            # Handle any remaining NaN values in the transformed data
            X_swing = np.nan_to_num(X_swing, nan=0.0)
            
            # Get prediction probability
            swing_probability = self.swing_classifier.predict_proba(X_swing)[0][1]
            
            # Apply count-specific thresholds to reduce early count false positives
            count_thresholds = {
                '0-0': 0.95,
                '0-1': 0.90,
                '0-2': 0.55,
                '1-0': 0.95,
                '1-1': 0.95,
                '1-2': 0.55,
                '2-0': 0.50,
                '2-1': 0.80,
                '2-2': 0.55,
                '3-0': 0.95,
                '3-1': 0.50,
                '3-2': 0.50,
                'default': 0.9  # Fallback default
            }
            
            # Determine count situation
            balls = pitch_data.get('balls', 0)
            strikes = pitch_data.get('strikes', 0)
            
            # Create count key
            count_key = f"{balls}-{strikes}"
            
            # Get threshold for specific count
            if count_key in count_thresholds:
                threshold = count_thresholds[count_key]
                count_situation = count_key
            else:
                threshold = count_thresholds['default']
                count_situation = "default"
            
            # Make prediction with count-specific threshold
            is_swing = swing_probability >= threshold
            noswing_prob = 1 - swing_probability
            
            result = {
                'is_swing': is_swing,
                'swing_probability': swing_probability,
                'noswing_probability': noswing_prob,
                'prediction_confidence': max(swing_probability, noswing_prob),
                'threshold': threshold,
                'count_situation': count_situation
            }
            
            print(f"Prediction: {'SWING' if is_swing else 'NO SWING'}")
            print(f"Confidence: {result['prediction_confidence']:.3f}")
            print(f"Threshold: {threshold:.3f} ({count_situation})")
            
            return result
            
        except Exception as e:
            print(f"✗ Error making prediction: {e}")
            return None
    
    def determine_ball_or_strike(self, pitch_data):
        """Determine if a no-swing pitch is a ball or strike based on location"""
        plate_x = pitch_data.get('plate_x', 0)
        plate_z = pitch_data.get('plate_z', 2.5)
        sz_top = pitch_data.get('sz_top', 3.5)
        sz_bot = pitch_data.get('sz_bot', 1.5)
        
        # Check if pitch is in strike zone
        in_strike_zone = (
            abs(plate_x) <= 0.85 and  # Within horizontal bounds
            sz_bot <= plate_z <= sz_top  # Within vertical bounds
        )
        
        return 'strike' if in_strike_zone else 'ball'
    
    def update_count(self, result):
        """Update the ball-strike count based on the pitch result"""
        if result == 'ball':
            self.current_count['balls'] += 1
            print(f"Ball! Count: {self.current_count['balls']}-{self.current_count['strikes']}")
        elif result == 'strike':
            self.current_count['strikes'] += 1
            print(f"Strike! Count: {self.current_count['balls']}-{self.current_count['strikes']}")
        elif result == 'swing':
            # For swings, we need to determine the outcome (this will be expanded later)
            print(f"Swing! Count: {self.current_count['balls']}-{self.current_count['strikes']}")
        
        # Check for at-bat ending conditions
        if self.current_count['strikes'] >= 3:
            self.at_bat_outcome = 'strikeout'
            print("⚡ STRIKEOUT! At-bat over.")
        elif self.current_count['balls'] >= 4:
            self.at_bat_outcome = 'walk'
            print("🚶 WALK! At-bat over.")
    
    def prepare_whiff_features(self, pitch_data):
        """Prepare features for whiff vs contact model from a single pitch dict"""
        import numpy as np
        import pandas as pd
        d = pitch_data.copy()
        # Fill in missing values with defaults
        d.setdefault('sz_top', 3.5)
        d.setdefault('sz_bot', 1.5)
        d.setdefault('plate_x', 0.0)
        d.setdefault('plate_z', 2.5)
        d.setdefault('balls', 0)
        d.setdefault('strikes', 0)
        d.setdefault('pitch_type', 'FF')
        d.setdefault('release_speed', 90.0)
        d.setdefault('api_break_x_batter_in', 0.0)
        d.setdefault('api_break_z_with_gravity', 0.0)
        d.setdefault('zone', 5)
        # Feature engineering (adapted from test_whiff_vs_contact_model.py)
        zone_center_x = 0
        zone_center_z = (d['sz_top'] + d['sz_bot']) / 2
        zone_distance = np.sqrt((d['plate_x'] - zone_center_x) ** 2 + (d['plate_z'] - zone_center_z) ** 2)
        horizontal_break = d.get('api_break_x_batter_in', 0)
        vertical_break = d.get('api_break_z_with_gravity', 0)
        movement_magnitude = np.sqrt(horizontal_break ** 2 + vertical_break ** 2)
        count_pressure = d['balls'] - d['strikes']
        count_total = d['balls'] + d['strikes']
        behind_in_count = int(d['strikes'] > d['balls'])
        ahead_in_count = int(d['balls'] > d['strikes'])
        two_strikes = int(d['strikes'] >= 2)
        three_balls = int(d['balls'] >= 3)
        in_strike_zone = int((-0.85 <= d['plate_x'] <= 0.85 and d['sz_bot'] <= d['plate_z'] <= d['sz_top']))
        far_from_zone = int(zone_distance > 1.0)
        high_pitch = int(d['plate_z'] > d['sz_top'])
        low_pitch = int(d['plate_z'] < d['sz_bot'])
        inside_pitch = int(d['plate_x'] < -0.85)
        outside_pitch = int(d['plate_x'] > 0.85)
        is_fastball = int(d['pitch_type'] in ['FF', 'SI', 'FC'])
        is_breaking_ball = int(d['pitch_type'] in ['SL', 'CU', 'KC'])
        is_offspeed = int(d['pitch_type'] in ['CH', 'FS'])
        high_velocity = int(d['release_speed'] > 95)
        low_velocity = int(d['release_speed'] < 85)
        velocity_movement_ratio = d['release_speed'] / (movement_magnitude + 0.1)
        high_movement = int(movement_magnitude > 6)
        low_movement = int(movement_magnitude < 2)
        movement_ratio = abs(horizontal_break) / (abs(vertical_break) + 0.1)
        zone_distance_x_count_pressure = zone_distance * count_pressure
        movement_x_count_pressure = movement_magnitude * count_pressure
        in_zone_x_two_strikes = in_strike_zone * two_strikes
        far_from_zone_x_ahead = far_from_zone * ahead_in_count
        velocity_diff_from_avg = d['release_speed'] - 90.0
        movement_diff_from_avg = movement_magnitude - 5.0
        zone_corner = int(d['zone'] in [1, 3, 7, 9])
        zone_heart = int(d['zone'] in [2, 5, 8])
        zone_shadow = int(d['zone'] in [4, 6])
        # BABIP features (use defaults for now)
        batting_average_bip = 0.25
        whiff_rate = 0.35
        field_out_rate_bip = 0.40
        balls_in_play = 0
        total_swings = 0
        total_whiffs = 0
        # Compose feature dict
        features = {
            'zone_center_x': zone_center_x,
            'zone_center_z': zone_center_z,
            'zone_distance': zone_distance,
            'horizontal_break': horizontal_break,
            'vertical_break': vertical_break,
            'movement_magnitude': movement_magnitude,
            'count_pressure': count_pressure,
            'count_total': count_total,
            'behind_in_count': behind_in_count,
            'ahead_in_count': ahead_in_count,
            'two_strikes': two_strikes,
            'three_balls': three_balls,
            'in_strike_zone': in_strike_zone,
            'far_from_zone': far_from_zone,
            'high_pitch': high_pitch,
            'low_pitch': low_pitch,
            'inside_pitch': inside_pitch,
            'outside_pitch': outside_pitch,
            'is_fastball': is_fastball,
            'is_breaking_ball': is_breaking_ball,
            'is_offspeed': is_offspeed,
            'high_velocity': high_velocity,
            'low_velocity': low_velocity,
            'velocity_movement_ratio': velocity_movement_ratio,
            'high_movement': high_movement,
            'low_movement': low_movement,
            'movement_ratio': movement_ratio,
            'zone_distance_x_count_pressure': zone_distance_x_count_pressure,
            'movement_x_count_pressure': movement_x_count_pressure,
            'in_zone_x_two_strikes': in_zone_x_two_strikes,
            'far_from_zone_x_ahead': far_from_zone_x_ahead,
            'velocity_diff_from_avg': velocity_diff_from_avg,
            'movement_diff_from_avg': movement_diff_from_avg,
            'zone_corner': zone_corner,
            'zone_heart': zone_heart,
            'zone_shadow': zone_shadow,
            'batting_average_bip': batting_average_bip,
            'whiff_rate': whiff_rate,
            'field_out_rate_bip': field_out_rate_bip,
            'balls_in_play': balls_in_play,
            'total_swings': total_swings,
            'total_whiffs': total_whiffs,
            'pitch_type': d['pitch_type'],
            'zone': d['zone'],
        }
        # Prepare DataFrame for model
        X = pd.DataFrame([features])
        # Handle categorical features
        cat_features = self.whiff_preprocessor['cat_features']
        for feat in cat_features:
            if feat in X.columns:
                X[feat] = X[feat].fillna('unknown').astype(str)
        # Handle numeric features
        num_features = self.whiff_preprocessor['num_features']
        for feat in num_features:
            if feat in X.columns:
                X[feat] = pd.to_numeric(X[feat], errors='coerce').fillna(0)
        # Ensure all required columns are present
        required_cols = self.whiff_preprocessor['num_features'] + self.whiff_preprocessor['cat_features']
        for col in required_cols:
            if col not in X.columns:
                if col in self.whiff_preprocessor['cat_features']:
                    X[col] = 'unknown'
                else:
                    X[col] = 0
        # Reorder columns to match preprocessor expectation
        X = X[required_cols]
        return X

    def simulate_pitch(self, pitch_data):
        """Simulate a single pitch and update the count"""
        print(f"\n🎯 Simulating pitch at count {self.current_count['balls']}-{self.current_count['strikes']}")
        print(f"Pitch type: {pitch_data.get('pitch_type', 'Unknown')}")
        print(f"Location: ({pitch_data.get('plate_x', 0):.2f}, {pitch_data.get('plate_z', 0):.2f})")

        # Predict swing vs no swing
        swing_result = self.predict_swing_vs_noswing(pitch_data)
        if swing_result is None:
            outcome = 'error'
            prediction_confidence = 0.0
            details = 'Prediction failed.'
        else:
            is_swing = swing_result['is_swing']
            prediction_confidence = swing_result['prediction_confidence']
            threshold = swing_result['threshold']
            count_situation = swing_result['count_situation']
            details = f"Confidence: {prediction_confidence:.3f}, Threshold: {threshold:.3f} ({count_situation})"

            if is_swing:
                # --- Whiff vs Contact Integration ---
                if self.whiff_model is not None and self.whiff_preprocessor is not None:
                    # Use the new comprehensive whiff vs contact prediction method
                    whiff_result = self.predict_whiff_vs_contact(pitch_data)
                    
                    if whiff_result is not None:
                        whiff_proba = whiff_result['whiff_probability']
                        contact_proba = whiff_result['contact_probability']
                        whiff_pred = whiff_result['is_whiff']
                        details += f' | Whiff prob: {whiff_proba:.2f}, Contact prob: {contact_proba:.2f}'
                        
                        if whiff_pred:
                            # SWING & MISS
                            self.current_count['strikes'] += 1
                            if self.current_count['strikes'] >= 3:
                                outcome = 'whiff'
                                at_bat_outcome = 'strikeout'
                                details += ' | SWING & MISS (strikeout)'
                            else:
                                outcome = 'whiff'
                                at_bat_outcome = None
                                details += ' | SWING & MISS (strike)'
                        else:
                            # CONTACT
                            outcome = 'contact'
                            at_bat_outcome = 'contact'
                            details += ' | CONTACT (at-bat ends)'
                    else:
                        # Fallback if prediction fails
                        self.current_count['strikes'] += 1
                        if self.current_count['strikes'] >= 3:
                            outcome = 'swing'
                            at_bat_outcome = 'strikeout'
                            details += ' | SWING (strikeout)'
                        else:
                            outcome = 'swing'
                            at_bat_outcome = None
                            details += ' | SWING (strike assumed for demo)'
                else:
                    self.current_count['strikes'] += 1
                    if self.current_count['strikes'] >= 3:
                        outcome = 'swing'
                        at_bat_outcome = 'strikeout'
                        details += ' | SWING (strikeout)'
                    else:
                        outcome = 'swing'
                        at_bat_outcome = None
                        details += ' | SWING (strike assumed for demo)'
            else:
                # Determine if it's a ball or strike
                ball_or_strike = self.determine_ball_or_strike(pitch_data)
                outcome = ball_or_strike
                self.update_count(ball_or_strike)
                details += f' | NO SWING - {ball_or_strike.upper()}'
                # If at-bat ends due to called strikeout or walk, set outcome
                if self.current_count['strikes'] >= 3:
                    at_bat_outcome = 'strikeout'
                elif self.current_count['balls'] >= 4:
                    at_bat_outcome = 'walk'
                else:
                    at_bat_outcome = None

        # Always append the actual pitch to the sequence
        self.pitch_sequence.append({
            'outcome': outcome,
            'confidence': prediction_confidence,
            'details': details,
            'count': dict(self.current_count),
            'at_bat_outcome': at_bat_outcome,
            'pitch_type': pitch_data.get('pitch_type', None)
        })

        # If at-bat ends, append a summary and reset at-bat (but do NOT clear pitch_sequence)
        if at_bat_outcome:
            self.pitch_sequence.append({
                'outcome': at_bat_outcome,
                'confidence': 1.0,
                'details': f'At-bat ended with {at_bat_outcome.upper()}',
                'count': dict(self.current_count),
                'at_bat_outcome': at_bat_outcome
            })
            self.reset_at_bat()

        result = {
            'outcome': outcome,
            'confidence': prediction_confidence,
            'details': details,
            'count': dict(self.current_count),
            'at_bat_outcome': at_bat_outcome,
        }
        # Always return the full pitch sequence
        result['pitch_sequence'] = self.pitch_sequence
        return result
    
    def get_at_bat_summary(self):
        """Get summary of the current at-bat"""
        return {
            'current_count': self.current_count,
            'at_bat_outcome': self.at_bat_outcome,
            'total_pitches': len(self.pitch_sequence),
            'pitch_sequence': self.pitch_sequence
        }

    def get_hitter_avg_stat(self, pitch_type, stat):
        """Get stat from hitter averages for pitch_type, fallback to ALL, else 0.0"""
        if self.hitter_averages is None:
            print(f"[DEBUG] hitter_averages is None!")
            return 0.0
        # Normalize pitch_type for robust matching
        pt = str(pitch_type).strip().upper()
        df = self.hitter_averages
        # Normalize the DataFrame column for matching
        df['pitch_type_norm'] = df['pitch_type'].astype(str).str.strip().str.upper()
        print(f"[DEBUG] Requested pitch_type: '{pitch_type}' (normalized: '{pt}'), stat: '{stat}'")
        print(f"[DEBUG] Available pitch_type_norm values: {df['pitch_type_norm'].unique()}")
        row = df[df['pitch_type_norm'] == pt]
        print(f"[DEBUG] Row found for '{pt}': {not row.empty}")
        if row.empty:
            row = df[df['pitch_type_norm'] == 'ALL']
            print(f"[DEBUG] Row found for 'ALL': {not row.empty}")
        if not row.empty and stat in row.columns:
            val = row.iloc[0][stat]
            print(f"[DEBUG] Value found for stat '{stat}': {val} (NaN: {pd.isna(val)})")
            if pd.isna(val):
                return 0.0
            return val
        print(f"[DEBUG] Stat '{stat}' not found in columns: {list(row.columns) if not row.empty else 'No row'}")
        return 0.0

    def predict_whiff_vs_contact(self, pitch_data):
        """Predict whiff vs contact for a swing"""
        try:
            print("\n" + "="*50)
            print("🔍 DEBUG: WHIFF VS CONTACT PREDICTION")
            print("="*50)
            
            # Check model loading status
            print(f"📊 Model Status:")
            print(f"  - Whiff model loaded: {hasattr(self, 'whiff_model') and self.whiff_model is not None}")
            print(f"  - Whiff preprocessor loaded: {hasattr(self, 'whiff_preprocessor') and self.whiff_preprocessor is not None}")
            if hasattr(self, 'whiff_model_features'):
                print(f"  - Expected features: {len(self.whiff_model_features) if self.whiff_model_features else 'None'}")
            
            # Apply pitcher averages for the specific pitch type
            pitch_type = pitch_data.get('pitch_type', 'FF')
            print(f"\n🎯 Pitch Type: {pitch_type}")
            print(f"📈 Original pitch data: {pitch_data}")
            
            if self.pitcher_averages is not None and pitch_type in self.pitcher_averages:
                pitcher_avg = self.pitcher_averages[pitch_type]
                print(f"📊 Pitcher averages for {pitch_type}: {pitcher_avg}")
                
                # Override pitch data with pitcher averages
                pitch_data.update({
                    'release_speed': pitcher_avg.get('avg_release_speed', pitch_data.get('release_speed', 90)),
                    'release_spin_rate': pitcher_avg.get('avg_release_spin_rate', pitch_data.get('release_spin_rate', 2200)),
                    'release_extension': pitcher_avg.get('avg_release_extension', pitch_data.get('release_extension', 6.5)),
                    'pfx_x': pitcher_avg.get('avg_pfx_x', pitch_data.get('pfx_x', 0)),
                    'pfx_z': pitcher_avg.get('avg_pfx_z', pitch_data.get('pfx_z', 0)),
                    'plate_x': pitcher_avg.get('avg_plate_x', pitch_data.get('plate_x', 0)),
                    'plate_z': pitcher_avg.get('avg_plate_z', pitch_data.get('plate_z', 2.5)),
                    'vx0': pitcher_avg.get('avg_vx0', pitch_data.get('vx0', 0)),
                    'vy0': pitcher_avg.get('avg_vy0', pitch_data.get('vy0', -130)),
                    'vz0': pitcher_avg.get('avg_vz0', pitch_data.get('vz0', -5)),
                    'ax': pitcher_avg.get('avg_ax', pitch_data.get('ax', 0)),
                    'ay': pitcher_avg.get('avg_ay', pitch_data.get('ay', 25)),
                    'az': pitcher_avg.get('avg_az', pitch_data.get('az', -15)),
                })
                print(f"✓ Applied pitcher averages for {pitch_type} in whiff vs contact prediction")
            else:
                print(f"⚠️ No pitcher averages found for {pitch_type}")
            
            print(f"📈 Updated pitch data: {pitch_data}")
            
            # Use the same feature generation as swing vs no swing model
            print("\n🔧 Using same feature generation as swing vs no swing model...")
            feature_df = self.generate_features_for_pitch(pitch_data)
            print(f"📊 Base feature DataFrame shape: {feature_df.shape}")
            
            # Add any additional features that the whiff model needs
            print("\n🔧 Adding whiff-specific features...")
            
            # Calculate additional features that might be needed for whiff model
            additional_features = {
                'low_movement': 1.0 if feature_df.get('movement_magnitude', 0).iloc[0] < 5.0 else 0.0,
                'movement_diff_from_avg': feature_df.get('movement_magnitude', 0).iloc[0] - 8.0,  # Assuming 8.0 is average
                'batting_average_bip': self.get_hitter_avg_stat(pitch_type, 'batting_average_bip'),
                'whiff_rate': self.get_hitter_avg_stat(pitch_type, 'whiff_rate'),
                'field_out_rate_bip': self.get_hitter_avg_stat(pitch_type, 'field_out_rate_bip'),
                'balls_in_play': self.get_hitter_avg_stat(pitch_type, 'balls_in_play'),
                'total_swings': self.get_hitter_avg_stat(pitch_type, 'total_swings'),
                'total_whiffs': self.get_hitter_avg_stat(pitch_type, 'total_whiffs'),
            }
            
            # Add the additional features to the DataFrame
            for feature_name, value in additional_features.items():
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = value
                    print(f"  + Added {feature_name} = {value}")
            
            print(f"📊 Final feature DataFrame shape: {feature_df.shape}")
            
            # Check feature alignment with model
            if hasattr(self, 'whiff_preprocessor') and self.whiff_preprocessor is not None:
                # Try to get expected features from preprocessor
                if hasattr(self.whiff_preprocessor, 'feature_names_in_'):
                    expected_features = self.whiff_preprocessor.feature_names_in_
                    print(f"📋 Expected features: {len(expected_features)}")
                    print(f"📋 Actual features: {len(feature_df.columns)}")
                    
                    missing_cols = set(expected_features) - set(feature_df.columns)
                    extra_cols = set(feature_df.columns) - set(expected_features)
                    
                    if missing_cols:
                        print(f"⚠️ Missing columns: {missing_cols}")
                        for col in missing_cols:
                            feature_df[col] = 0.0
                            print(f"  + Added {col} = 0.0")
                    
                    if extra_cols:
                        print(f"⚠️ Extra columns: {extra_cols}")
                    
                    feature_df = feature_df[expected_features]
                    print(f"✅ Feature alignment complete")
                else:
                    print("⚠️ Preprocessor doesn't have feature_names_in_ attribute")
                    print("📋 Using all available features for prediction")
            else:
                print("⚠️ No preprocessor available for feature alignment")
            
            print("==== DEBUG: DataFrame dtypes before prediction ====")
            print(feature_df.dtypes)
            print("==== DEBUG: DataFrame head ====")
            print(feature_df.head())
            print("==== DEBUG: DataFrame values dtype ====")
            print(feature_df.values.dtype)
            print("==== DEBUG: First row types ====")
            for i, val in enumerate(feature_df.values[0]):
                print(f"Col {feature_df.columns[i]}: {val} (type: {type(val)})")
            print("==== DEBUG: Checking for non-numeric values in DataFrame ====")
            for col in feature_df.columns:
                for val in feature_df[col]:
                    if not (isinstance(val, (int, float, np.integer, np.floating)) or pd.isna(val)):
                        print(f"⚠️ Column '{col}' has non-numeric value: {val} (type: {type(val)})")
            
            try:
                prediction_proba = self.whiff_model.predict_proba(feature_df)[0]
                whiff_proba = prediction_proba[1] if len(prediction_proba) > 1 else 1 - prediction_proba[0]
                contact_proba = prediction_proba[0] if len(prediction_proba) > 1 else prediction_proba[0]
                confidence = max(prediction_proba)

                # Confidence difference threshold logic
                whiff_threshold = 0.35
                contact_threshold = 0.75
                confidence_diff_threshold = 0.15
                if whiff_proba >= whiff_threshold:
                    is_whiff = True
                    logic = 'whiff_proba >= whiff_threshold'
                elif contact_proba >= contact_threshold:
                    is_whiff = False
                    logic = 'contact_proba >= contact_threshold'
                elif (contact_proba - whiff_proba) >= confidence_diff_threshold:
                    is_whiff = False
                    logic = 'contact_proba - whiff_proba >= confidence_diff_threshold'
                else:
                    is_whiff = True
                    logic = 'default to whiff (uncertain)'

                print(f"📊 Prediction probabilities: {prediction_proba}")
                print(f"📊 Logic used: {logic}")
                print(f"📊 Confidence: {confidence}")

                result = {
                    'is_whiff': is_whiff,
                    'prediction_confidence': confidence,
                    'whiff_probability': whiff_proba,
                    'contact_probability': contact_proba,
                    'details': f"Whiff: {whiff_proba:.3f}, Contact: {contact_proba:.3f}, Logic: {logic}"
                }
                print(f"🎯 Final Result: {result}")
                print("="*50)
                return result
                
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
                print("==== FULL TRACEBACK ====")
                traceback.print_exc()
                print("==== DataFrame dtypes at error ====")
                print(feature_df.dtypes)
                print("==== DataFrame values at error ====")
                print(feature_df.values)
                print("==== DataFrame columns at error ====")
                print(feature_df.columns)
                print("==== DataFrame head at error ====")
                print(feature_df.head())
                print("==== END DEBUG ====")
                # Fallback: return a default prediction
                result = {
                    'is_whiff': True,  # Default to whiff
                    'prediction_confidence': 0.5,
                    'whiff_probability': 0.6,
                    'contact_probability': 0.4,
                    'details': "Fallback prediction due to model error"
                }
                print(f"🎯 Fallback Result: {result}")
                print("="*50)
                return result
        except Exception as e:
            print(f"❌ Error in whiff vs contact prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

def test_at_bat_simulation():
    """Test the at-bat simulation with sample pitches"""
    print("🧪 Testing At-Bat Simulation")
    print("=" * 50)
    
    # Initialize simulation
    sim = AtBatSimulation()
    
    # Test with a few sample pitches
    sample_pitches = [
        {
            'pitch_type': 'FF',
            'plate_x': 0.2,
            'plate_z': 2.5,
            'release_speed': 95,
            'release_spin_rate': 2200,
            'balls': 0,
            'strikes': 0,
            'sz_top': 3.5,
            'sz_bot': 1.5
        },
        {
            'pitch_type': 'SL',
            'plate_x': -0.5,
            'plate_z': 1.8,
            'release_speed': 85,
            'release_spin_rate': 2500,
            'balls': 1,
            'strikes': 0,
            'sz_top': 3.5,
            'sz_bot': 1.5
        }
    ]
    
    for i, pitch in enumerate(sample_pitches):
        print(f"\n--- Pitch {i+1} ---")
        result = sim.simulate_pitch(pitch)
        if result:
            print(f"Result: {result['outcome']}")
            print(f"New count: {result['count']}")
    
    # Print final summary
    summary = sim.get_at_bat_summary()
    print(f"\n📊 At-Bat Summary:")
    print(f"Final count: {summary['current_count']}")
    print(f"Outcome: {summary['at_bat_outcome']}")
    print(f"Total pitches: {summary['total_pitches']}")

if __name__ == "__main__":
    test_at_bat_simulation() 
 
 