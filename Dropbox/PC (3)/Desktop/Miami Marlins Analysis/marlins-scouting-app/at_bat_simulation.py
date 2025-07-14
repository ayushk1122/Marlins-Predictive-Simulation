import json
import os
import pandas as pd
import numpy as np
import joblib
import pickle
from test_pitch_json_classifier import load_career_data, calculate_hitter_features, load_average_metrics, prepare_pitch_features
import warnings
warnings.filterwarnings('ignore')

class AtBatSimulation:
    def __init__(self, hitter_name='acuna', pitcher_name='unknown'):
        """Initialize at-bat simulation with hitter and pitcher"""
        self.hitter_name = hitter_name
        self.pitcher_name = pitcher_name
        self.current_count = {'balls': 0, 'strikes': 0}
        self.at_bat_outcome = None
        self.pitch_sequence = []
        
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
    
    def reset_at_bat(self):
        """Reset the at-bat to 0-0 count"""
        self.current_count = {'balls': 0, 'strikes': 0}
        self.at_bat_outcome = None
        self.pitch_sequence = []
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
        
        # Prepare features using the same approach as the test script
        print("🔧 Generating features for swing vs no swing model...")
        
        # Use the same feature preparation as the test script
        features = prepare_pitch_features(pitch_data, self.averages)
        
        # Update features with actual career data
        if self.hitter_features:
            for feature_name, value in self.hitter_features.items():
                if feature_name in features:
                    features[feature_name] = value
        
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
                '0-2': 0.50,
                '1-0': 0.95,
                '1-1': 0.95,
                '1-2': 0.55,
                '2-0': 0.95,
                '2-1': 0.90,
                '2-2': 0.75,
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
                    X_whiff = self.prepare_whiff_features(pitch_data)
                    whiff_proba = self.whiff_model.predict_proba(X_whiff)[0][1]
                    contact_proba = self.whiff_model.predict_proba(X_whiff)[0][0]
                    whiff_pred = self.whiff_model.predict(X_whiff)[0]
                    details += f' | Whiff prob: {whiff_proba:.2f}, Contact prob: {contact_proba:.2f}'
                    if whiff_pred == 1:
                        outcome = 'whiff'
                        self.current_count['strikes'] += 1
                        details += ' | SWING & MISS (strike)'
                    else:
                        outcome = 'contact'
                        details += ' | CONTACT (at-bat ends)'
                        at_bat_outcome = 'contact'
                else:
                    outcome = 'swing'
                    self.current_count['strikes'] += 1
                    details += ' | SWING (strike assumed for demo)'
            else:
                # Determine if it's a ball or strike
                ball_or_strike = self.determine_ball_or_strike(pitch_data)
                outcome = ball_or_strike
                self.update_count(ball_or_strike)
                details += f' | NO SWING - {ball_or_strike.upper()}'

        # Check for at-bat ending conditions
        at_bat_outcome = None if 'at_bat_outcome' not in locals() else at_bat_outcome
        if self.current_count['strikes'] >= 3:
            at_bat_outcome = 'strikeout'
        elif self.current_count['balls'] >= 4:
            at_bat_outcome = 'walk'

        result = {
            'outcome': outcome,
            'confidence': prediction_confidence,
            'details': details,
            'count': dict(self.current_count),
            'at_bat_outcome': at_bat_outcome,
        }
        self.pitch_sequence.append(result)

        # If at-bat ends, append summary and do NOT reset at-bat here
        if at_bat_outcome:
            summary_result = {
                'outcome': at_bat_outcome,
                'confidence': 1.0,
                'details': f'At-bat ended with {at_bat_outcome.upper()}',
                'count': dict(self.current_count),
                'at_bat_outcome': at_bat_outcome,
            }
            self.pitch_sequence.append(summary_result)
            # Do NOT reset here; let frontend display the outcome
            # self.reset_at_bat()

        return result
    
    def get_at_bat_summary(self):
        """Get summary of the current at-bat"""
        return {
            'current_count': self.current_count,
            'at_bat_outcome': self.at_bat_outcome,
            'total_pitches': len(self.pitch_sequence),
            'pitch_sequence': self.pitch_sequence
        }

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
 