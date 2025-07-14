from flask import Flask, request, jsonify
from flask_cors import CORS
from pybaseball import playerid_lookup, statcast_pitcher
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os
from datetime import datetime
from at_bat_simulation import AtBatSimulation

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global at-bat simulation instances
at_bat_simulations = {}

MARLINS_PITCHERS = [
    "Sandy Alcántara",
    "Lake Bachar",
    "Valente Bellozo",
    "Anthony Bender",
    "Edward Cabrera",
    "Calvin Faucher",
    "Cade Gibson",
    "Ronny Henriquez",
    "Janson Junk",
    "Eury Pérez",
    "Tyler Phillips",
    "Cal Quantrill",
    "Josh Simpson"
]

@app.route('/pitcher_report', methods=['GET'])
def pitcher_report():
    pitcher_name = request.args.get('pitcher')
    season = request.args.get('season', '2025')

    if pitcher_name not in MARLINS_PITCHERS:
        return jsonify({'error': f'Pitcher not found or not active for Marlins: {pitcher_name}'}), 404

    # Dynamically get MLBAM ID
    first, last = pitcher_name.split(" ", 1)
    df = playerid_lookup(last, first)
    if df.empty:
        return jsonify({'error': f'Could not find MLBAM ID for {pitcher_name}'}), 404
    pitcher_id = int(df['key_mlbam'].iloc[0])

    # Retrieve Statcast data for the pitcher for the season
    data = statcast_pitcher(f"{season}-03-01", f"{season}-10-01", pitcher_id)
    unique_pitches = data['pitch_type'].dropna().unique().tolist()
    print("Pitch types:", unique_pitches)
    return jsonify({'pitch_types': unique_pitches})

@app.route('/start_at_bat', methods=['POST'])
def start_at_bat():
    """Start a new at-bat simulation"""
    try:
        data = request.json
        pitcher = data.get('pitcher')
        hitter = data.get('hitter')
        
        if not pitcher or not hitter:
            return jsonify({'error': 'Pitcher and hitter are required'}), 400
        
        # Create unique session ID
        session_id = f"{pitcher}_{hitter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize at-bat simulation
        simulation = AtBatSimulation(hitter_name=hitter.lower().replace(' ', '_'), pitcher_name=pitcher)
        simulation.reset_at_bat()
        
        # Store simulation instance
        at_bat_simulations[session_id] = simulation
        
        return jsonify({
            'session_id': session_id,
            'count': simulation.current_count,
            'message': f'At-bat started: {pitcher} vs {hitter}'
        })
        
    except Exception as e:
        print(f"Error starting at-bat: {str(e)}")
        return jsonify({'error': f'Failed to start at-bat: {str(e)}'}), 500

@app.route('/get_at_bat_status', methods=['GET'])
def get_at_bat_status():
    """Get current at-bat status"""
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in at_bat_simulations:
        return jsonify({'error': 'Invalid session ID'}), 404
    
    simulation = at_bat_simulations[session_id]
    
    return jsonify({
        'count': simulation.current_count,
        'outcome': simulation.at_bat_outcome,
        'pitch_sequence': simulation.pitch_sequence,
        'is_complete': simulation.at_bat_outcome is not None
    })

@app.route('/simulate_pitch', methods=['POST'])
def simulate_pitch():
    try:
        data = request.json
        session_id = data.get('session_id')
        pitcher = data.get('pitcher')
        hitter = data.get('hitter')
        pitch_type = data.get('pitch_type')
        plate_x = data.get('plate_x')
        plate_z = data.get('plate_z')
        zone = data.get('zone')
        balls = data.get('balls', 0)
        strikes = data.get('strikes', 0)
        handedness = data.get('handedness', 'R')
        
        print(f"Received pitch simulation request:")
        print(f"  Session ID: {session_id}")
        print(f"  Pitcher: {pitcher}")
        print(f"  Hitter: {hitter}")
        print(f"  Pitch Type: {pitch_type}")
        print(f"  Plate X: {plate_x}")
        print(f"  Plate Z: {plate_z}")
        print(f"  Zone: {zone}")
        print(f"  Balls: {balls}, Strikes: {strikes}")
        print(f"  Handedness: {handedness}")
        
        # Check if we have an active at-bat simulation
        if session_id and session_id in at_bat_simulations:
            simulation = at_bat_simulations[session_id]
            
            # Create pitch data for simulation
            pitch_data = {
                'pitch_type': pitch_type,
                'plate_x': plate_x,
                'plate_z': plate_z,
                'zone': zone,
                'balls': balls,
                'strikes': strikes,
                'handedness': handedness,
                'sz_top': 3.5,
                'sz_bot': 1.5
            }
            
            # Simulate the pitch using the at-bat simulation
            result = simulation.simulate_pitch(pitch_data)
            
            if result:
                return jsonify({
                    'outcome': result['outcome'],
                    'confidence': result['confidence'],
                    'count': simulation.current_count,
                    'at_bat_outcome': simulation.at_bat_outcome,
                    'pitch_sequence': simulation.pitch_sequence,
                    'is_complete': simulation.at_bat_outcome is not None,
                    'details': result.get('details', '')
                })
            else:
                return jsonify({'error': 'Simulation failed'}), 500
        else:
            # Fallback to simple mock response for backward compatibility
            import random
            
            # Determine if it's in the strike zone (using inner boundaries)
            in_strike_zone = (-0.7 <= plate_x <= 0.7) and (1.5 <= plate_z <= 3.5)
            
            # Simple logic for mock outcomes
            if in_strike_zone:
                # More likely to swing at strikes
                if random.random() < 0.7:  # 70% chance to swing
                    outcomes = ['Swinging Strike', 'Hit Safely', 'Field Out']
                    outcome = random.choice(outcomes)
                    confidence = random.uniform(0.6, 0.9)
                else:
                    outcome = 'Called Strike'
                    confidence = random.uniform(0.7, 0.95)
            else:
                # Less likely to swing at balls
                if random.random() < 0.3:  # 30% chance to swing
                    outcomes = ['Swinging Strike', 'Hit Safely', 'Field Out']
                    outcome = random.choice(outcomes)
                    confidence = random.uniform(0.5, 0.8)
                else:
                    outcome = 'Ball'
                    confidence = random.uniform(0.8, 0.95)
            
            # Create pitch data for saving
            pitch_data = {
                'pitcher': pitcher,
                'hitter': hitter,
                'pitch_type': pitch_type,
                'plate_x': plate_x,
                'plate_z': plate_z,
                'zone': zone,
                'balls': balls,
                'strikes': strikes,
                'handedness': handedness,
                'in_strike_zone': in_strike_zone,
                'outcome': outcome,
                'confidence': confidence,
                'swing_probability': random.uniform(0.3, 0.7),
                'timestamp': datetime.now().isoformat(),
                'details': f"Pitch at ({plate_x:.2f}, {plate_z:.2f}) - {pitch_type} pitch - Count: {balls}-{strikes}"
            }
            
            # Save pitch data to JSON file
            save_pitch_data(pitch_data)
            
            return jsonify({
                'outcome': outcome,
                'confidence': confidence,
                'swing_probability': pitch_data['swing_probability'],
                'details': pitch_data['details']
            })
        
    except Exception as e:
        print(f"Error in simulate_pitch: {str(e)}")
        return jsonify({'error': f'Simulation failed: {str(e)}'}), 500

def save_pitch_data(pitch_data):
    """Save pitch simulation data to JSON file in Pitches folder"""
    try:
        # Create safe filename
        pitcher_name = pitch_data['pitcher'].replace(' ', '_').replace('á', 'a')
        hitter_name = pitch_data['hitter'].replace(' ', '_').replace('.', '')
        pitch_type = pitch_data['pitch_type']
        zone = pitch_data['zone'] if pitch_data['zone'] else 'unknown'
        
        # Create filename: pitchername_hittername_pitchtype_zone.json
        filename = f"{pitcher_name}_{hitter_name}_{pitch_type}_zone{zone}.json"
        
        # Ensure Pitches folder exists
        pitches_dir = "Pitches"
        if not os.path.exists(pitches_dir):
            os.makedirs(pitches_dir)
        
        # Save to JSON file
        filepath = os.path.join(pitches_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(pitch_data, f, indent=2)
        
        print(f"Saved pitch data to: {filepath}")
        
    except Exception as e:
        print(f"Error saving pitch data: {str(e)}")

if __name__ == '__main__':
    app.run(port=5001, debug=True) 