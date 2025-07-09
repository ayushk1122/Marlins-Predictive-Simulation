from flask import Flask, request, jsonify
from flask_cors import CORS
from pybaseball import playerid_lookup, statcast_pitcher
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

if __name__ == '__main__':
    app.run(port=5001, debug=True) 