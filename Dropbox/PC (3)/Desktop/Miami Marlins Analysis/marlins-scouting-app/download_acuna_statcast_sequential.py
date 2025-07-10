from pybaseball import statcast_batter
import pandas as pd

# Ronald Acuña Jr. MLBAM ID
mlbam_id = 660670

# Download all pitch data for his career
print('Downloading Statcast data for Ronald Acuña Jr...')
data = statcast_batter('2018-01-01', '2025-12-31', player_id=mlbam_id)

# Columns to keep (including sequential modeling keys)
columns = [
    'game_date', 'at_bat_number', 'game_pk',
    'pitch_type', 'pitch_name', 'release_speed', 'release_spin_rate', 'spin_axis', 'release_extension',
    'release_pos_x', 'release_pos_y', 'release_pos_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone', 'sz_top', 'sz_bot',
    'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in',
    'p_throws', 'if_fielding_alignment', 'of_fielding_alignment', 'balls', 'strikes',
    'description', 'events', 'launch_speed', 'launch_angle', 'hit_distance_sc', 'bb_type',
    'estimated_ba_using_speedangle', 'estimated_slg_using_speedangle', 'estimated_woba_using_speedangle',
    'swing_length', 'bat_speed', 'attack_angle', 'swing_path_tilt', 'arm_angle'
]
columns = [col for col in columns if col in data.columns]
data = data[columns]

# Save to CSV
out_csv = 'ronald_acuna_jr_statcast_pitches_sequential.csv'
data.to_csv(out_csv, index=False)
print(f'Saved to {out_csv} ({len(data)} rows)') 