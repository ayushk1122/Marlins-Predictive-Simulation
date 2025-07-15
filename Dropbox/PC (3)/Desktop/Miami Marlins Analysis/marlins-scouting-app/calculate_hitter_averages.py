import pandas as pd
import os
import numpy as np

# Features to average for hitter
HITTER_AVG_FEATURES = [
    'bat_speed', 'attack_angle', 'swing_path_tilt', 'swing_length',
    'babip_value', 'launch_speed_angle', 'attack_direction'
]

# Statcast event/description mappings
SWING_DESCRIPTIONS = [
    'swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play'
]
WHIFF_DESCRIPTIONS = [
    'swinging_strike', 'swinging_strike_blocked'
]
BIP_DESCRIPTIONS = ['hit_into_play']
HIT_EVENTS = ['single', 'double', 'triple']
FIELD_OUT_EVENTS = ['field_out']


def calculate_hitter_averages(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    # Only drop rows where pitch_type is null
    df = df.dropna(subset=['pitch_type'])
    # Group by pitch_type and calculate mean for each feature, ignoring nulls for each column
    avg_df = df.groupby('pitch_type')[HITTER_AVG_FEATURES].mean().reset_index()
    # Remove any pitch types where bat_speed average is null
    avg_df = avg_df[~avg_df['bat_speed'].isnull()].copy()
    # Add a row for overall averages (all pitches)
    overall_averages = {feat: df[feat].mean() for feat in HITTER_AVG_FEATURES}
    overall_averages['pitch_type'] = 'ALL'
    avg_df = pd.concat([avg_df, pd.DataFrame([overall_averages])], ignore_index=True)

    # --- Calculate advanced whiff/contact stats for each pitch_type and ALL ---
    def calc_stats(subdf):
        # Swings
        swing_mask = subdf['description'].isin(SWING_DESCRIPTIONS) if 'description' in subdf else pd.Series([False]*len(subdf))
        total_swings = swing_mask.sum()
        # Whiffs
        whiff_mask = subdf['description'].isin(WHIFF_DESCRIPTIONS) if 'description' in subdf else pd.Series([False]*len(subdf))
        total_whiffs = whiff_mask.sum()
        # Balls in play
        bip_mask = subdf['description'].isin(BIP_DESCRIPTIONS) if 'description' in subdf else pd.Series([False]*len(subdf))
        balls_in_play = bip_mask.sum()
        # Hits on balls in play (exclude HR)
        hits_bip = subdf[bip_mask & subdf['events'].isin(HIT_EVENTS)].shape[0] if 'events' in subdf else 0
        # Field outs on balls in play
        field_outs_bip = subdf[bip_mask & subdf['events'].isin(FIELD_OUT_EVENTS)].shape[0] if 'events' in subdf else 0
        whiff_rate = total_whiffs / total_swings if total_swings > 0 else np.nan
        field_out_rate_bip = field_outs_bip / balls_in_play if balls_in_play > 0 else np.nan
        batting_average_bip = hits_bip / balls_in_play if balls_in_play > 0 else np.nan
        return pd.Series({
            'batting_average_bip': batting_average_bip,
            'whiff_rate': whiff_rate,
            'field_out_rate_bip': field_out_rate_bip,
            'balls_in_play': balls_in_play,
            'total_swings': total_swings,
            'total_whiffs': total_whiffs
        })

    # Calculate for each pitch_type
    adv_stats = df.groupby('pitch_type').apply(calc_stats).reset_index()
    # Calculate for ALL
    all_stats = calc_stats(df)
    all_stats['pitch_type'] = 'ALL'
    adv_stats = pd.concat([adv_stats, pd.DataFrame([all_stats])], ignore_index=True)

    # Merge with avg_df
    avg_df = pd.merge(avg_df, adv_stats, on='pitch_type', how='left')

    # Output filename
    if output_csv is None:
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = f"{base.replace('_complete_career_statcast','')}_averages.csv"
    avg_df.to_csv(output_csv, index=False)
    print(f"Saved hitter averages to {output_csv}")
    print(avg_df)
    return avg_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python calculate_hitter_averages.py <hitter_career_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    calculate_hitter_averages(input_csv) 
 
 
 
 