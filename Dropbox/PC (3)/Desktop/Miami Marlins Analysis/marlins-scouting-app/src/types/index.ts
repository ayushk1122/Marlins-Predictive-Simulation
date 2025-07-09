export interface HitterStats {
    xBA: number;
    whiffPercent: number;
    avgExitVelo: number;
}

export interface HitterReport {
    name: string;
    weaknesses: string[];
    stats: HitterStats;
    recommendations: string[];
}

export interface PitcherReport {
    outings: any[];
    pitch_usage: any[];
    tto_usage: any[];
    player_id: number;
    error?: string;
}

export interface ScoutingReport {
    pitcher: string;
    team: string;
    lineup: string[];
    summary: string;
    hitters: HitterReport[];
    pitcherReport?: PitcherReport;
} 