import { NextRequest, NextResponse } from "next/server";
import axios from "axios";

export async function POST(req: NextRequest) {
    const { pitcher, team, lineup } = await req.json();
    const season = "2025"; // Force season to 2025

    // Call the Python microservice for pitcher report
    let pitcherReport = null;
    try {
        const res = await axios.get("http://localhost:5001/pitcher_report", {
            params: {
                pitcher,
                opponent: team,
                season,
            },
        });
        pitcherReport = res.data;
    } catch (err) {
        pitcherReport = { error: "Could not fetch advanced pitcher data" };
    }

    // Placeholder: Replace with real analysis for hitters
    const report = {
        pitcher,
        team,
        lineup,
        summary: `Scouting report for ${pitcher} vs. ${team}`,
        pitcherReport, // <-- advanced pitcher data
        hitters: lineup.map((hitter: string) => ({
            name: hitter,
            weaknesses: ["High fastballs", "Sliders away"],
            stats: {
                xBA: 0.210,
                whiffPercent: 32,
                avgExitVelo: 87.5
            },
            recommendations: [
                "Throw more changeups",
                "Work up in the zone"
            ]
        }))
    };
    return NextResponse.json(report);
} 