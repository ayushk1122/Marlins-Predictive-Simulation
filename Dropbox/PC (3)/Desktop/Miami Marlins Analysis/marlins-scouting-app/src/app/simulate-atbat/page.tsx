"use client";
import React, { useState } from "react";
import PitcherSelector from "../../components/PitcherSelector";
import Image from "next/image";
import { useSearchParams } from "next/navigation";
import axios from "axios";

const DEMO_TEAMS = ["Braves", "Mets", "Phillies"];
const DEMO_LINEUPS: Record<string, string[]> = {
    Braves: ["Acuna Jr.", "Albies", "Riley", "Olson", "Ozuna", "Rosario", "Arcia", "Murphy", "Harris II"],
    Mets: ["Nimmo", "Lindor", "Alonso", "McNeil", "Marte", "Baty", "Vientos", "Alvarez", "Stewart"],
    Phillies: ["Schwarber", "Turner", "Harper", "Bohm", "Castellanos", "Stott", "Realmuto", "Marsh", "Sosa"]
};
const DEMO_PITCH_TYPES: Record<string, string[]> = {
    "Sandy Alcántara": ["FF", "SI", "CH", "SL"],
    "Lake Bachar": ["FF", "CU", "CH"],
    "Edward Cabrera": ["FF", "SI", "CH", "SL"],
};
const DEMO_HANDEDNESS: Record<string, "R" | "L" | "S"> = {
    "Acuna Jr.": "R",
    "Albies": "S",
    "Riley": "R",
    "Olson": "L",
    "Ozuna": "R",
    "Rosario": "L",
    "Arcia": "R",
    "Murphy": "R",
    "Harris II": "L",
    "Nimmo": "L",
    "Lindor": "S",
    "Alonso": "R",
    "McNeil": "L",
    "Marte": "R",
    "Baty": "L",
    "Vientos": "R",
    "Alvarez": "R",
    "Stewart": "L",
    "Schwarber": "L",
    "Turner": "R",
    "Harper": "L",
    "Bohm": "R",
    "Castellanos": "R",
    "Stott": "L",
    "Realmuto": "R",
    "Marsh": "L",
    "Sosa": "R"
};

const HITTER_IMAGES = {
    R: "/hitter_right.png", // Save the righty silhouette as public/hitter_right.png
    L: "/hitter_left.png",  // Save the lefty silhouette as public/hitter_left.png
};

// Pitch type color palette (Savant-inspired)
const PITCH_TYPE_COLORS: string[] = [
    "#1f77b4", // blue
    "#ff7f0e", // orange
    "#2ca02c", // green
    "#d62728", // red
    "#9467bd", // purple
    "#8c564b", // brown
    "#e377c2", // pink
    "#7f7f7f", // gray
    "#bcbd22", // olive
    "#17becf", // cyan
];

function getPitchColor(pitchType: string, pitchTypes: string[]): string {
    const idx = pitchTypes.indexOf(pitchType);
    return PITCH_TYPE_COLORS[idx % PITCH_TYPE_COLORS.length] || "#222";
}

function StrikeZoneDisplay({ marker, onSelect, markerColor }: {
    marker: { x: number; y: number } | null;
    onSelect: (x: number, y: number) => void;
    markerColor?: string;
}) {
    const scale = 100; // 1 foot = 100 pixels

    // Real-world dimensions (in feet)
    const plateWidthFt = 1.417;
    const strikeZoneTopFt = 3.25;
    const strikeZoneBottomFt = 1.64;
    const fullZoneHeightFt = 6.25; // From ground to top of head
    const extendedZoneWidthFt = 2.417; // Inner LH to RH batter's box

    // Scaled pixel dimensions
    const plateWidth = plateWidthFt * scale;
    const strikeZoneHeight = (strikeZoneTopFt - strikeZoneBottomFt) * scale;
    const strikeZoneBottomOffset = strikeZoneBottomFt * scale;
    const fullZoneHeight = fullZoneHeightFt * scale;
    const fullZoneWidth = extendedZoneWidthFt * scale;

    // SVG layout
    const svgWidth = 700;
    const svgHeight = 800;
    const xCenter = svgWidth / 2;
    const plateY = 700;
    const strikeZoneTopY = plateY - strikeZoneBottomOffset - strikeZoneHeight;
    const extendedZoneTopY = plateY - fullZoneHeight;

    return (
        <svg width={svgWidth} height={svgHeight} style={{ background: "white" }}
            onClick={e => {
                const rect = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                // Allow clicks anywhere in the extended zone
                if (
                    x >= xCenter - fullZoneWidth / 2 &&
                    x <= xCenter + fullZoneWidth / 2 &&
                    y >= extendedZoneTopY &&
                    y <= extendedZoneTopY + fullZoneHeight
                ) {
                    onSelect(x, y);
                }
            }}
        >
            {/* Extended Strike Zone (outer chalk lines) */}
            <rect
                x={xCenter - fullZoneWidth / 2}
                y={extendedZoneTopY}
                width={fullZoneWidth}
                height={fullZoneHeight}
                stroke="#cccccc"
                strokeWidth={2}
                fill="none"
            />

            {/* Main Strike Zone */}
            <rect
                x={xCenter - plateWidth / 2}
                y={strikeZoneTopY}
                width={plateWidth}
                height={strikeZoneHeight}
                stroke="black"
                strokeWidth={2}
                fill="none"
            />

            {/* Home Plate */}
            <polygon
                points={[
                    `${xCenter - plateWidth / 2},${plateY}`,
                    `${xCenter + plateWidth / 2},${plateY}`,
                    `${xCenter + plateWidth / 2},${plateY + 20}`,
                    `${xCenter},${plateY + 40}`,
                    `${xCenter - plateWidth / 2},${plateY + 20}`,
                ].join(" ")}
                stroke="black"
                strokeWidth={2}
                fill="none"
            />
            {/* Marker (baseball) */}
            {marker && (
                <circle cx={marker.x} cy={marker.y} r={12} fill="#fff" stroke={markerColor || "#d97706"} strokeWidth={4} />
            )}
        </svg>
    );
}

export default function SimulateAtBatPage() {
    const searchParams = useSearchParams();
    const [pitcher, setPitcher] = useState<string | null>(searchParams.get("pitcher") || null);
    const [team, setTeam] = useState<string | null>(searchParams.get("team") || null);
    const [hitter, setHitter] = useState<string | null>(searchParams.get("hitter") || null);
    const [pitchType, setPitchType] = useState<string | null>(null);
    const [marker, setMarker] = useState<{ x: number; y: number } | null>(null);
    const [pitchTypes, setPitchTypes] = useState<string[]>([]);
    const [loadingPitches, setLoadingPitches] = useState(false);
    const [pitchError, setPitchError] = useState<string | null>(null);

    // Fetch pitch types for selected pitcher
    React.useEffect(() => {
        if (!pitcher) {
            setPitchTypes([]);
            setPitchType(null);
            return;
        }
        setLoadingPitches(true);
        setPitchError(null);
        axios.get("http://localhost:5001/pitcher_report", {
            params: { pitcher, season: "2025" }
        })
            .then(res => {
                const types = (res.data.pitch_types || []) as string[];
                setPitchTypes(types);
                setPitchType(null);
            })
            .catch(() => {
                setPitchTypes([]);
                setPitchType(null);
                setPitchError("Could not fetch pitch types for this pitcher.");
            })
            .finally(() => setLoadingPitches(false));
    }, [pitcher]);

    let handedness: "R" | "L" = "R";
    if (hitter) {
        const h = DEMO_HANDEDNESS[hitter];
        handedness = h === "L" ? "L" : "R";
    }

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center py-8">
            <div className="w-full max-w-3xl bg-white rounded-lg shadow p-8 flex flex-col gap-6 text-gray-900">
                <h1 className="text-3xl font-bold mb-2 text-center">At-Bat Simulator</h1>
                <div className="flex flex-wrap gap-4 justify-center">
                    <div className="min-w-[200px]">
                        <PitcherSelector value={pitcher} onChange={p => { setPitcher(p); }} />
                    </div>
                    <div className="min-w-[200px]">
                        <label className="block font-semibold mb-2">Opponent Team</label>
                        <select className="w-full border rounded px-3 py-2" value={team || ""} onChange={e => { setTeam(e.target.value); setHitter(null); }}>
                            <option value="" disabled>Choose a team</option>
                            {DEMO_TEAMS.map(t => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </div>
                    <div className="min-w-[200px]">
                        <label className="block font-semibold mb-2">Hitter</label>
                        <select className="w-full border rounded px-3 py-2" value={hitter || ""} onChange={e => setHitter(e.target.value)} disabled={!team}>
                            <option value="" disabled>Choose a hitter</option>
                            {team && DEMO_LINEUPS[team].map(h => <option key={h} value={h}>{h}</option>)}
                        </select>
                    </div>
                    <div className="min-w-[200px]">
                        <label className="block font-semibold mb-2">Pitch Type</label>
                        <select className="w-full border rounded px-3 py-2" value={pitchType || ""} onChange={e => setPitchType(e.target.value)} disabled={!pitcher || loadingPitches || pitchTypes.length === 0} style={{ color: pitchType ? getPitchColor(pitchType, pitchTypes) : undefined }}>
                            <option value="" disabled>{loadingPitches ? "Loading..." : "Choose pitch"}</option>
                            {pitchTypes.map((pt, i) => (
                                <option key={pt} value={pt} style={{ color: getPitchColor(pt, pitchTypes) }}>{pt}</option>
                            ))}
                        </select>
                        {pitchError && <div className="text-red-600 text-xs mt-1">{pitchError}</div>}
                    </div>
                </div>
                <div className="flex flex-col items-center mt-6">
                    <div className="mb-2 font-semibold text-lg">Click in the box to place your pitch</div>
                    <StrikeZoneDisplay marker={marker} onSelect={(x, y) => setMarker({ x, y })} markerColor={pitchType ? getPitchColor(pitchType, pitchTypes) : "#d97706"} />
                </div>
                <button className="bg-green-600 text-white px-4 py-2 rounded font-semibold w-full mt-6" disabled={!pitchType || !marker}>
                    Simulate Pitch (Demo)
                </button>
            </div>
        </div>
    );
} 