"use client";
import React, { useState } from "react";
import PitcherSelector from "../../components/PitcherSelector";
import CountDisplay from "../../components/CountDisplay";
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

    // Outer strike zone (Statcast): plate_x from -1.5 to 1.5, plate_z from 1.0 to 4.0, centered at the same vertical center as the inner strike zone
    // Outer Strike Zone centered around inner zone
    const outerZoneWidthPx = scale * 3.0;
    const outerZoneHeightPx = scale * 3.0;
    const innerZoneCenterY = strikeZoneTopY + strikeZoneHeight / 2;

    const outerZoneLeftX = xCenter - outerZoneWidthPx / 2;
    const outerZoneTopY = innerZoneCenterY - outerZoneHeightPx / 2;

    const outerZoneWidth = outerZoneWidthPx;
    const outerZoneHeight = outerZoneHeightPx;

    // Convert pixel coordinates to Statcast coordinates
    const convertToStatcastCoords = (pixelX: number, pixelY: number) => {
        // Convert from pixel coordinates to Statcast coordinates
        // Inner strike zone: plate_x ranges from -0.7 to 0.7, plate_z from 1.5 to 3.5
        const plateX = ((pixelX - xCenter) / (fullZoneWidth / 2)) * 0.7;
        const plateZ = ((plateY - pixelY - strikeZoneBottomOffset) / strikeZoneHeight) * (3.5 - 1.5) + 1.5;

        return { plate_x: plateX, plate_z: plateZ };
    };

    return (
        <svg width={svgWidth} height={svgHeight} style={{ background: "white" }}
            onClick={e => {
                const rect = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                // Allow clicks anywhere in the extended zone
                if (
                    x >= outerZoneLeftX &&
                    x <= outerZoneLeftX + outerZoneWidth &&
                    y >= outerZoneTopY &&
                    y <= outerZoneTopY + outerZoneHeight
                ) {
                    onSelect(x, y);
                }
            }}
        >
            {/* Outer Strike Zone (Statcast) */}
            <rect
                x={outerZoneLeftX}
                y={outerZoneTopY}
                width={outerZoneWidth}
                height={outerZoneHeight}
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
    const [simulating, setSimulating] = useState(false);
    const [simulationResult, setSimulationResult] = useState<any>(null);

    // At-bat simulation state
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [atBatActive, setAtBatActive] = useState(false);
    const [currentCount, setCurrentCount] = useState({ balls: 0, strikes: 0 });
    const [atBatOutcome, setAtBatOutcome] = useState<string | null>(null);
    const [pitchSequence, setPitchSequence] = useState<any[]>([]);
    const [startingAtBat, setStartingAtBat] = useState(false);

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

    // When atBatOutcome is set, reset at-bat controls and count, but keep results visible
    React.useEffect(() => {
        if (atBatOutcome) {
            setAtBatActive(false);
            setCurrentCount({ balls: 0, strikes: 0 });
        }
    }, [atBatOutcome]);

    // Convert pixel coordinates to Statcast coordinates
    const convertToStatcastCoords = (pixelX: number, pixelY: number) => {
        const scale = 100;
        const plateWidthFt = 1.417;
        const strikeZoneTopFt = 3.25;
        const strikeZoneBottomFt = 1.64;
        const fullZoneHeightFt = 6.25;
        const extendedZoneWidthFt = 2.417;

        const plateWidth = plateWidthFt * scale;
        const strikeZoneHeight = (strikeZoneTopFt - strikeZoneBottomFt) * scale;
        const strikeZoneBottomOffset = strikeZoneBottomFt * scale;
        const fullZoneHeight = fullZoneHeightFt * scale;
        const fullZoneWidth = extendedZoneWidthFt * scale;

        const svgWidth = 700;
        const svgHeight = 800;
        const xCenter = svgWidth / 2;
        const plateY = 700;
        const strikeZoneTopY = plateY - strikeZoneBottomOffset - strikeZoneHeight;

        // Find pixel boundaries of the inner strike zone
        const left_px = xCenter - plateWidth / 2;
        const right_px = xCenter + plateWidth / 2;
        // Map left edge to -0.71, right edge to 0.71
        const plateX = ((pixelX - left_px) / (right_px - left_px)) * 1.42 - 0.71;
        // plate_z scaling remains the same
        const plateZ = ((plateY - pixelY - strikeZoneBottomOffset) / strikeZoneHeight) * (3.5 - 1.5) + 1.5;
        return { plate_x: plateX, plate_z: plateZ };
    };

    // Helper: Point-in-polygon (ray-casting algorithm)
    function pointInPolygon(x: number, y: number, polygon: [number, number][]) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];
            const intersect = ((yi > y) !== (yj > y)) &&
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    // Zone polygons for 11,12,13,14
    const zonePolygons: Record<number, [number, number][]> = {
        13: [
            [-1.5, 0.5], [-1.5, 2.5], [-0.71, 2.5], [0, 1.5], [0, 0.5], [-0.71, 1.5]
        ],
        14: [
            [0, 1.5], [0, 0.5], [0.71, 1.5], [0.71, 2.5], [1.5, 2.5], [1.5, 0.5]
        ],
        11: [
            [-1.5, 2.5], [-0.71, 2.5], [-0.71, 3.5], [0, 3.5], [0, 4.5], [-1.5, 4.5]
        ],
        12: [
            [0, 4.5], [0, 3.5], [0.71, 3.5], [0.71, 2.5], [1.5, 2.5], [1.5, 4.5]
        ]
    };

    // Get MLB Statcast zone
    function getZone(plate_x: number, plate_z: number): number | null {
        // Inner zone boundaries
        const xMin = -0.71, xMax = 0.71;
        const zMin = 1.5, zMax = 3.5;
        // 3x3 grid
        if (plate_x >= xMin && plate_x <= xMax && plate_z >= zMin && plate_z <= zMax) {
            const cellWidth = (xMax - xMin) / 3;
            const cellHeight = (zMax - zMin) / 3;
            const col = Math.floor((plate_x - xMin) / cellWidth); // 0,1,2
            const row = 2 - Math.floor((plate_z - zMin) / cellHeight); // flip: 0=top, 2=bottom
            const zone = row * 3 + col + 1;
            return zone;
        }
        // Check polygons for 11,12,13,14
        for (const zoneNum of [11, 12, 13, 14]) {
            if (pointInPolygon(plate_x, plate_z, zonePolygons[zoneNum])) {
                return zoneNum;
            }
        }
        return null;
    }

    // Start at-bat simulation
    const handleStartAtBat = async () => {
        if (!pitcher || !hitter) return;

        setStartingAtBat(true);
        try {
            const response = await axios.post("http://localhost:5001/start_at_bat", {
                pitcher,
                hitter
            });

            setSessionId(response.data.session_id);
            setCurrentCount(response.data.count);
            setAtBatActive(true);
            setAtBatOutcome(null);
            setPitchSequence([]);
            setSimulationResult(null);
            setMarker(null);

            console.log("At-bat started:", response.data.message);
        } catch (error) {
            console.error("Failed to start at-bat:", error);
            setPitchError("Failed to start at-bat simulation.");
        } finally {
            setStartingAtBat(false);
        }
    };

    // Handle pitch simulation
    const handleSimulatePitch = async () => {
        if (!pitcher || !hitter || !pitchType || !marker) return;

        setSimulating(true);
        setSimulationResult(null);

        try {
            const coords = convertToStatcastCoords(marker.x, marker.y);
            const zone = getZone(coords.plate_x, coords.plate_z);

            const requestData = {
                session_id: sessionId,
                pitcher,
                hitter,
                pitch_type: pitchType,
                plate_x: coords.plate_x,
                plate_z: coords.plate_z,
                zone,
                balls: currentCount.balls,
                strikes: currentCount.strikes,
                handedness: DEMO_HANDEDNESS[hitter] || "R"
            };

            const response = await axios.post("http://localhost:5001/simulate_pitch", requestData);

            setSimulationResult(response.data);

            // Update at-bat state if we have session data
            if (response.data.count) {
                setCurrentCount(response.data.count);
            }
            if (response.data.at_bat_outcome) {
                setAtBatOutcome(response.data.at_bat_outcome);
                setAtBatActive(false);
            }
            if (response.data.pitch_sequence) {
                setPitchSequence(response.data.pitch_sequence);
            }

            // Clear marker for next pitch
            setMarker(null);

        } catch (error) {
            console.error("Pitch simulation failed:", error);
            setPitchError("Failed to simulate pitch. Please try again.");
        } finally {
            setSimulating(false);
        }
    };

    // Reset at-bat
    const handleResetAtBat = () => {
        setSessionId(null);
        setAtBatActive(false);
        setCurrentCount({ balls: 0, strikes: 0 });
        setAtBatOutcome(null);
        setPitchSequence([]);
        setSimulationResult(null);
        setMarker(null);
    };

    let handedness: "R" | "L" = "R";
    if (hitter) {
        const h = DEMO_HANDEDNESS[hitter];
        handedness = h === "L" ? "L" : "R";
    }

    return (
        <div className="min-h-screen bg-gray-50 flex flex-col items-center py-8">
            <div className="w-full flex flex-col items-center gap-6">
                {/* Controls Card */}
                <div className="w-full max-w-md mx-auto bg-white rounded-lg shadow p-8 flex flex-col gap-4 items-center">
                    <h1 className="text-3xl font-bold mb-2 text-center text-gray-900">At-Bat Simulator</h1>
                    <div className="flex flex-col gap-4 w-full max-w-sm items-center">
                        <PitcherSelector value={pitcher} onChange={p => { setPitcher(p); }} />
                        <div className="w-full">
                            <label className="block font-semibold mb-2 text-gray-900">Opponent Team</label>
                            <select className="w-full border rounded px-3 py-2 text-gray-900" value={team || ""} onChange={e => { setTeam(e.target.value); setHitter(null); }}>
                                <option value="" disabled>Choose a team</option>
                                {DEMO_TEAMS.map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                        </div>
                        <div className="w-full">
                            <label className="block font-semibold mb-2 text-gray-900">Hitter</label>
                            <select className="w-full border rounded px-3 py-2 text-gray-900" value={hitter || ""} onChange={e => setHitter(e.target.value)} disabled={!team}>
                                <option value="" disabled>Choose a hitter</option>
                                {team && DEMO_LINEUPS[team].map(h => <option key={h} value={h}>{h}</option>)}
                            </select>
                        </div>
                        <div className="w-full">
                            <label className="block font-semibold mb-2 text-gray-900">Pitch Type</label>
                            <select className="w-full border rounded px-3 py-2 text-gray-900" value={pitchType || ""} onChange={e => setPitchType(e.target.value)} disabled={!pitcher || loadingPitches || pitchTypes.length === 0 || !atBatActive} style={{ color: pitchType ? getPitchColor(pitchType, pitchTypes) : undefined }}>
                                <option value="" disabled>{loadingPitches ? "Loading..." : !atBatActive ? "Start at-bat first" : "Choose pitch"}</option>
                                {pitchTypes.map((pt, i) => (
                                    <option key={pt} value={pt} style={{ color: getPitchColor(pt, pitchTypes) }}>{pt}</option>
                                ))}
                            </select>
                            {pitchError && <div className="text-red-600 text-xs mt-1">{pitchError}</div>}
                        </div>
                        <div className="w-full flex justify-center">
                            {atBatActive ? (
                                <button
                                    className="bg-red-600 text-white px-6 py-3 rounded-lg font-semibold text-lg hover:bg-red-700 w-full"
                                    onClick={handleResetAtBat}
                                >
                                    Reset At-Bat
                                </button>
                            ) : (
                                <button
                                    className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold text-lg hover:bg-green-700 disabled:bg-gray-400 w-full"
                                    disabled={!pitcher || !hitter || startingAtBat}
                                    onClick={handleStartAtBat}
                                >
                                    {startingAtBat ? "Starting..." : "Start At-Bat"}
                                </button>
                            )}
                        </div>
                    </div>
                    {/* Only show count if at-bat is active and not over */}
                    {atBatActive && !atBatOutcome && (
                        <div className="flex justify-center mb-2 w-full mt-4">
                            <CountDisplay
                                balls={currentCount.balls}
                                strikes={currentCount.strikes}
                                isComplete={atBatOutcome !== null}
                                outcome={atBatOutcome || undefined}
                            />
                        </div>
                    )}
                </div>
                {/* Strike Zone */}
                <div className="flex flex-col items-center w-full mt-6">
                    <div className="mb-2 font-semibold text-lg text-gray-900">Click in the box to place your pitch</div>
                    <StrikeZoneDisplay marker={marker} onSelect={(x, y) => setMarker({ x, y })} markerColor={pitchType ? getPitchColor(pitchType, pitchTypes) : "#d97706"} />
                    {marker && (() => {
                        const coords = convertToStatcastCoords(marker.x, marker.y);
                        const zone = getZone(coords.plate_x, coords.plate_z);
                        return (
                            <div className="mt-2 text-base text-blue-700 font-bold">
                                Zone: {zone !== null ? zone : "(none)"} (plate_x: {coords.plate_x.toFixed(2)}, plate_z: {coords.plate_z.toFixed(2)})
                            </div>
                        );
                    })()}
                    <button
                        className="bg-blue-600 text-white px-4 py-2 rounded font-semibold w-full mt-6 disabled:bg-gray-400 max-w-md"
                        disabled={!pitchType || !marker || simulating || !atBatActive || !!atBatOutcome}
                        onClick={handleSimulatePitch}
                    >
                        {simulating ? "Simulating..." : "Simulate Pitch"}
                    </button>
                </div>
                {/* Results - Unified Card */}
                {(simulationResult || pitchSequence.length > 0 || atBatOutcome) && (
                    <div className="p-6 bg-blue-50 rounded-lg border w-full max-w-md mx-auto mt-6 flex flex-col gap-4 items-center text-gray-900">
                        <h3 className="font-semibold text-lg mb-2 text-gray-900">Pitch Results & Sequence</h3>
                        {simulationResult && (
                            <div className="w-full">
                                <div>
                                    <span className="font-medium">Last Pitch Outcome:</span> {simulationResult.outcome}
                                </div>
                                <div>
                                    <span className="font-medium">Confidence:</span> {(simulationResult.confidence * 100).toFixed(1)}%
                                </div>
                                {simulationResult.details && (
                                    <div>
                                        <span className="font-medium">Details:</span> {simulationResult.details}
                                    </div>
                                )}
                            </div>
                        )}
                        {pitchSequence.length > 0 && (
                            <div className="w-full">
                                <div className="font-semibold mb-1">Pitch Sequence:</div>
                                <div className="space-y-1">
                                    {pitchSequence.map((pitch, index) => {
                                        // Filter out entries that don't have a pitch_type (these are summary entries added by backend)
                                        if (!pitch.pitch_type && atBatOutcome) {
                                            return null;
                                        }
                                        // Filter out duplicate contact entries and at-bat outcome entries
                                        if (pitch.outcome === 'contact' && atBatOutcome === 'contact') {
                                            return null;
                                        }
                                        // Also filter out any pitch that has the same outcome as the at-bat outcome
                                        if (pitch.outcome === atBatOutcome && atBatOutcome) {
                                            return null;
                                        }
                                        return (
                                            <div key={index} className="text-sm">
                                                <span className="font-medium">Pitch {index + 1}:</span> {pitch.outcome}{pitch.pitch_type ? ` (${pitch.pitch_type})` : ''}
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                        {atBatOutcome && (
                            <div className="w-full mt-2 p-3 rounded bg-green-100 text-center text-xl font-bold text-green-700 border border-green-300">
                                At-Bat Result: {atBatOutcome.toUpperCase()}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
} 