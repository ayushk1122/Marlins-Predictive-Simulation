"use client";
import React, { useState } from "react";
import axios from "axios";
// Components to be created
import PitcherSelector from "../components/PitcherSelector";
import LineupEditor from "../components/LineupEditor";
import ReportModal from "../components/ReportModal";
import { ScoutingReport } from "../types";
import { useRouter } from "next/navigation";

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
  // Add more as needed
};

function StrikeZoneDisplay({ onSelect, selected }: { onSelect: (x: number, y: number) => void; selected: [number, number] | null }) {
  // 3x3 grid for demo
  const grid = Array.from({ length: 3 }, (_, row) => Array.from({ length: 3 }, (_, col) => [row, col]));
  return (
    <svg width={180} height={220} viewBox="0 0 180 220">
      {/* Plate and silhouette placeholder */}
      <rect x={60} y={40} width={60} height={120} fill="#f3f4f6" stroke="#333" strokeWidth={2} />
      {/* 3x3 grid */}
      {grid.flat().map(([row, col]) => (
        <rect
          key={`${row}-${col}`}
          x={60 + col * 20}
          y={40 + row * 40}
          width={20}
          height={40}
          fill={selected && selected[0] === row && selected[1] === col ? "#60a5fa" : "#fff"}
          stroke="#333"
          strokeWidth={1}
          style={{ cursor: "pointer" }}
          onClick={() => onSelect(row, col)}
        />
      ))}
    </svg>
  );
}

function AtBatSimulatorDemo() {
  const [pitcher, setPitcher] = useState<string | null>(null);
  const [team, setTeam] = useState<string | null>(null);
  const [hitter, setHitter] = useState<string | null>(null);
  const router = useRouter();

  const canSim = pitcher && team && hitter;

  const handleSimulate = () => {
    if (canSim) {
      // Pass selections as query params for demo
      router.push(`/simulate-atbat?pitcher=${encodeURIComponent(pitcher!)}&team=${encodeURIComponent(team!)}&hitter=${encodeURIComponent(hitter!)}`);
    }
  };

  return (
    <div className="w-full max-w-xl mx-auto mt-10 p-6 bg-white rounded shadow text-gray-900">
      <h2 className="text-2xl font-bold mb-4">At-Bat Simulator Demo</h2>
      <div className="mb-4">
        <PitcherSelector value={pitcher} onChange={setPitcher} />
      </div>
      <div className="mb-4">
        <label className="block font-semibold mb-2">Select Opponent Team</label>
        <select className="w-full border rounded px-3 py-2" value={team || ""} onChange={e => { setTeam(e.target.value); setHitter(null); }}>
          <option value="" disabled>Choose a team</option>
          {DEMO_TEAMS.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>
      <div className="mb-4">
        <label className="block font-semibold mb-2">Select Hitter</label>
        <select className="w-full border rounded px-3 py-2" value={hitter || ""} onChange={e => setHitter(e.target.value)} disabled={!team}>
          <option value="" disabled>Choose a hitter</option>
          {team && DEMO_LINEUPS[team].map(h => <option key={h} value={h}>{h}</option>)}
        </select>
      </div>
      <button className="bg-blue-600 text-white px-4 py-2 rounded font-semibold w-full" disabled={!canSim} onClick={handleSimulate}>
        Simulate At-Bat
      </button>
    </div>
  );
}

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gray-50 p-4 flex flex-col items-center">
      <AtBatSimulatorDemo />
    </main>
  );
}
