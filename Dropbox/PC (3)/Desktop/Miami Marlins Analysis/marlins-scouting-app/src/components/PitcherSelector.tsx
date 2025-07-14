import React from "react";

interface PitcherSelectorProps {
    value: string | null;
    onChange: (value: string) => void;
}

const MARLINS_PITCHERS = [
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
];

const PitcherSelector: React.FC<PitcherSelectorProps> = ({ value, onChange }) => (
    <div className="w-full">
        <label className="block font-semibold mb-2 text-gray-900">Select Marlins Pitcher</label>
        <select
            className="w-full border rounded px-3 py-2 text-gray-900"
            value={value || ""}
            onChange={e => onChange(e.target.value)}
        >
            <option value="" disabled>
                Choose a pitcher
            </option>
            {MARLINS_PITCHERS.map(p => (
                <option key={p} value={p} className="text-gray-900">
                    {p}
                </option>
            ))}
        </select>
    </div>
);

export default PitcherSelector; 