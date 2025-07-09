import React from "react";

const PITCHERS = [
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

type Props = {
    value: string | null;
    onChange: (pitcher: string) => void;
};

const PitcherSelector: React.FC<Props> = ({ value, onChange }) => (
    <div>
        <label className="block font-semibold mb-2">Select Marlins Pitcher</label>
        <select
            className="w-full border rounded px-3 py-2"
            value={value || ""}
            onChange={e => onChange(e.target.value)}
        >
            <option value="" disabled>
                Choose a pitcher
            </option>
            {PITCHERS.map(p => (
                <option key={p} value={p}>
                    {p}
                </option>
            ))}
        </select>
    </div>
);

export default PitcherSelector; 