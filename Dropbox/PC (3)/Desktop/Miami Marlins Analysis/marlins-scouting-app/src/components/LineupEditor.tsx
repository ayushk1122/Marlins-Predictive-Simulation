import React, { useEffect, useState } from "react";
import axios from "axios";
import {
    DndContext,
    closestCenter,
    PointerSensor,
    useSensor,
    useSensors,
    DragEndEvent
} from "@dnd-kit/core";
import {
    arrayMove,
    SortableContext,
    useSortable,
    verticalListSortingStrategy
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

type Props = {
    selectedTeam: string | null;
    onTeamChange: (team: string) => void;
    lineup: string[];
    setLineup: (lineup: string[]) => void;
    onViewHitterReport: (hitter: string) => void;
    teams?: string[];
};

const MLB_TEAMS = [
    "Braves", "Mets", "Phillies", "Nationals", "Cardinals", "Cubs", "Dodgers", "Giants", "Padres", "Yankees", "Red Sox",
    "Brewers", "Pirates", "Reds", "Diamondbacks", "Rockies", "Astros", "Rangers", "Angels", "Mariners", "Athletics", "Blue Jays", "Orioles", "Rays", "Royals", "Tigers", "Twins", "White Sox", "Guardians" // All except Marlins
];

// For demo, a pool of hitters to choose from
const HITTER_POOL = [
    "Acuna Jr.", "Albies", "Riley", "Olson", "Ozuna", "Rosario", "Arcia", "Murphy", "Harris II",
    "Nimmo", "Lindor", "Alonso", "McNeil", "Marte", "Baty", "Vientos", "Alvarez", "Stewart",
    "Schwarber", "Turner", "Harper", "Bohm", "Castellanos", "Stott", "Realmuto", "Marsh", "Sosa"
];

function DraggableHitter({ hitter, id, onViewHitterReport, onReplace, isEditing, onEdit, idx, lineup }: {
    hitter: string;
    id: string;
    onViewHitterReport: (hitter: string) => void;
    onReplace: (idx: number, newHitter: string) => void;
    isEditing: boolean;
    onEdit: (idx: number) => void;
    idx: number;
    lineup: string[];
}) {
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
        background: isDragging ? "#e0e7ff" : undefined,
        borderRadius: 6,
        padding: "0.25rem 0.5rem",
        display: "flex",
        alignItems: "center",
        gap: 8,
        marginBottom: 4,
        cursor: "grab"
    };
    return (
        <li ref={setNodeRef} style={style} {...attributes} {...listeners}>
            <span className="flex-1">{hitter}</span>
            <button
                className="text-blue-600 underline text-xs"
                onClick={e => { e.stopPropagation(); onViewHitterReport(hitter); }}
                type="button"
            >
                View Report
            </button>
            <button
                className="text-green-600 underline text-xs ml-2"
                onClick={e => { e.stopPropagation(); onEdit(idx); }}
                type="button"
            >
                Replace
            </button>
            {isEditing && (
                <select
                    className="ml-2 border rounded px-1 py-0.5 text-xs"
                    value={hitter}
                    onChange={e => onReplace(idx, e.target.value)}
                >
                    {HITTER_POOL.filter(h => !lineup.includes(h) || h === hitter).map(h => (
                        <option key={h} value={h}>{h}</option>
                    ))}
                </select>
            )}
            <span className="ml-2 text-gray-400 cursor-move">☰</span>
        </li>
    );
}

const LineupEditor: React.FC<Props> = ({ selectedTeam, onTeamChange, lineup, setLineup, onViewHitterReport, teams = MLB_TEAMS }) => {
    const [loading, setLoading] = useState(false);
    const [editingIdx, setEditingIdx] = useState<number | null>(null);
    const sensors = useSensors(useSensor(PointerSensor));

    useEffect(() => {
        if (!selectedTeam) return;
        setLoading(true);
        axios.get(`/api/lineup?team=${encodeURIComponent(selectedTeam)}`)
            .then(res => setLineup(res.data))
            .finally(() => setLoading(false));
    }, [selectedTeam, setLineup]);

    const handleDragEnd = (event: DragEndEvent) => {
        const { active, over } = event;
        if (active.id !== over?.id) {
            const oldIndex = lineup.findIndex(h => h === active.id);
            const newIndex = lineup.findIndex(h => h === over?.id);
            setLineup(arrayMove(lineup, oldIndex, newIndex));
        }
    };

    const handleReplace = (idx: number, newHitter: string) => {
        const newLineup = [...lineup];
        newLineup[idx] = newHitter;
        setLineup(newLineup);
        setEditingIdx(null);
    };

    return (
        <div>
            <label className="block font-semibold mb-2">Select Opponent Team</label>
            <select
                className="w-full border rounded px-3 py-2 mb-4"
                value={selectedTeam || ""}
                onChange={e => onTeamChange(e.target.value)}
            >
                <option value="" disabled>
                    Choose a team
                </option>
                {teams.map(t => (
                    <option key={t} value={t}>
                        {t}
                    </option>
                ))}
            </select>
            <div>
                <label className="block font-semibold mb-2">Lineup</label>
                {loading ? (
                    <div className="text-gray-500">Loading lineup...</div>
                ) : (
                    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                        <SortableContext items={lineup} strategy={verticalListSortingStrategy}>
                            <ol className="list-decimal pl-5">
                                {lineup.map((hitter, idx) => (
                                    <DraggableHitter
                                        key={hitter}
                                        id={hitter}
                                        hitter={hitter}
                                        onViewHitterReport={onViewHitterReport}
                                        onReplace={handleReplace}
                                        isEditing={editingIdx === idx}
                                        onEdit={setEditingIdx}
                                        idx={idx}
                                        lineup={lineup}
                                    />
                                ))}
                            </ol>
                        </SortableContext>
                    </DndContext>
                )}
            </div>
        </div>
    );
};

export default LineupEditor; 