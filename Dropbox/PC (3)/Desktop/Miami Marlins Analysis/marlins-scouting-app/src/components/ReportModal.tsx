import React from "react";
import { ScoutingReport, HitterReport } from "../types";
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

type Props = {
    open: boolean;
    onClose: () => void;
    type: "pitcher" | "hitter" | null;
    target: string | null;
    pitcher: string | null;
    lineup: string[];
    team: string | null;
    report?: ScoutingReport | null;
};

const ReportModal: React.FC<Props> = ({ open, onClose, type, target, pitcher, lineup, team, report }) => {
    if (!open) return null;

    let content = <p className="text-gray-900">Scouting analysis and stats will appear here.</p>;

    // Helper to render advanced pitcher data
    const renderPitcherReport = (pitcherReport: any) => {
        if (!pitcherReport || pitcherReport.error) return null;
        // Prepare TTO bar chart data
        const tto: any[] = pitcherReport.tto_usage || [];
        const ttoGroups = ['1st', '2nd', '3rd+'];
        const pitchTypes = Array.from(new Set(tto.map((d: any) => d.pitch_type)));
        const chartData = {
            labels: ttoGroups,
            datasets: (pitchTypes as string[]).map((pt: string, i: number) => ({
                label: pt,
                data: ttoGroups.map(tg => {
                    const found = tto.find((d: any) => d.tto_group === tg && d.pitch_type === pt);
                    return found ? found.percent : 0;
                }),
                backgroundColor: `hsl(${(i * 60) % 360}, 70%, 60%)`,
            }))
        };
        return (
            <div className="mb-4">
                <h3 className="text-lg font-bold text-gray-900 mb-2">Outings vs {team}</h3>
                <table className="w-full text-sm mb-2 border text-gray-900">
                    <thead>
                        <tr className="bg-gray-100 text-gray-900 font-bold">
                            <th className="p-1 border">Date</th>
                            <th className="p-1 border">IP</th>
                            <th className="p-1 border">ER</th>
                            <th className="p-1 border">R</th>
                            <th className="p-1 border">BB</th>
                            <th className="p-1 border">K</th>
                            <th className="p-1 border">Num Pitches</th>
                        </tr>
                    </thead>
                    <tbody>
                        {pitcherReport.outings?.map((o: any, i: number) => (
                            <tr key={i}>
                                <td className="p-1 border">{o.game_date}</td>
                                <td className="p-1 border">{o.IP}</td>
                                <td className="p-1 border">{o.ER}</td>
                                <td className="p-1 border">{o.R}</td>
                                <td className="p-1 border">{o.BB}</td>
                                <td className="p-1 border">{o.K}</td>
                                <td className="p-1 border">{o.num_pitches}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                <h3 className="text-lg font-bold text-gray-900 mb-2">Pitch Usage</h3>
                <table className="w-full text-sm mb-2 border text-gray-900">
                    <thead>
                        <tr className="bg-gray-100 text-gray-900 font-bold">
                            <th className="p-1 border">Pitch Type</th>
                            <th className="p-1 border">Count</th>
                        </tr>
                    </thead>
                    <tbody>
                        {pitcherReport.pitch_usage?.map((p: any, i: number) => (
                            <tr key={i}>
                                <td className="p-1 border">{p.pitch_type}</td>
                                <td className="p-1 border">{p.count}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                <h3 className="text-lg font-bold text-gray-900 mb-2">Pitch Usage by Times Through Order</h3>
                <div className="mb-4">
                    <Bar data={chartData} options={{
                        responsive: true,
                        plugins: { legend: { position: 'top' as const }, title: { display: false } },
                        scales: { y: { title: { display: true, text: 'Pitch Usage (%)' }, beginAtZero: true, max: 100 } }
                    }} />
                </div>
            </div>
        );
    };

    if (report) {
        if (type === "pitcher") {
            content = (
                <div>
                    <p className="mb-2 font-semibold text-gray-900">{report.summary}</p>
                    {renderPitcherReport(report.pitcherReport)}
                    <table className="w-full text-sm mb-2 border text-gray-900">
                        <thead>
                            <tr className="bg-gray-100 text-gray-900 font-bold">
                                <th className="p-1 border">Hitter</th>
                                <th className="p-1 border">Weaknesses</th>
                                <th className="p-1 border">xBA</th>
                                <th className="p-1 border">Whiff%</th>
                                <th className="p-1 border">Exit Velo</th>
                            </tr>
                        </thead>
                        <tbody>
                            {report.hitters.map((h: HitterReport) => (
                                <tr key={h.name}>
                                    <td className="p-1 border">{h.name}</td>
                                    <td className="p-1 border">{h.weaknesses.join(", ")}</td>
                                    <td className="p-1 border">{h.stats.xBA}</td>
                                    <td className="p-1 border">{h.stats.whiffPercent}%</td>
                                    <td className="p-1 border">{h.stats.avgExitVelo} mph</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <div className="text-xs text-gray-800">Recommendations and visuals coming soon.</div>
                </div>
            );
        } else if (type === "hitter" && target) {
            const h = report.hitters.find((h: HitterReport) => h.name === target);
            if (h) {
                content = (
                    <div>
                        <div className="mb-2 font-semibold text-gray-900">Key Weaknesses: {h.weaknesses.join(", ")}</div>
                        <div className="mb-2 text-gray-900">xBA: {h.stats.xBA} | Whiff%: {h.stats.whiffPercent}% | Exit Velo: {h.stats.avgExitVelo} mph</div>
                        <div className="mb-2 text-gray-900">Recommendations: {h.recommendations.join(", ")}</div>
                        <div className="text-xs text-gray-800">Visuals and advanced stats coming soon.</div>
                    </div>
                );
            }
        }
    }

    return (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-lg max-w-lg w-full max-h-[80vh] overflow-y-auto p-6 relative">
                <button
                    className="absolute top-2 right-2 text-gray-500 hover:text-gray-800"
                    onClick={onClose}
                >
                    ×
                </button>
                <h2 className="text-xl font-bold mb-2 text-gray-900">
                    {type === "pitcher" ? `Pitcher Report: ${target}` : `Hitter Report: ${target}`}
                </h2>
                <div className="mb-4 text-sm text-gray-900">
                    <div>Pitcher: {pitcher}</div>
                    <div>Opponent: {team}</div>
                    <div>Lineup: {lineup.join(", ")}</div>
                </div>
                <div className="text-base">
                    {content}
                </div>
            </div>
        </div>
    );
};

export default ReportModal; 