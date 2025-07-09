import { NextRequest, NextResponse } from "next/server";

const LINEUPS: Record<string, string[]> = {
    Braves: ["Acuna Jr.", "Albies", "Riley", "Olson", "Ozuna", "Rosario", "Arcia", "Murphy", "Harris II"],
    Mets: ["Nimmo", "Lindor", "Alonso", "McNeil", "Marte", "Baty", "Vientos", "Alvarez", "Stewart"],
    Phillies: ["Schwarber", "Turner", "Harper", "Bohm", "Castellanos", "Stott", "Realmuto", "Marsh", "Sosa"],
    // Add more teams as needed
};

export async function GET(req: NextRequest) {
    const { searchParams } = new URL(req.url);
    const team = searchParams.get("team") || "Braves";
    const lineup = LINEUPS[team] || [];
    return NextResponse.json(lineup);
} 