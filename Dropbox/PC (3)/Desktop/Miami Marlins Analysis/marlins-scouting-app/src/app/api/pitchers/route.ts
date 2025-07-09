import { NextResponse } from "next/server";

export async function GET() {
    // Placeholder: Replace with real data fetching
    const pitchers = [
        "Sandy Alcantara",
        "Jesús Luzardo",
        "Edward Cabrera",
        "Braxton Garrett",
        "Trevor Rogers",
        "Ryan Weathers"
    ];
    return NextResponse.json(pitchers);
} 