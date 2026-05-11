import { NextRequest, NextResponse } from "next/server";
import prisma from "@/lib/prisma";


export async function GET(
    req: NextRequest,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        const { id } = await params;
        const run = await prisma.run.findUnique({
            where: { id },
        });

        if (!run) {
            return NextResponse.json({ error: "Run not found" }, { status: 404 });
        }

        const results = await prisma.runResult.findMany({
            where: { runId: id },
            orderBy: { rowIndex: "asc" }
        });

        return NextResponse.json({ run, results });
    } catch (error: unknown) {
        return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
    }
}

export async function PATCH(
    req: NextRequest,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        const { id } = await params;
        const body = await req.json();
        const data: Record<string, unknown> = {};
        if (typeof body.inputFile === "string" && body.inputFile.trim()) {
            data.inputFile = body.inputFile.trim();
        }
        if (typeof body.avgLatency === "number") data.avgLatency = body.avgLatency;
        if (typeof body.totalDuration === "number") data.totalDuration = body.totalDuration;
        if (typeof body.status === "string") data.status = body.status;
        if (typeof body.completedAt === "string") data.completedAt = new Date(body.completedAt);
        if (typeof body.systemPrompt === "string") data.systemPrompt = body.systemPrompt;
        if (Object.keys(data).length === 0) {
            return NextResponse.json({ error: "no valid fields to update" }, { status: 400 });
        }
        const run = await prisma.run.update({ where: { id }, data });
        return NextResponse.json({ run });
    } catch (error: unknown) {
        return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
    }
}

export async function DELETE(
    _req: NextRequest,
    { params }: { params: Promise<{ id: string }> }
) {
    try {
        const { id } = await params;
        await prisma.runResult.deleteMany({ where: { runId: id } });
        await prisma.run.delete({ where: { id } });
        return NextResponse.json({ ok: true });
    } catch (error: unknown) {
        return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
    }
}
