import { NextRequest, NextResponse } from "next/server";
import { generateText } from "ai";
import { getModel } from "@/lib/ai/providers";
import { withRetry } from "@/lib/retry";
import { AgentNetworkRowSchema } from "@/lib/validation";
import prisma from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const parsed = AgentNetworkRowSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid request", details: parsed.error.flatten() },
        { status: 400 }
      );
    }

    const { agents, userContent, maxRounds, communicationStyle, convergenceMode, convergenceThreshold, rowIdx, runId, temperature, maxTokens } = parsed.data;
    const adaptive = convergenceMode === "adaptive";
    const threshold = convergenceThreshold ?? 15;
    const hardMaxRounds = adaptive ? 5 : maxRounds;

    const STYLE_PROMPTS: Record<string, string> = {
      collaborative: "Review the other agents' outputs below. Incorporate their strongest insights into your answer. Build on areas of agreement, resolve minor differences by merging perspectives, and produce an improved unified response.",
      adversarial: "Review the other agents' outputs below. Critically challenge their claims — identify weaknesses, logical gaps, unsupported assumptions, or factual errors. Defend your own position where you believe you are correct. If another agent makes a valid point that contradicts yours, acknowledge it but explain why your overall conclusion still holds or revise only what is genuinely wrong.",
      deliberative: "Review the other agents' outputs below. Systematically evaluate each agent's position: list the strengths and weaknesses of each perspective. Then produce your revised answer by weighing the evidence objectively. Explain briefly which points you adopted and which you rejected and why.",
      socratic: "Review the other agents' outputs below. For each agent's position, identify the underlying assumptions and question whether they hold. Ask probing questions about their reasoning. Then produce your revised answer that addresses these deeper questions and strengthens your reasoning.",
    };
    const styleInstruction = communicationStyle ? (STYLE_PROMPTS[communicationStyle] || "Refine your answer based on the above.") : "Refine your answer based on the above.";

    const roundOutputs: Array<{
      round: number;
      outputs: Array<{ label: string; output: string; latency: number }>;
    }> = [];
    const latestOutputs: Record<string, string> = {};

    let converged = false;
    let roundsTaken = 0;

    for (let round = 1; round <= hardMaxRounds; round++) {
      roundsTaken = round;
      const previousOutputs = { ...latestOutputs };

      const promises = agents.map(async (agent) => {
        const model = getModel(agent.provider, agent.model, agent.apiKey || "local", agent.baseUrl);

        // Per-agent input (its own column subset) when provided, else the shared content.
        let agentContent = agent.userContent ?? userContent;

        // For rounds 2+, append all other agents' previous outputs
        if (round > 1) {
          const othersSection = Object.entries(previousOutputs)
            .filter(([label]) => label !== agent.label)
            .map(([label, output]) => `[${label}]:\n${output}`)
            .join("\n\n");
          agentContent += `\n\n--- Other agents' outputs from round ${round - 1} ---\n\n${othersSection}\n\n--- ${styleInstruction} ---`;
        }

        const start = Date.now();
        const { text } = await withRetry(
          () => generateText({
            model,
            system: agent.role || undefined,
            prompt: agentContent,
            ...(temperature !== undefined && { temperature }),
            ...(maxTokens ? { maxOutputTokens: maxTokens } : {}),
          }),
          { maxAttempts: 3, baseDelayMs: 100 }
        );
        return { label: agent.label, output: text.trim(), latency: (Date.now() - start) / 1000 };
      });

      const settled = await Promise.allSettled(promises);
      const results = settled
        .filter((r): r is PromiseFulfilledResult<{ label: string; output: string; latency: number }> => r.status === "fulfilled")
        .map((r) => r.value);

      if (results.length === 0) {
        const errors = settled
          .filter((r): r is PromiseRejectedResult => r.status === "rejected")
          .map((r) => (r.reason instanceof Error ? r.reason.message : String(r.reason)));
        return NextResponse.json(
          { error: `All agents failed in round ${round}: ${errors.join("; ")}` },
          { status: 502 }
        );
      }

      for (const r of results) {
        latestOutputs[r.label] = r.output;
      }

      roundOutputs.push({ round, outputs: results });

      // Strict convergence: all outputs identical to previous round
      if (round > 1) {
        converged = results.every((r) => previousOutputs[r.label] === r.output);
        if (converged) break;
      }

      // Adaptive convergence: judge model scores semantic delta
      if (adaptive && round > 1) {
        try {
          const judgeAgent = agents[0];
          const judgeModel = getModel(judgeAgent.provider, judgeAgent.model, judgeAgent.apiKey || "local", judgeAgent.baseUrl);
          const comparison = agents.map((a) => {
            const prev = previousOutputs[a.label] ?? "";
            const curr = latestOutputs[a.label] ?? "";
            return `[${a.label}]\nROUND ${round - 1}:\n${prev}\n\nROUND ${round}:\n${curr}`;
          }).join("\n\n---\n\n");

          const { text: judgeText } = await withRetry(
            () => generateText({
              model: judgeModel,
              system: `You are a convergence detector for a multi-agent deliberation. Compare each agent's output between two consecutive rounds. Score how much agents have changed their positions overall.

Reply with ONLY a JSON object: {"delta": <number 0-100>, "stable": <boolean>}
- delta: 0 = essentially identical (semantic), 100 = completely different positions
- stable: true if agents have converged to stable positions (small revisions only)`,
              prompt: comparison,
              temperature: 0,
              maxOutputTokens: 500,
            }),
            { maxAttempts: 2, baseDelayMs: 100 }
          );

          let delta = 100;
          try {
            let cleaned = judgeText.trim();
            if (cleaned.startsWith("```")) cleaned = cleaned.replace(/^```(?:json)?\s*/, "").replace(/\s*```$/, "");
            const parsed = JSON.parse(cleaned) as { delta: number; stable?: boolean };
            delta = typeof parsed.delta === "number" ? parsed.delta : 100;
          } catch { /* parse failed */ }

          if (delta <= threshold) {
            converged = true;
            break;
          }
        } catch {
          // Judge failed — continue
        }
      }
    }

    // If converged, all agents agree — pick the first
    const allOutputs = Object.values(latestOutputs);
    const allSame = allOutputs.every((o) => o === allOutputs[0]);
    const finalConsensus = allSame ? allOutputs[0] : null;

    const totalLatency = roundOutputs.reduce(
      (sum, r) => sum + Math.max(...r.outputs.map((o) => o.latency), 0),
      0
    );

    if (runId) {
      await prisma.runResult.create({
        data: {
          runId,
          rowIndex: rowIdx ?? 0,
          inputJson: JSON.stringify({ content: userContent }),
          output: JSON.stringify({ roundOutputs, rounds: roundsTaken, converged, finalConsensus }),
          status: "SUCCESS",
          latency: totalLatency,
        },
      });
    }

    return NextResponse.json({ roundOutputs, roundsTaken, converged, finalConsensus });
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error("agent-network-row error:", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
