import { NextRequest, NextResponse } from "next/server";
import { generateText } from "ai";
import { getModel } from "@/lib/ai/providers";
import { withRetry } from "@/lib/retry";
import { ConsensusRowSchema } from "@/lib/validation";
import { pairwiseJaccard, pairwiseAgreement } from "@/lib/analytics";
import prisma from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const parsed = ConsensusRowSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid request", details: parsed.error.flatten() },
        { status: 400 }
      );
    }

    const {
      workers,
      reconciler,
      workerPrompt,
      reconcilerPrompt,
      userContent,
      rowIdx,
      runId,
      enableQualityScoring,
      enableDisagreementAnalysis,
      includeReasoning,
      temperature,
      maxTokens,
    } = parsed.data;
    const llmOpts = {
      ...(temperature !== undefined && { temperature }),
      ...(maxTokens ? { maxOutputTokens: maxTokens } : {}),
    };

    // Default direct-answer-only rules — overridden when the instruction above explicitly asks for reasoning
    const strictSuffix = `\n\nDEFAULT OUTPUT RULES (apply only when the instruction above does NOT explicitly request explanations, reasoning, arguments, justification, or rationale; if it does, follow the instruction and ignore these rules):
- Output ONLY the answer to the instruction. No notes, no explanations, no reasoning, no commentary, no caveats.
- Plain text or CSV only. NEVER use markdown: no **, no ## headings, no bullet points, no code blocks, no backticks.
- Do NOT add headers, labels, introductions, or sign-offs.
- Do NOT prefix with "Answer:", "Result:", "Note:", or any label.
- Do NOT add extra sentences, context, or qualifications.
- If the instruction asks for a single value, return that value and NOTHING else.`;

    // Step 1: Run workers in parallel
    const enforcedWorkerPrompt = workerPrompt + strictSuffix;
    const workerPromises = workers.map(async (w, i) => {
      const model = getModel(w.provider, w.model, w.apiKey || "local", w.baseUrl);
      // Per-worker persona prepended to system prompt
      const workerSystem = w.persona ? `${w.persona}\n\n${enforcedWorkerPrompt}` : enforcedWorkerPrompt;
      const start = Date.now();
      const { text } = await withRetry(
        () =>
          generateText({
            model,
            system: workerSystem,
            prompt: userContent,
            ...llmOpts,
          }),
        { maxAttempts: 3, baseDelayMs: 100 }
      );
      return {
        id: `worker_${i + 1}`,
        output: text,
        latency: (Date.now() - start) / 1000,
      };
    });

    const workerSettled = await Promise.allSettled(workerPromises);
    const workerResults = workerSettled
      .filter((r): r is PromiseFulfilledResult<{ id: string; output: string; latency: number }> => r.status === "fulfilled")
      .map((r) => r.value);

    if (workerResults.length < 1) {
      const errors = workerSettled
        .filter((r): r is PromiseRejectedResult => r.status === "rejected")
        .map((r) => (r.reason instanceof Error ? r.reason.message : String(r.reason)));
      return NextResponse.json(
        { error: `Not enough workers succeeded (${workerResults.length}/${workers.length}). Errors: ${errors.join("; ")}` },
        { status: 502 }
      );
    }

    // Step 2: Inter-rater analytics (all workers, set-based)
    const outputs = workerResults.map((r) => r.output.trim());
    const allSame = outputs.every((o) => o === outputs[0]);
    const kappa = pairwiseJaccard(outputs);

    // Pairwise matrix for detailed view
    const allTokenized = outputs.map((o) =>
      o
        .split(/[,\n]+/)
        .map((s) => s.trim())
        .filter(Boolean)
    );
    const maxAllLen = Math.max(...allTokenized.map((t) => t.length), 1);
    const padded = allTokenized.map((t) =>
      t.concat(new Array(Math.max(0, maxAllLen - t.length)).fill(""))
    );
    const agreementMatrix = pairwiseAgreement(padded);

    // Step 3: Run reconciler — also classify consensus level
    const reconcilerModel = getModel(
      reconciler.provider,
      reconciler.model,
      reconciler.apiKey || "local",
      reconciler.baseUrl
    );
    const workersFormatted = workerResults
      .map((r) => `${r.id} response:\n${r.output}`)
      .join("\n\n---\n\n");
    const combinedContent = `Worker Instruction: ${workerPrompt}\n\nOriginal Data: ${userContent}\n\nWorker Responses:\n${workersFormatted}`;
    const reconcilerPersonaPrefix = reconciler.persona ? `${reconciler.persona}\n\n` : "";

    const taskContext = `\n\nTHE WORKERS WERE GIVEN THIS INSTRUCTION:
${workerPrompt}

GROUNDING RULE (highest priority — read carefully):
- Your output MUST be derived ONLY from the content of the Worker Responses.
- The "Original Data" is provided ONLY as context so you can understand what the workers were given. You MUST NOT use it as a source for your own output.
- Do NOT translate, summarize, classify, rewrite, or otherwise transform the Original Data yourself. That was the workers' job, not yours.
- Do NOT add facts, claims, or content that no worker provided. If a detail is not present in any worker's output, it must NOT appear in yours.
- Your output's language and format MUST match what the worker outputs use. If workers output English text, you output English text. If workers output keywords or codes, you output keywords or codes. Do NOT translate, paraphrase, or transform worker outputs into a different language or form — even if the worker instruction asked for that language/form.
- Your job is to judge or combine the workers' outputs — never to redo the workers' task. You are not allowed to "fix" or "complete" a worker's failure to comply.

TASK COMPLIANCE GUIDANCE:
- Use the worker instruction above to recognize whether each worker actually performed the task. A worker that returned the Original Data verbatim, returned content in the wrong language/format, or refused has NOT complied.
- "Accuracy" means accuracy with respect to the worker instruction applied to the data, NOT similarity to the Original Data itself.
- Prefer outputs from workers that complied with the instruction. If only one worker complied, base your answer on that worker.
- If NO worker complied, you must STILL stay grounded in the worker outputs as they are. Pick or merge from what the workers actually produced — do NOT silently perform the task yourself on the Original Data, and do NOT translate or transform worker outputs to compensate for their failure.`;

    let reconcilerOutput: string;
    let reconcilerLatency: number;
    let consensusType: string;
    let reconcilerReasoning: string | undefined;

    const reconcilerDirectSuffix = `\n\nDEFAULT OUTPUT RULES (apply only when the worker instruction above does NOT explicitly request explanations, reasoning, arguments, justification, or rationale; if it does, mirror that level of detail in your final answer and ignore these rules):
- Output ONLY the final answer. No explanations, no reasoning, no commentary, no justifications.
- Plain text only. No markdown, no headings, no bullet points, no code fences.
- Do NOT explain why you chose this answer — just give the answer directly.`;

    if (allSame) {
      consensusType = "Unanimous";
      const reconcilerStart = Date.now();
      const { text } = await withRetry(
        () =>
          generateText({
            model: reconcilerModel,
            system: reconcilerPersonaPrefix + reconcilerPrompt + taskContext + reconcilerDirectSuffix,
            prompt: combinedContent,
            ...llmOpts,
          }),
        { maxAttempts: 3, baseDelayMs: 100 }
      );
      reconcilerOutput = text;
      reconcilerLatency = (Date.now() - reconcilerStart) / 1000;
    } else {
      const consensusSuffix = reconcilerDirectSuffix + `\n\nADDITIONAL TASK — After producing your direct answer, you MUST end your response with a consensus classification on its own line, in this exact format:
[CONSENSUS: <level>]
Where <level> is one of:
- "Unanimous" — all workers conveyed the same meaning (even if worded differently)
- "Strong Agreement" — workers mostly agree with only minor differences in detail or phrasing
- "Partial Agreement" — workers agree on some points but differ on others
- "Divergent" — workers gave substantially different or contradictory responses`;

      const reconcilerStart = Date.now();
      const { text: rawReconciler } = await withRetry(
        () =>
          generateText({
            model: reconcilerModel,
            system: reconcilerPersonaPrefix + reconcilerPrompt + taskContext + consensusSuffix,
            prompt: combinedContent,
            ...llmOpts,
          }),
        { maxAttempts: 3, baseDelayMs: 100 }
      );
      reconcilerLatency = (Date.now() - reconcilerStart) / 1000;

      // Parse [CONSENSUS: ...] — tolerant of missing ], quotes, trailing whitespace
      const consensusMatch = rawReconciler.match(/\[CONSENSUS:\s*"?([^"\]\n]+)"?\]?\s*$/im);
      if (consensusMatch) {
        const level = consensusMatch[1].trim();
        const valid = ["Unanimous", "Strong Agreement", "Partial Agreement", "Divergent"];
        consensusType = valid.includes(level) ? level : "Partial Agreement";
        reconcilerOutput = rawReconciler.slice(0, consensusMatch.index).trim();
      } else {
        consensusType = "Partial Agreement";
        reconcilerOutput = rawReconciler.trim();
      }
    }

    const totalLatency = reconcilerLatency + Math.max(...workerResults.map((r) => r.latency));

    // Step 4 (optional): Quality scoring
    let qualityScores: (number | null)[] | undefined;
    if (enableQualityScoring) {
      try {
        const { text: qsText } = await withRetry(
          () =>
            generateText({
              model: reconcilerModel,
              system: `You are a quality assessor evaluating worker responses. Rate each worker on a scale of 1-10.

SCORING CRITERIA:
- Task compliance (did the worker actually perform the Worker Instruction on the Original Data? A response that returns the Original Data verbatim, refuses, or is in the wrong language fails the instruction and must score very low — typically 1-3.)
- Accuracy (is the response correct with respect to the Worker Instruction applied to the Original Data — NOT similarity to the Original Data itself?)
- Completeness (does it cover all relevant aspects required by the instruction?)
- Relevance (does it stay focused on the task?)
- Alignment with best answer (how close is it to the chosen reconciler output?)

RULES:
- If all workers gave the same answer, they should all receive the same score.
- A response that matches the reconciler's chosen answer closely should score higher.
- A response that did not perform the requested task is invalid — score it 1-3 regardless of fluency.
- Deduct points for: factual errors, missing key information, off-topic content, unnecessary additions.
- Be consistent: similar quality responses should get similar scores.

Return ONLY valid JSON: {"quality_scores":[N,N,...]} where N is a decimal number between 1.0 and 10.0 (one decimal place, e.g. 6.5, 7.3, 9.0). No other text.`,
              prompt: `Worker Instruction: ${workerPrompt}\n\nOriginal Data: ${userContent}\n\nWorker Responses:\n${workersFormatted}\n\nReconciler's Chosen Answer:\n${reconcilerOutput}\n\nConsensus Level: ${consensusType}`,
              ...llmOpts,
            }),
          { maxAttempts: 2, baseDelayMs: 100 }
        );
        const cleaned = qsText.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
        const tryParse = (s: string): { quality_scores: unknown } | null => {
          try { return JSON.parse(s) as { quality_scores: unknown }; } catch { return null; }
        };
        let parsed = tryParse(cleaned);
        if (!parsed) {
          const objMatch = cleaned.match(/\{[\s\S]*\}/);
          if (objMatch) parsed = tryParse(objMatch[0]);
        }
        if (!parsed) {
          const arrMatch = cleaned.match(/\[[\s\S]*?\]/);
          if (arrMatch) parsed = tryParse(`{"quality_scores":${arrMatch[0]}}`);
        }

        if (parsed && Array.isArray(parsed.quality_scores)) {
          const raw = parsed.quality_scores as unknown[];
          const isFiniteNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
          // Normalize: workers with identical outputs must receive identical scores.
          // Group by normalized output text, average the reconciler's scores within each
          // group, and assign the group average to every member.
          const normalize = (s: string) => s.trim().replace(/\s+/g, " ").toLowerCase();
          const groups = new Map<string, number[]>();
          outputs.forEach((out, i) => {
            const v = raw[i];
            if (!isFiniteNum(v)) return;
            const key = normalize(out);
            const bucket = groups.get(key);
            if (bucket) bucket.push(v);
            else groups.set(key, [v]);
          });
          qualityScores = outputs.map((out, i) => {
            const bucket = groups.get(normalize(out));
            if (bucket && bucket.length > 0) {
              const avg = bucket.reduce((a, b) => a + b, 0) / bucket.length;
              return Math.round(avg * 10) / 10;
            }
            const v = raw[i];
            return isFiniteNum(v) ? v : null;
          });
        } else {
          console.warn("[consensus-row] quality_scores parse failed:", qsText.slice(0, 300));
          qualityScores = outputs.map(() => null);
        }
      } catch (err) {
        console.warn("[consensus-row] quality scoring error:", err instanceof Error ? err.message : String(err));
        qualityScores = outputs.map(() => null);
      }
    }

    // Step 5 (optional): Reconciler reasoning — separate call for reliability
    if (includeReasoning && !allSame) {
      try {
        const { text: jrText } = await withRetry(
          () =>
            generateText({
              model: reconcilerModel,
              system: `You are a reconciler explaining your decision. Given the original data, the worker responses, and your chosen best answer, explain in one or two sentences why you chose this answer over the alternatives. Return ONLY the explanation, no labels or prefixes.`,
              prompt: `Worker Instruction: ${workerPrompt}\n\nOriginal Data: ${userContent}\n\nWorker Responses:\n${workersFormatted}\n\nChosen Answer:\n${reconcilerOutput}`,
              ...llmOpts,
            }),
          { maxAttempts: 2, baseDelayMs: 100 }
        );
        reconcilerReasoning = jrText.trim() || "Could not generate reasoning";
      } catch {
        reconcilerReasoning = "Could not generate reasoning";
      }
    }

    // Step 6 (optional): Disagreement analysis
    let disagreementReason: string | undefined;
    if (enableDisagreementAnalysis && consensusType !== "Unanimous") {
      try {
        const { text: drText } = await withRetry(
          () =>
            generateText({
              model: reconcilerModel,
              system: `You are an expert analyst. In exactly one sentence, explain why the workers disagreed.`,
              prompt: `Worker Instruction: ${workerPrompt}\n\nOriginal Data: ${userContent}\n\nWorker Responses:\n${workersFormatted}`,
              ...llmOpts,
            }),
          { maxAttempts: 2, baseDelayMs: 100 }
        );
        disagreementReason = drText.trim() || "Could not analyze disagreement";
      } catch {
        disagreementReason = "Could not analyze disagreement";
      }
    }

    if (runId) {
      await prisma.runResult.create({
        data: {
          runId,
          rowIndex: rowIdx ?? 0,
          inputJson: JSON.stringify({ content: userContent }),
          output: JSON.stringify({ workers: workerResults, reconciler: reconcilerOutput, consensus: consensusType, kappa }),
          status: "SUCCESS",
          latency: totalLatency,
        },
      });
    }

    return NextResponse.json({
      workerResults,
      reconcilerOutput,
      reconcilerLatency,
      consensusType,
      kappa: isNaN(kappa) ? null : kappa,
      kappaLabel: isNaN(kappa) ? "N/A" : kappa < 0.2 ? "Very Low" : kappa < 0.4 ? "Low" : kappa < 0.6 ? "Moderate" : kappa < 0.8 ? "High" : "Very High",
      agreementMatrix,
      ...(reconcilerReasoning !== undefined ? { reconcilerReasoning } : {}),
      ...(qualityScores !== undefined ? { qualityScores } : {}),
      ...(disagreementReason !== undefined ? { disagreementReason } : {}),
    });
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error("consensus-row error:", msg);
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
