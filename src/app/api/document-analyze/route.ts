import { NextRequest, NextResponse } from "next/server";
import { generateText } from "ai";
import { getModel } from "@/lib/ai/providers";
import { withRetry } from "@/lib/retry";
import { DocumentAnalyzeSchema } from "@/lib/validation";
import { getPrompt } from "@/lib/prompts";

// Only take the first 3000 chars for field analysis — enough context, cheaper call
const ANALYSIS_CHAR_LIMIT = 3_000;

async function extractTextForAnalysis(fileContent: string, fileType: string): Promise<string> {
  const buffer = Buffer.from(fileContent, "base64");

  if (["txt", "md", "json", "html", "csv"].includes(fileType)) {
    return buffer.toString("utf-8").slice(0, ANALYSIS_CHAR_LIMIT);
  }

  if (fileType === "pdf") {
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const pdfjsLib = await import("pdfjs-dist") as any;
      pdfjsLib.GlobalWorkerOptions.workerSrc = "";
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { PDFParse } = await import("pdf-parse") as any;
      const parser = new PDFParse({ data: buffer });
      const result = await parser.getText();
      return (result.text as string).slice(0, ANALYSIS_CHAR_LIMIT);
    } catch {
      return "";
    }
  }

  if (fileType === "docx") {
    try {
      const mammoth = await import("mammoth");
      const result = await mammoth.extractRawText({ buffer });
      return result.value.slice(0, ANALYSIS_CHAR_LIMIT);
    } catch {
      return "";
    }
  }

  return "";
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const parsed = DocumentAnalyzeSchema.safeParse(body);
    if (!parsed.success) {
      return NextResponse.json(
        { error: "Invalid request", details: parsed.error.flatten() },
        { status: 400 }
      );
    }

    const { fileContent, fileType, fileName, provider, model, apiKey, baseUrl } = parsed.data;

    const text = await extractTextForAnalysis(fileContent, fileType);
    if (!text.trim()) {
      return NextResponse.json({ fields: [] });
    }

    const aiModel = getModel(provider, model, apiKey, baseUrl);

    const { text: response } = await withRetry(
      () =>
        generateText({
          model: aiModel,
          system: getPrompt("document.analysis"),
          prompt: `Document: ${fileName ?? "untitled"}\n\n${text}`,
          temperature: 0,
          maxOutputTokens: 1024,
        }),
      { maxAttempts: 2, baseDelayMs: 200 }
    );

    let fields: unknown[] = [];
    try {
      const cleaned = response
        .replace(/^```(?:json)?\s*/im, "")
        .replace(/\s*```\s*$/m, "")
        .trim();
      const parsedJson = JSON.parse(cleaned);
      fields = Array.isArray(parsedJson) ? parsedJson : [];
    } catch {
      // Graceful degradation — return empty fields rather than error
    }

    return NextResponse.json({ fields });
  } catch (error: unknown) {
    const msg = error instanceof Error ? error.message : String(error);
    console.error("document-analyze error:", msg);
    return NextResponse.json({ fields: [] });
  }
}
