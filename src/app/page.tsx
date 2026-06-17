"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Bot, Database, Edit3, Wand2, Columns, History, BookOpen, Sparkles, FileArchive, TableProperties, FlaskConical, ArrowRight, MoreVertical, Printer, Video, RefreshCw, Settings, Clock, LayoutDashboard } from "lucide-react";
import Link from "next/link";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";

const CATEGORIES = [
  {
    name: "Data Processing",
    tools: [
      {
        title: "Transform Data",
        description: "Apply a single AI prompt to every row of a CSV — classify, label, summarize, or rewrite at scale.",
        icon: Wand2,
        href: "/transform",
        color: "text-blue-400",
        bg: "bg-blue-50 dark:bg-blue-950/30",
        border: "hover:border-blue-200 dark:hover:border-blue-800",
      },
      {
        title: "General Automator",
        description: "Chain multiple AI prompts into a pipeline — each step feeds the next, processed in parallel across rows.",
        icon: Database,
        href: "/automator",
        color: "text-indigo-500",
        bg: "bg-indigo-50 dark:bg-indigo-950/30",
        border: "hover:border-indigo-200 dark:hover:border-indigo-800",
      },
      {
        title: "Generate Data",
        description: "Generate synthetic datasets from a column schema or a free-form description — useful for prototyping pipelines.",
        icon: Sparkles,
        href: "/generate",
        color: "text-cyan-500",
        bg: "bg-cyan-50 dark:bg-cyan-950/30",
        border: "hover:border-cyan-200 dark:hover:border-cyan-800",
      },
      {
        title: "Extract Data",
        description: "Pull structured rows out of PDFs or DOCX into a table — define columns once, run across many files.",
        icon: TableProperties,
        href: "/extract-data",
        color: "text-teal-500",
        bg: "bg-teal-50 dark:bg-teal-950/30",
        border: "hover:border-teal-200 dark:hover:border-teal-800",
      },
      {
        title: "Process Documents",
        description: "Run free-form prompts over PDFs, DOCX, or plain text — one AI response per file, exportable as CSV.",
        icon: FileArchive,
        href: "/process-documents",
        color: "text-violet-500",
        bg: "bg-violet-50 dark:bg-violet-950/30",
        border: "hover:border-violet-200 dark:hover:border-violet-800",
      },
    ],
  },
  {
    name: "Multi Agent System",
    tools: [
      {
        title: "Model Comparison",
        description: "Run one prompt across multiple LLMs side-by-side, reconcile into a single answer, and score agreement with Cohen's kappa.",
        icon: Columns,
        href: "/model-comparison",
        color: "text-blue-500",
        bg: "bg-blue-50 dark:bg-blue-950/30",
        border: "hover:border-blue-200 dark:hover:border-blue-800",
      },
      {
        title: "Multi-Agent Workflows",
        description: "Orchestrate multi-agent workflows — sequential pipelines, parallel deliberation rounds, or reconciler-with-workers.",
        icon: LayoutDashboard,
        href: "/multi-agent-workflows",
        color: "text-purple-500",
        bg: "bg-purple-50 dark:bg-purple-950/30",
        border: "hover:border-purple-200 dark:hover:border-purple-800",
      },
    ],
  },
  {
    name: "Data Coding",
    tools: [
      {
        title: "Codebook Generator",
        description: "Build a qualitative codebook from raw text in three AI passes — discovery, consolidation, definition — with editable output.",
        icon: BookOpen,
        href: "/codebook-generator",
        color: "text-emerald-500",
        bg: "bg-emerald-50 dark:bg-emerald-950/30",
        border: "hover:border-emerald-200 dark:hover:border-emerald-800",
      },
      {
        title: "Qualitative Coder",
        description: "Assign codes from your codebook to each text row — AI does the first pass, you keep the final say.",
        icon: Edit3,
        href: "/qualitative-coder",
        color: "text-orange-500",
        bg: "bg-orange-50 dark:bg-orange-950/30",
        border: "hover:border-orange-200 dark:hover:border-orange-800",
      },
      {
        title: "AI Coder",
        description: "Thematic analysis with confidence-scored AI suggestions, row-by-row review, and built-in agreement analytics.",
        icon: Bot,
        href: "/ai-coder",
        color: "text-orange-400",
        bg: "bg-orange-50 dark:bg-orange-950/30",
        border: "hover:border-orange-200 dark:hover:border-orange-800",
      },
      {
        title: "Abstract Screener",
        description: "Include/exclude screening for systematic reviews — title + abstract in, decision and rationale out, PRISMA-ready.",
        icon: FlaskConical,
        href: "/abstract-screener",
        color: "text-pink-500",
        bg: "bg-pink-50 dark:bg-pink-950/30",
        border: "hover:border-pink-200 dark:hover:border-pink-800",
      },
    ],
  },
  {
    name: "System",
    tools: [
      {
        title: "Historical Runs",
        description: "Browse, restore, or re-export every past run — full prompts, results, and latency/cost metrics preserved.",
        icon: History,
        href: "/history",
        color: "text-slate-500",
        bg: "bg-slate-50 dark:bg-slate-900/30",
        border: "hover:border-slate-300 dark:hover:border-slate-600",
      },
      {
        title: "Settings",
        description: "Configure API keys and providers, tune concurrency and retry behavior, and probe local Ollama / LM Studio models.",
        icon: Settings,
        href: "/settings",
        color: "text-slate-500",
        bg: "bg-slate-50 dark:bg-slate-900/30",
        border: "hover:border-slate-300 dark:hover:border-slate-600",
      },
    ],
  },
];

export default function HomePage() {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const toggleScreencast = useCallback(async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
      const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : "video/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      chunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `handai_screencast_${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
        stream.getTracks().forEach((t) => t.stop());
        setIsRecording(false);
      };
      stream.getVideoTracks()[0]?.addEventListener("ended", () => {
        recorder.stop();
      });
      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
    } catch {
      // User cancelled display picker
    }
  }, [isRecording]);

  useEffect(() => {
    return () => { mediaRecorderRef.current?.stop(); };
  }, []);

  return (
    <div className="space-y-10 pb-16 animate-in fade-in duration-500">

      {/* Hero */}
      <div className="space-y-2 pb-2 flex items-start justify-between flex-wrap gap-2">
        <div className="min-w-0 flex-1">
          <h1 className="text-4xl font-bold tracking-tight">Welcome to Handai</h1>
          <p className="text-lg text-muted-foreground">
            Your AI-powered qualitative research and data science suite.
          </p>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="shrink-0 mt-1 relative">
              <MoreVertical className="h-5 w-5" />
              {isRecording && (
                <span className="absolute top-0.5 right-0.5 h-2.5 w-2.5 rounded-full bg-red-500 animate-pulse" />
              )}
              <span className="sr-only">Actions</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => window.print()}>
              <Printer className="h-4 w-4" />
              Print Page
            </DropdownMenuItem>
            <DropdownMenuItem onClick={toggleScreencast}>
              <Video className="h-4 w-4" />
              {isRecording ? "Stop Recording" : "Screencast"}
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => window.location.reload()}>
              <RefreshCw className="h-4 w-4" />
              Refresh
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <Link href="/settings">
                <Settings className="h-4 w-4" />
                Settings
              </Link>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <Link href="/history">
                <Clock className="h-4 w-4" />
                Historical Runs
              </Link>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Categories */}
      {CATEGORIES.map((cat) => (
        <div key={cat.name} className="space-y-4">
          <h2 className="text-2xl font-bold">{cat.name}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {cat.tools.map((tool) => (
              <Link key={tool.title} href={tool.href} className="group">
                <div className={`h-full rounded-xl border bg-card p-5 transition-all duration-200 hover:shadow-md ${tool.border}`}>
                  {/* Icon */}
                  <div className={`inline-flex items-center justify-center w-11 h-11 rounded-lg ${tool.bg} mb-4`}>
                    <tool.icon className={`w-6 h-6 ${tool.color}`} />
                  </div>

                  {/* Title */}
                  <h3 className="text-base font-semibold mb-1 group-hover:text-primary transition-colors">
                    {tool.title}
                  </h3>

                  {/* Description */}
                  <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                    {tool.description}
                  </p>

                  {/* CTA */}
                  <div className="flex items-center gap-1 text-xs font-medium text-primary opacity-0 group-hover:opacity-100 transition-opacity">
                    Open <ArrowRight className="h-3 w-3" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
