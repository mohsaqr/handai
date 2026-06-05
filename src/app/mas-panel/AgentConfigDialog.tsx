"use client";

import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  AGENT_AVATAR_INDICES,
  AGENT_CATEGORIES,
  COMMUNICATION_STYLES,
  PERSONALITY_STYLES,
  RESPONSE_STYLES,
  type Agent,
  avatarStyle,
  buildAgentSystemPrefix,
  deleteAgent,
  listAgents,
  makeAgentId,
  saveAgent,
} from "@/lib/agent-library";
import { toast } from "sonner";
import {
  Save,
  Trash2,
  FolderOpen,
  Pencil,
  User,
  FileText,
  SlidersHorizontal,
  Cpu,
  Check,
  Ban,
  X,
} from "lucide-react";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  agent: Agent;
  onSave: (agent: Agent) => void;
  enabledProviders: { providerId: string; defaultModel: string }[];
}

const LIBRARY_COLLAPSED_COUNT = 12;

const PERSONALITY_DESCRIPTIONS: Record<string, string> = {
  Neutral: "No specific tone — nothing added to the prompt",
  Formal: "Professional, precise, business-like",
  Concise: "Short and to the point, no filler",
  Verbose: "Expansive, thorough, lots of detail",
  Technical: "Precise terminology, depth over accessibility",
  Empathetic: "Warm, supportive, attuned to the reader",
  Direct: "Blunt and unambiguous, no hedging",
  Diplomatic: "Tactful, balanced, softens disagreement",
};

const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  Neutral: "No specific role — nothing added to the prompt",
  Analyst: "Breaks down problems, finds patterns, reasons from data",
  Critic: "Probes weaknesses, flags risks, challenges claims",
  Creative: "Generates novel ideas, explores wide possibilities",
  Synthesizer: "Combines viewpoints, builds coherent summaries",
  Researcher: "Gathers evidence, cites sources, surveys context",
  "Devil's Advocate": "Argues the opposite, stress-tests consensus",
  Specialist: "Deep expertise in a narrow domain",
};

function providerLabel(id: string) {
  if (id === "lmstudio") return "LM Studio";
  if (id === "ollama") return "Ollama";
  return id.charAt(0).toUpperCase() + id.slice(1);
}

export function AgentConfigDialog({ open, onOpenChange, agent, onSave, enabledProviders }: Props) {
  const [draft, setDraft] = useState<Agent>(agent);
  const [library, setLibrary] = useState<Agent[]>([]);
  const [systemPromptOpen, setSystemPromptOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [showAllLibrary, setShowAllLibrary] = useState(false);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [avatarPickerOpen, setAvatarPickerOpen] = useState(false);
  const [doInput, setDoInput] = useState("");
  const [dontInput, setDontInput] = useState("");
  // Tracks the library-entry id this draft was loaded from (if any).
  // Null when editing an agent that doesn't yet exist in the library.
  const [sourceLibraryId, setSourceLibraryId] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      const all = listAgents();
      setDraft(agent);
      setLibrary(all);
      setPendingDeleteId(null);
      // If the incoming agent id matches a library entry, treat it as "loaded from library".
      const existing = all.find((a) => a.id === agent.id);
      setSourceLibraryId(existing ? existing.id : null);
    }
  }, [open, agent]);

  useEffect(() => {
    const refresh = () => setLibrary(listAgents());
    window.addEventListener("handai-agent-library-changed", refresh);
    return () => window.removeEventListener("handai-agent-library-changed", refresh);
  }, []);

  const handleSaveAndClose = () => {
    onSave(draft);
    onOpenChange(false);
  };

  // Update the existing library entry this draft was loaded from.
  const handleUpdateInLibrary = () => {
    if (!sourceLibraryId) return;
    if (!draft.name.trim()) {
      toast.error("Give the agent a name before saving");
      return;
    }
    saveAgent({ ...draft, id: sourceLibraryId });
    toast.success(`Updated "${draft.name}" in library`);
  };

  // Create a new library entry — independent of whatever the draft was loaded from.
  const handleSaveAsNew = () => {
    if (!draft.name.trim()) {
      toast.error("Give the agent a name before saving");
      return;
    }
    const newId = makeAgentId();
    saveAgent({ ...draft, id: newId });
    setSourceLibraryId(newId);
    toast.success(`Saved "${draft.name}" as new agent`);
  };

  const handleLoadFromLibrary = (id: string) => {
    const loaded = library.find((a) => a.id === id);
    if (!loaded) return;
    // Keep the current draft's id so we're updating the card slot, not swapping its identity.
    setDraft({ ...loaded, id: draft.id });
    setSourceLibraryId(loaded.id);
    setLibraryOpen(false);
    toast.success(`Loaded "${loaded.name}"`);
  };

  const handleDeleteFromLibrary = (id: string) => {
    deleteAgent(id);
    toast.success("Removed from library");
  };

  const addDo = () => {
    const v = doInput.trim();
    if (!v) return;
    setDraft({ ...draft, dos: [...(draft.dos ?? []), v] });
    setDoInput("");
  };
  const removeDo = (i: number) =>
    setDraft({ ...draft, dos: (draft.dos ?? []).filter((_, idx) => idx !== i) });
  const addDont = () => {
    const v = dontInput.trim();
    if (!v) return;
    setDraft({ ...draft, donts: [...(draft.donts ?? []), v] });
    setDontInput("");
  };
  const removeDont = (i: number) =>
    setDraft({ ...draft, donts: (draft.donts ?? []).filter((_, idx) => idx !== i) });

  const visibleLibrary = useMemo(
    () => (showAllLibrary ? library : library.slice(0, LIBRARY_COLLAPSED_COUNT)),
    [library, showAllLibrary],
  );
  const hiddenCount = Math.max(0, library.length - LIBRARY_COLLAPSED_COUNT);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl w-[92vw] max-h-[95vh] overflow-y-auto sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle>Configure Agent</DialogTitle>
          <DialogDescription>
            Set provider, model, and personality. Save to the shared library to reuse on other pages.
          </DialogDescription>
        </DialogHeader>

        {library.length > 0 && (
          <div className="border rounded-lg overflow-hidden">
            <button
              type="button"
              className="w-full px-4 py-3 text-left text-sm font-medium flex items-center justify-between bg-muted/20 hover:bg-muted/30 transition-colors"
              onClick={() => setLibraryOpen((o) => !o)}
            >
              <span className="flex items-center gap-2">
                <FolderOpen className="h-4 w-4" /> Load from library
                <span className="text-xs text-muted-foreground">({library.length})</span>
              </span>
              <span className="text-xs text-muted-foreground">{libraryOpen ? "▲" : "▼"}</span>
            </button>
            {libraryOpen && (
              <div className="border-t p-3 space-y-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
                  {visibleLibrary.map((a) => {
                    const isPending = pendingDeleteId === a.id;
                    return (
                      <div
                        key={a.id}
                        className="flex items-center gap-2 border rounded-md px-3 py-2 bg-background"
                      >
                        <button
                          className="flex-1 min-w-0 text-left hover:underline"
                          onClick={() => handleLoadFromLibrary(a.id)}
                        >
                          <div className="text-sm font-medium truncate">
                            {a.name || "(unnamed)"}
                          </div>
                          <div className="text-[11px] text-muted-foreground font-mono truncate">
                            {providerLabel(a.providerId)} / {a.model}
                          </div>
                        </button>
                        {isPending ? (
                          <div className="flex items-center gap-1 shrink-0">
                            <span className="text-[11px] text-muted-foreground">Delete?</span>
                            <Button
                              variant="destructive"
                              size="sm"
                              className="h-7 text-xs px-2"
                              onClick={() => {
                                handleDeleteFromLibrary(a.id);
                                setPendingDeleteId(null);
                              }}
                            >
                              Yes
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs px-2"
                              onClick={() => setPendingDeleteId(null)}
                            >
                              No
                            </Button>
                          </div>
                        ) : (
                          <button
                            className="text-muted-foreground hover:text-destructive p-1 shrink-0"
                            onClick={() => setPendingDeleteId(a.id)}
                            title="Delete from library"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
                {hiddenCount > 0 && (
                  <button
                    type="button"
                    onClick={() => setShowAllLibrary((v) => !v)}
                    className="w-full text-xs text-muted-foreground hover:text-foreground py-1 transition-colors"
                  >
                    {showAllLibrary ? "Show less" : `+ Show ${hiddenCount} more`}
                  </button>
                )}
              </div>
            )}
          </div>
        )}

        <Tabs defaultValue="basic" className="gap-4">
          <TabsList variant="line" className="w-full justify-start gap-6 border-b rounded-none px-0">
            <TabsTrigger value="basic" className="px-1">
              <FileText className="h-4 w-4" /> Basic Info
            </TabsTrigger>
            <TabsTrigger value="personality" className="px-1">
              <SlidersHorizontal className="h-4 w-4" /> Personality
            </TabsTrigger>
            <TabsTrigger value="advanced" className="px-1">
              <Cpu className="h-4 w-4" /> Advanced
            </TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="grid grid-cols-2 gap-4">
            <div className="flex items-center gap-3 col-span-2">
              <button
                type="button"
                onClick={() => setAvatarPickerOpen(true)}
                title="Choose avatar"
                className="relative shrink-0 w-28 h-28"
              >
                <div
                  className={`w-full h-full rounded-md overflow-hidden bg-muted/40 hover:bg-muted/60 transition-colors flex items-center justify-center ${
                    typeof draft.avatar === "number"
                      ? "border"
                      : "border border-dashed border-muted-foreground/40"
                  }`}
                >
                  {typeof draft.avatar === "number" ? (
                    <div className="w-full h-full" style={avatarStyle(draft.avatar)} aria-hidden />
                  ) : (
                    <User className="h-12 w-12 text-muted-foreground/60" />
                  )}
                </div>
                <span className="absolute -bottom-1 -right-1 w-7 h-7 rounded-full bg-background border flex items-center justify-center shadow-sm">
                  <Pencil className="h-3.5 w-3.5 text-muted-foreground" />
                </span>
              </button>
              <div className="flex-1 space-y-2">
                <div className="space-y-1">
                  <Label className="text-xs">Agent Name</Label>
                  <Input
                    value={draft.name}
                    onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                    placeholder="e.g. Senior Reviewer"
                  />
                </div>
                <div className="space-y-1">
                  <Label className="text-xs">Role</Label>
                  <Input
                    value={draft.role}
                    onChange={(e) => setDraft({ ...draft, role: e.target.value })}
                    placeholder="e.g. Agent"
                  />
                </div>
              </div>
            </div>

            <div className="space-y-1">
              <Label className="text-xs">AI Provider</Label>
              <Select
                value={draft.providerId}
                onValueChange={(v) => setDraft({ ...draft, providerId: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {enabledProviders.map((p) => (
                    <SelectItem key={p.providerId} value={p.providerId}>
                      {providerLabel(p.providerId)}
                    </SelectItem>
                  ))}
                  {enabledProviders.length === 0 && (
                    <SelectItem value={draft.providerId}>No providers</SelectItem>
                  )}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Label className="text-xs">Model</Label>
              <Input
                value={draft.model}
                onChange={(e) => setDraft({ ...draft, model: e.target.value })}
                placeholder="e.g. gpt-4o"
                className="font-mono text-xs"
              />
            </div>

            <div className="space-y-2 col-span-2">
              <Label className="text-xs">Category</Label>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                {AGENT_CATEGORIES.map((c) => {
                  const selected = draft.category === c;
                  return (
                    <button
                      key={c}
                      type="button"
                      onClick={() => setDraft({ ...draft, category: c })}
                      className={`text-left rounded-lg border p-3 transition-colors ${
                        selected
                          ? "border-primary ring-1 ring-primary/30 bg-primary/5"
                          : "border-border hover:bg-muted/40"
                      }`}
                    >
                      <div className="text-sm font-medium mb-1">{c}</div>
                      <div className="text-xs text-muted-foreground leading-snug line-clamp-3">
                        {CATEGORY_DESCRIPTIONS[c]}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="space-y-1 col-span-2">
              <Label className="text-xs">Main System Prompt</Label>
              <Textarea
                value={draft.goal}
                onChange={(e) => setDraft({ ...draft, goal: e.target.value })}
                placeholder="The agent's main goal — e.g. You will help students with their homework, explaining concepts step-by-step…"
                rows={4}
                className="min-h-[8rem]"
              />
            </div>
          </TabsContent>

          <TabsContent value="personality" className="grid grid-cols-2 gap-4">
            <div className="space-y-2 col-span-2">
              <Label className="text-xs">Personality Style</Label>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
                {PERSONALITY_STYLES.map((s) => {
                  const selected = draft.personalityStyle === s;
                  return (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setDraft({ ...draft, personalityStyle: s })}
                      className={`text-left rounded-lg border p-3 transition-colors ${
                        selected
                          ? "border-primary ring-1 ring-primary/30 bg-primary/5"
                          : "border-border hover:bg-muted/40"
                      }`}
                    >
                      <div className="text-sm font-medium mb-1">{s}</div>
                      <div className="text-xs text-muted-foreground leading-snug line-clamp-3">
                        {PERSONALITY_DESCRIPTIONS[s]}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="space-y-1">
              <Label className="text-xs">Communication Style</Label>
              <Select
                value={draft.communicationStyle}
                onValueChange={(v) => setDraft({ ...draft, communicationStyle: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {COMMUNICATION_STYLES.map((s) => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1">
              <Label className="text-xs">Response Style</Label>
              <Select
                value={draft.responseStyle}
                onValueChange={(v) => setDraft({ ...draft, responseStyle: v })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {RESPONSE_STYLES.map((s) => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1 col-span-2">
              <Label className="text-xs">Personality Description</Label>
              <Textarea
                value={draft.personalityInstruction}
                onChange={(e) => setDraft({ ...draft, personalityInstruction: e.target.value })}
                placeholder="Free-form directives that steer this agent's responses…"
                rows={8}
              />
            </div>
          </TabsContent>

          <TabsContent value="advanced" className="grid grid-cols-1 gap-5">
            <div className="space-y-3">
              <Label className="text-xs">Strict Rules</Label>
              <div className="space-y-2">
                <Label className="text-xs flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                  <Check className="h-3.5 w-3.5" /> Do&apos;s (behaviors to encourage)
                </Label>
                {(draft.dos ?? []).map((r, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded-md border border-emerald-200 bg-emerald-50 dark:border-emerald-900/50 dark:bg-emerald-950/20 px-3 py-2 text-sm"
                  >
                    <Check className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                    <span className="flex-1 min-w-0 break-words">{r}</span>
                    <button
                      type="button"
                      onClick={() => removeDo(i)}
                      className="shrink-0 text-muted-foreground hover:text-destructive"
                      title="Remove"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
                <div className="flex gap-2">
                  <Input
                    value={doInput}
                    onChange={(e) => setDoInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        addDo();
                      }
                    }}
                    placeholder="Add a behavior to encourage…"
                  />
                  <Button type="button" variant="outline" onClick={addDo}>
                    Add
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="text-xs flex items-center gap-1.5 text-red-600 dark:text-red-400">
                  <Ban className="h-3.5 w-3.5" /> Don&apos;ts (behaviors to avoid)
                </Label>
                {(draft.donts ?? []).map((r, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded-md border border-red-200 bg-red-50 dark:border-red-900/50 dark:bg-red-950/20 px-3 py-2 text-sm"
                  >
                    <Ban className="h-3.5 w-3.5 shrink-0 text-red-600 dark:text-red-400" />
                    <span className="flex-1 min-w-0 break-words">{r}</span>
                    <button
                      type="button"
                      onClick={() => removeDont(i)}
                      className="shrink-0 text-muted-foreground hover:text-destructive"
                      title="Remove"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ))}
                <div className="flex gap-2">
                  <Input
                    value={dontInput}
                    onChange={(e) => setDontInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        addDont();
                      }
                    }}
                    placeholder="Add a behavior to avoid…"
                  />
                  <Button type="button" variant="outline" onClick={addDont}>
                    Add
                  </Button>
                </div>
              </div>
            </div>

            <div className="space-y-1">
              <Label className="text-xs">Max Response Length (tokens)</Label>
              <Input
                type="number"
                min={1}
                value={draft.maxTokens ?? ""}
                onChange={(e) => {
                  const n = e.target.value === "" ? null : Math.max(1, Number(e.target.value));
                  setDraft({ ...draft, maxTokens: Number.isFinite(n as number) ? n : null });
                }}
                placeholder="Leave empty to use the default"
              />
              <p className="text-[11px] text-muted-foreground">
                Approximate: 100 tokens ≈ 75 words. Leave empty to use the default.
              </p>
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Additional Knowledge Context</Label>
                {draft.knowledgeContext && (
                  <span className="text-[10px] text-muted-foreground">
                    {draft.knowledgeContext.length.toLocaleString()} chars
                  </span>
                )}
              </div>
              <Textarea
                value={draft.knowledgeContext}
                onChange={(e) => setDraft({ ...draft, knowledgeContext: e.target.value })}
                placeholder="Paste reference material the agent should treat as background knowledge…"
                rows={10}
              />
            </div>
          </TabsContent>
        </Tabs>

        <div className="border-t border-border/60 pt-4 mt-2" />

        <div className="border rounded-lg overflow-hidden">
          <button
            type="button"
            className="w-full px-4 py-3 text-left text-sm font-medium flex items-center justify-between bg-muted/20 hover:bg-muted/30 transition-colors"
            onClick={() => setSystemPromptOpen((o) => !o)}
          >
            <span>System Prompt — composed from all fields above</span>
            <span className="text-xs text-muted-foreground">{systemPromptOpen ? "▲" : "▼"}</span>
          </button>
          {systemPromptOpen && (
            <div className="border-t p-4">
              <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap break-words p-3 rounded-md bg-slate-50 dark:bg-slate-900/50 max-h-80 overflow-y-auto">
                {buildAgentSystemPrefix(draft) || "(empty — fill in the fields above)"}
              </pre>
            </div>
          )}
        </div>

        <Dialog open={avatarPickerOpen} onOpenChange={setAvatarPickerOpen}>
          <DialogContent className="sm:max-w-lg">
            <DialogHeader>
              <DialogTitle>Choose Avatar</DialogTitle>
              <DialogDescription>
                Pick a character to represent this agent.
              </DialogDescription>
            </DialogHeader>
            <div className="grid grid-cols-4 gap-3 py-2">
              {AGENT_AVATAR_INDICES.map((sheetIdx) => {
                const isSelected = draft.avatar === sheetIdx;
                return (
                  <button
                    key={sheetIdx}
                    type="button"
                    onClick={() => {
                      setDraft({ ...draft, avatar: sheetIdx });
                      setAvatarPickerOpen(false);
                    }}
                    className={`aspect-square rounded-md border overflow-hidden hover:opacity-80 transition-all ${
                      isSelected ? "border-primary ring-2 ring-primary/30" : "border-border"
                    }`}
                    style={avatarStyle(sheetIdx)}
                    title={`Avatar #${sheetIdx + 1}`}
                  />
                );
              })}
            </div>
            {typeof draft.avatar === "number" && (
              <DialogFooter>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setDraft({ ...draft, avatar: undefined });
                    setAvatarPickerOpen(false);
                  }}
                >
                  Clear avatar
                </Button>
              </DialogFooter>
            )}
          </DialogContent>
        </Dialog>

        <DialogFooter className="gap-2">
          {sourceLibraryId && (
            <Button variant="outline" onClick={handleUpdateInLibrary} className="gap-2">
              <Save className="h-3.5 w-3.5" /> Save
            </Button>
          )}
          <Button variant="outline" onClick={handleSaveAsNew} className="gap-2">
            <Save className="h-3.5 w-3.5" /> Save as…
          </Button>
          <div className="flex-1" />
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSaveAndClose}>Apply to Card</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
