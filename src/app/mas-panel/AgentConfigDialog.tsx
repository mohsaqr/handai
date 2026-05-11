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
import { Save, Trash2, FolderOpen, Pencil, User } from "lucide-react";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  agent: Agent;
  onSave: (agent: Agent) => void;
  enabledProviders: { providerId: string; defaultModel: string }[];
}

const LIBRARY_COLLAPSED_COUNT = 12;

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

        <div className="grid grid-cols-2 gap-4">
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
            <div className="flex-1 space-y-1">
              <Label className="text-xs">Agent Name</Label>
              <Input
                value={draft.name}
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                placeholder="e.g. Senior Reviewer"
              />
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

          <div className="space-y-1">
            <Label className="text-xs">Category</Label>
            <Select
              value={draft.category}
              onValueChange={(v) => setDraft({ ...draft, category: v })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {AGENT_CATEGORIES.map((c) => (
                  <SelectItem key={c} value={c}>{c}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1">
            <Label className="text-xs">Personality Style</Label>
            <Select
              value={draft.personalityStyle}
              onValueChange={(v) => setDraft({ ...draft, personalityStyle: v })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {PERSONALITY_STYLES.map((s) => (
                  <SelectItem key={s} value={s}>{s}</SelectItem>
                ))}
              </SelectContent>
            </Select>
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

          <div className="space-y-1 col-span-2">
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
              rows={8}
            />
          </div>
        </div>

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
