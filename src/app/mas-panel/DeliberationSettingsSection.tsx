"use client";

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { type DeliberationSettings } from "./workflow-types";

interface Props {
  value: DeliberationSettings;
  onChange: (v: DeliberationSettings) => void;
}

export function DeliberationSettingsSection({ value, onChange }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="space-y-1">
        <Label className="text-xs">Max Rounds</Label>
        <Input
          type="number"
          min={1}
          max={10}
          value={value.maxRounds}
          onChange={(e) => onChange({ ...value, maxRounds: Math.max(1, Math.min(10, parseInt(e.target.value) || 1)) })}
          className="h-9"
        />
        <p className="text-[11px] text-muted-foreground">
          Max number of deliberation rounds. Stops earlier if converged.
        </p>
      </div>

      <div className="space-y-1">
        <Label className="text-xs">Convergence Mode</Label>
        <Select
          value={value.convergenceMode}
          onValueChange={(v) => onChange({ ...value, convergenceMode: v as "fixed" | "adaptive" })}
        >
          <SelectTrigger className="h-9">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="fixed">Fixed — run all rounds</SelectItem>
            <SelectItem value="adaptive">Adaptive — stop when settled</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-[11px] text-muted-foreground">
          Fixed runs every round. Adaptive stops when agents&apos; answers stabilize.
        </p>
      </div>

      <div className="space-y-1">
        <Label className="text-xs">Convergence Threshold</Label>
        <Input
          type="number"
          min={0}
          max={50}
          value={value.convergenceThreshold}
          onChange={(e) => onChange({ ...value, convergenceThreshold: Math.max(0, Math.min(50, parseInt(e.target.value) || 0)) })}
          disabled={value.convergenceMode !== "adaptive"}
          className="h-9"
        />
        <p className="text-[11px] text-muted-foreground">
          Used only in adaptive mode. Lower = stricter (0-50).
        </p>
      </div>
    </div>
  );
}
