/**
 * Global store for session restoration from history.
 *
 * Flow:
 * 1. History detail page calls setPending() with run data
 * 2. Navigates to the tool page (e.g. /transform)
 * 3. Tool page calls consume() on mount to get the payload
 * 4. Tool page uses the payload to populate its state
 */

import { create } from "zustand";

type Row = Record<string, unknown>;

export interface RestorePayload {
  runId: string;
  runType: string;
  /** Original input rows reconstructed from inputJson */
  data: Row[];
  dataName: string;
  systemPrompt: string;
  /** Merged rows: original input + output + status + latency */
  results: Row[];
  provider: string;
  model: string;
  temperature: number;
}

interface RestoreState {
  pending: RestorePayload | null;
  setPending: (payload: RestorePayload) => void;
  consume: () => RestorePayload | null;
}

export const useRestoreStore = create<RestoreState>((set, get) => ({
  pending: null,

  setPending: (payload) => set({ pending: payload }),

  /** Returns the pending payload and clears it (one-time consumption). */
  consume: () => {
    const current = get().pending;
    if (current) set({ pending: null });
    return current;
  },
}));
