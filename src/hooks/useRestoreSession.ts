"use client";

import { useEffect, useRef, useState } from "react";
import { useRestoreStore, type RestorePayload } from "@/lib/restore-store";

/**
 * Hook for tool pages to consume a pending session restore.
 *
 * Returns the restore payload if one is pending for this tool's runType,
 * or null otherwise. Payload is consumed on first mount — subsequent
 * renders return null.
 *
 * Usage:
 *   const restored = useRestoreSession("transform");
 *   useEffect(() => {
 *     if (!restored) return;
 *     setData(restored.data);
 *     setSystemPrompt(restored.systemPrompt);
 *     // ...populate other state
 *   }, [restored]);
 */
export function useRestoreSession(runType: string): RestorePayload | null {
  const [payload, setPayload] = useState<RestorePayload | null>(null);
  const consumed = useRef(false);

  useEffect(() => {
    if (consumed.current) return;
    consumed.current = true;

    const pending = useRestoreStore.getState().consume();
    if (pending && pending.runType === runType) {
      queueMicrotask(() => setPayload(pending));
    }
  }, [runType]);

  return payload;
}
