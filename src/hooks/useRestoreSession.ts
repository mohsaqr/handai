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
 * Pass an array of runType strings to also accept aliases (e.g. an old slug
 * still present in historical DB rows after the tool was renamed).
 *
 * Usage:
 *   const restored = useRestoreSession("transform");
 *   const restored = useRestoreSession(["model-comparison", "consensus-coder"]);
 */
export function useRestoreSession(runType: string | string[]): RestorePayload | null {
  const [payload, setPayload] = useState<RestorePayload | null>(null);
  const consumed = useRef(false);
  const accepted = Array.isArray(runType) ? runType : [runType];
  const acceptedKey = accepted.join("|");

  useEffect(() => {
    if (consumed.current) return;
    consumed.current = true;

    const pending = useRestoreStore.getState().consume();
    if (pending && accepted.includes(pending.runType)) {
      queueMicrotask(() => setPayload(pending));
    }
    // accepted is derived from acceptedKey; using the key keeps the dep array stable.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [acceptedKey]);

  return payload;
}
