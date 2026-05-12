import { useMemo, useRef } from "react";
import type { FileState } from "@/types";

export const fileKey = (f: File) => `${f.name}__${f.size}`;

export type FileResult = {
  status?: string;
  error_msg?: string;
  [k: string]: unknown;
};

/** Map File objects to FileStatus strings by combining the stored status on
 * each `FileState` with results emitted by `useBatchProcessor`. */
export function useFileStatuses(fileStates: FileState[], results: FileResult[]) {
  return useMemo(() => {
    if (results.length === 0) return fileStates.map((fs) => fs.status);
    return fileStates.map((_, i) => {
      const r = results[i];
      if (!r) return "pending" as const;
      if (r.status === "error") return "error" as const;
      if (r.status === "skipped") return "pending" as const;
      if (r.status === "success") return "done" as const;
      return "pending" as const;
    });
  }, [fileStates, results]);
}

// Module-level registry — survives component unmount (but not page reload).
// Keyed by tool id so tools don't share their uploaded files.
const moduleFileMaps = new Map<string, Map<string, File>>();

function getOrCreateMap(key: string): Map<string, File> {
  let map = moduleFileMaps.get(key);
  if (!map) {
    map = new Map<string, File>();
    moduleFileMaps.set(key, map);
  }
  return map;
}

/** Holds File objects in a module-level Map keyed by `fileKey(file)`.
 * File objects can't be serialized, so anything relying on them must coordinate
 * via this ref. Pass a tool-unique `key` to keep each tool's files isolated. */
export function useFilesRef(key: string = "default") {
  const ref = useRef<Map<string, File> | null>(null);
  if (ref.current === null) {
    ref.current = getOrCreateMap(key);
  }
  return ref as { current: Map<string, File> };
}
