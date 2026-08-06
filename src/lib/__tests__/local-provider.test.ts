import { describe, it, expect, vi, afterEach } from "vitest";
import {
  isLoopbackUrl,
  isLocalProvider,
  ollamaTagsUrl,
  lmStudioModelsUrl,
  probeLocalModels,
} from "../local-provider";

describe("isLoopbackUrl", () => {
  it("recognises loopback hosts", () => {
    expect(isLoopbackUrl("http://localhost:1234/v1")).toBe(true);
    expect(isLoopbackUrl("http://127.0.0.1:11434/v1")).toBe(true);
    expect(isLoopbackUrl("http://[::1]:1234/v1")).toBe(true);
    expect(isLoopbackUrl("http://0.0.0.0:8080/v1")).toBe(true);
    expect(isLoopbackUrl("http://my-mac.local:1234/v1")).toBe(true);
  });

  it("rejects remote hosts and junk", () => {
    expect(isLoopbackUrl("https://api.openai.com/v1")).toBe(false);
    expect(isLoopbackUrl("https://handai.lacarm.com/v1")).toBe(false);
    expect(isLoopbackUrl("not a url")).toBe(false);
    expect(isLoopbackUrl(undefined)).toBe(false);
  });
});

describe("isLocalProvider", () => {
  it("treats ollama and lmstudio as local regardless of base URL", () => {
    expect(isLocalProvider("ollama")).toBe(true);
    expect(isLocalProvider("lmstudio")).toBe(true);
  });

  it("treats custom as local only when it points at loopback", () => {
    expect(isLocalProvider("custom", "http://localhost:8080/v1")).toBe(true);
    expect(isLocalProvider("custom", "https://gateway.example.com/v1")).toBe(false);
    expect(isLocalProvider("custom")).toBe(false);
  });

  it("treats cloud providers as remote", () => {
    expect(isLocalProvider("openai")).toBe(false);
    expect(isLocalProvider("anthropic")).toBe(false);
    expect(isLocalProvider(undefined)).toBe(false);
  });
});

describe("endpoint builders", () => {
  it("puts ollama tags at the server root, not under /v1", () => {
    expect(ollamaTagsUrl("http://localhost:11434/v1")).toBe(
      "http://localhost:11434/api/tags"
    );
    expect(ollamaTagsUrl(undefined)).toBe("http://localhost:11434/api/tags");
    expect(ollamaTagsUrl("garbage")).toBe("http://localhost:11434/api/tags");
  });

  it("honors a custom port and trims trailing slashes for lmstudio", () => {
    expect(lmStudioModelsUrl("http://localhost:4321/v1")).toBe(
      "http://localhost:4321/v1/models"
    );
    expect(lmStudioModelsUrl("http://localhost:1234/v1/")).toBe(
      "http://localhost:1234/v1/models"
    );
    expect(lmStudioModelsUrl(undefined)).toBe("http://localhost:1234/v1/models");
  });
});

describe("probeLocalModels", () => {
  afterEach(() => vi.unstubAllGlobals());

  const jsonResponse = (body: unknown) =>
    ({ ok: true, json: async () => body }) as Response;

  it("maps both servers' payloads into model id lists", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.includes("11434")
          ? jsonResponse({ models: [{ name: "llama3" }, { name: "mistral" }] })
          : jsonResponse({ data: [{ id: "liquid/lfm2.5-1.2b" }] })
      )
    );

    expect(await probeLocalModels()).toEqual({
      ollama: ["llama3", "mistral"],
      lmstudio: ["liquid/lfm2.5-1.2b"],
    });
  });

  it("omits a server that refuses the connection, keeping the other", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.includes("11434")) throw new TypeError("Failed to fetch");
        return jsonResponse({ data: [{ id: "qwen/qwen3-4b-thinking-2507" }] });
      })
    );

    expect(await probeLocalModels()).toEqual({
      lmstudio: ["qwen/qwen3-4b-thinking-2507"],
    });
  });

  it("returns {} when nothing is reachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      })
    );

    expect(await probeLocalModels()).toEqual({});
  });

  it("ignores non-ok responses", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, json: async () => ({}) }) as Response)
    );

    expect(await probeLocalModels()).toEqual({});
  });

  it("probes the configured base URLs", async () => {
    const spy = vi.fn(
      async (url: string) => ({ ok: false, json: async () => ({ url }) }) as Response
    );
    vi.stubGlobal("fetch", spy);

    await probeLocalModels({
      ollama: "http://127.0.0.1:9999/v1",
      lmstudio: "http://127.0.0.1:4321/v1",
    });

    const called = spy.mock.calls.map((c) => c[0]);
    expect(called).toContain("http://127.0.0.1:9999/api/tags");
    expect(called).toContain("http://127.0.0.1:4321/v1/models");
  });
});
