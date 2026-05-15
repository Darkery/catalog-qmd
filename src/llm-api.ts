import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  RerankOptions,
  RerankResult,
  RerankDocument,
  RerankDocumentResult,
  ModelInfo,
  Queryable,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

export type ApiProvider = "openai" | "anthropic" | "gemini" | "custom";

export type ApiLLMConfig = {
  provider?: ApiProvider;

  baseUrl?: string;
  embedEndpoint?: string;
  chatEndpoint?: string;
  rerankEndpoint?: string;

  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;

  apiKey?: string;
  apiKeyEnv?: string;

  embedDimensions?: number;
  maxRetries?: number;
  timeoutMs?: number;
};

// =============================================================================
// Defaults
// =============================================================================

const DEFAULTS = {
  provider: "openai" as ApiProvider,
  baseUrl: "https://api.openai.com/v1",
  embedEndpoint: "/embeddings",
  chatEndpoint: "/chat/completions",
  rerankEndpoint: "/rerank",
  embedModel: "text-embedding-3-small",
  generateModel: "gpt-4.1-mini",
  rerankModel: "bge-reranker-v2-m3",
  apiKeyEnv: "QMD_API_KEY",
  maxRetries: 3,
  timeoutMs: 30000,
};

// =============================================================================
// ApiLLM Implementation
// =============================================================================

export class ApiLLM implements LLM {
  private config: typeof DEFAULTS & ApiLLMConfig;
  private apiKey: string;

  constructor(config: ApiLLMConfig = {}) {
    this.config = { ...DEFAULTS, ...config };
    this.apiKey = config.apiKey
      || process.env[this.config.apiKeyEnv ?? DEFAULTS.apiKeyEnv]
      || process.env.OPENAI_API_KEY
      || "";
  }

  // ─── Embedding ──────────────────────────────────────────────────────────────

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    const model = options?.model || this.config.embedModel!;
    const url = `${this.config.baseUrl}${this.config.embedEndpoint}`;

    const body: Record<string, unknown> = { model, input: text };
    if (this.config.embedDimensions) {
      body.dimensions = this.config.embedDimensions;
    }

    const resp = await this.request(url, body);
    if (!resp) return null;

    return { embedding: resp.data[0].embedding, model };
  }

  async embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const model = options?.model || this.config.embedModel!;
    const url = `${this.config.baseUrl}${this.config.embedEndpoint}`;

    const body: Record<string, unknown> = { model, input: texts };
    if (this.config.embedDimensions) {
      body.dimensions = this.config.embedDimensions;
    }

    const resp = await this.request(url, body);
    if (!resp) return texts.map(() => null);

    return resp.data.map((item: { embedding: number[] }) => ({
      embedding: item.embedding,
      model,
    }));
  }

  // ─── Generation ─────────────────────────────────────────────────────────────

  async generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> {
    const model = options?.model || this.config.generateModel!;
    const url = `${this.config.baseUrl}${this.config.chatEndpoint}`;

    const body = {
      model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: options?.maxTokens ?? 150,
      temperature: options?.temperature ?? 0.7,
    };

    const resp = await this.request(url, body);
    if (!resp) return null;

    return {
      text: resp.choices[0].message.content,
      model,
      done: true,
    };
  }

  // ─── Query Expansion ────────────────────────────────────────────────────────

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    const prompt = `You are a search query expansion system.
Given this search query, generate alternative search queries.
Output format: one per line, prefixed with type.
Types: lex (keyword search), vec (semantic search), hyde (hypothetical document)

Query: ${query}
${options?.context ? `Context: ${options.context}` : ""}

Generate 3-5 expanded queries:`;

    const result = await this.generate(prompt, { maxTokens: 300, temperature: 0.7 });
    if (!result) return [{ type: "vec", text: query }];

    const lines = result.text.trim().split("\n");
    const queries: Queryable[] = [];

    for (const line of lines) {
      const match = line.match(/^(lex|vec|hyde):\s*(.+)/);
      if (match && match[1] && match[2]) {
        queries.push({ type: match[1] as "lex" | "vec" | "hyde", text: match[2].trim() });
      }
    }

    if (queries.length === 0) {
      queries.push({ type: "vec", text: query });
    }
    return queries;
  }

  // ─── Reranking ──────────────────────────────────────────────────────────────

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    const model = options?.model || this.config.rerankModel!;

    const rerankResult = await this.tryRerankApi(query, documents, model);
    if (rerankResult) return rerankResult;

    return this.rerankViaChatCompletion(query, documents, model);
  }

  private async tryRerankApi(
    query: string,
    documents: RerankDocument[],
    model: string
  ): Promise<RerankResult | null> {
    const url = `${this.config.baseUrl}${this.config.rerankEndpoint}`;

    const body = {
      model,
      query,
      documents: documents.map(d => d.text),
      top_n: documents.length,
    };

    try {
      const resp = await this.request(url, body);
      if (!resp || !resp.results) return null;

      const results: RerankDocumentResult[] = resp.results.map(
        (r: { index: number; relevance_score: number }) => ({
          file: documents[r.index]?.file ?? "",
          score: r.relevance_score,
          index: r.index,
        })
      );

      results.sort((a, b) => b.score - a.score);
      return { results, model };
    } catch {
      return null;
    }
  }

  private async rerankViaChatCompletion(
    query: string,
    documents: RerankDocument[],
    model: string
  ): Promise<RerankResult> {
    const prompt = `Rate the relevance of each document to the query on a scale of 0-10.
Query: "${query}"

${documents.map((d, i) => `[${i}] ${d.text.slice(0, 200)}`).join("\n")}

Output JSON array of scores: [score0, score1, ...]`;

    const result = await this.generate(prompt, { maxTokens: 100, temperature: 0 });
    const scores: number[] = [];

    if (result) {
      try {
        const match = result.text.match(/\[[\d,.\s]+\]/);
        if (match) {
          scores.push(...JSON.parse(match[0]));
        }
      } catch { /* use empty scores */ }
    }

    const results: RerankDocumentResult[] = documents.map((doc, i) => ({
      file: doc.file,
      score: (scores[i] ?? 0) / 10,
      index: i,
    }));
    results.sort((a, b) => b.score - a.score);

    return { results, model };
  }

  // ─── Properties (compatible with LlamaCpp usage in store.ts) ─────────────────

  get embedModelName(): string {
    return this.config.embedModel!;
  }

  // ─── Tokenization (approximate, for chunking fallback) ──────────────────────

  async tokenize(text: string): Promise<number[]> {
    const approxTokens = Math.ceil(text.length / 4);
    return Array.from({ length: approxTokens }, (_, i) => i);
  }

  async detokenize(tokens: number[]): Promise<string> {
    return "";
  }

  // ─── Utility ────────────────────────────────────────────────────────────────

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async dispose(): Promise<void> {}

  // ─── HTTP Helper ────────────────────────────────────────────────────────────

  private async request(url: string, body: unknown): Promise<any | null> {
    for (let attempt = 0; attempt < this.config.maxRetries!; attempt++) {
      try {
        const resp = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(this.config.timeoutMs!),
        });

        if (!resp.ok) {
          if (resp.status === 429) {
            await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
            continue;
          }
          const errText = await resp.text();
          process.stderr.write(`[ApiLLM] HTTP ${resp.status}: ${errText.slice(0, 200)}\n`);
          return null;
        }

        return await resp.json();
      } catch (err) {
        if (attempt === this.config.maxRetries! - 1) {
          process.stderr.write(`[ApiLLM] Request failed: ${err}\n`);
          return null;
        }
        await new Promise(r => setTimeout(r, 500 * (attempt + 1)));
      }
    }
    return null;
  }
}
