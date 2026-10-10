// DetectX API service — the single place that talks to the FastAPI backend.
// Base URL: VITE_API_BASE_URL at build time, overridable at runtime in the header.

export type TxType = "PAYMENT" | "TRANSFER" | "CASH_OUT" | "CASH_IN" | "DEBIT";
export const TX_TYPES: TxType[] = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"];

export interface TransactionInput {
  step: number;
  type: TxType;
  amount: number;
  oldbalanceOrg: number;
  newbalanceOrig: number;
  oldbalanceDest: number;
  newbalanceDest: number;
}

export interface PredictionResult {
  id?: string;
  prediction: number; // 1 = fraud, 0 = legitimate
  label?: string;
  probability?: number | null; // 0..1 if model provides it
  risk_level?: string;
}

export interface TransactionRecord extends TransactionInput {
  id: string;
  prediction: number;
  probability?: number | null;
  timestamp: string;
  label?: string;
  risk_level?: string;
}

export interface Stats {
  total: number;
  fraud: number;
  legitimate: number;
  daily?: { date: string; total: number; fraud: number }[];
}

export interface DatasetTypeStat {
  transactions: number;
  actual_fraud: number;
  predicted_fraud: number;
}

export interface DatasetStats {
  total_transactions: number;
  actual_fraud: number;
  actual_legitimate: number;
  predicted_fraud: number;
  predicted_legitimate: number;
  true_positive: number;
  false_positive: number;
  true_negative: number;
  false_negative: number;
  by_type: Record<string, DatasetTypeStat>;
  actual_fraud_rate: number;
  predicted_fraud_rate: number;
  evaluation: {
    precision: number;
    recall: number;
    accuracy: number;
  };
}

const STORAGE_KEY = "detectx_api_base_url";

export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) return saved;
  }
  return (import.meta.env["VITE_API_BASE_URL"] as string | undefined) ?? "http://localhost:8000";
}

export function setApiBaseUrl(url: string) {
  window.localStorage.setItem(STORAGE_KEY, url.replace(/\/+$/, ""));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${getApiBaseUrl()}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () =>
    request<{ status: string; model_loaded?: boolean }>("/health", {
      signal: AbortSignal.timeout(4000),
    }),

  predict: async (tx: TransactionInput): Promise<PredictionResult> => {
    const raw = await request<PredictionResult & { fraud_probability?: number | null }>(
      "/predict",
      {
        method: "POST",
        body: JSON.stringify(tx),
      },
    );
    return {
      ...raw,
      probability: raw.probability ?? raw.fraud_probability ?? null,
      label: raw.label ?? (raw.prediction === 1 ? "Fraudulent" : "Legitimate"),
    };
  },

  transactions: async (): Promise<TransactionRecord[]> => {
    const raw = await request<TransactionRecord[] | { transactions: TransactionRecord[] }>(
      "/transactions",
    );
    const items = Array.isArray(raw) ? raw : (raw.transactions ?? []);
    return items.map((t) => ({
      ...t,
      id: String(t.id),
      probability:
        t.probability ??
        (t as unknown as { fraud_probability?: number | null }).fraud_probability ??
        null,
    }));
  },

  stats: async (): Promise<Stats> => {
    const raw = await request<
      Stats & {
        total_transactions?: number;
        fraudulent_transactions?: number;
        legitimate_transactions?: number;
      }
    >("/stats");
    return {
      total: raw.total ?? raw.total_transactions ?? 0,
      fraud: raw.fraud ?? raw.fraudulent_transactions ?? 0,
      legitimate: raw.legitimate ?? raw.legitimate_transactions ?? 0,
      daily: raw.daily ?? [],
    };
  },

  datasetStats: () => request<DatasetStats>("/dataset-stats"),
};

export function riskLevel(
  p: number | null | undefined,
  prediction: number,
): "High" | "Medium" | "Low" {
  if (p == null) return prediction === 1 ? "High" : "Low";
  if (p >= 0.7) return "High";
  if (p >= 0.3) return "Medium";
  return "Low";
}
