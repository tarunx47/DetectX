// SAMPLE DATA — shown only while the backend is offline, always labeled "Demo data" in the UI.
import type { Stats, TransactionRecord, TxType } from "./api";

const types: TxType[] = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"];

function seeded(i: number) {
  const x = Math.sin(i * 9301 + 49297) * 233280;
  return x - Math.floor(x);
}

export const demoTransactions: TransactionRecord[] = Array.from({ length: 40 }, (_, i) => {
  const type = types[Math.floor(seeded(i) * types.length)] as TxType;
  const amount = Math.round(seeded(i + 100) * 250000 * 100) / 100;
  const old = Math.round(seeded(i + 200) * 400000);
  const isFraud = (type === "TRANSFER" || type === "CASH_OUT") && seeded(i + 300) > 0.7;
  const p = isFraud ? 0.7 + seeded(i + 400) * 0.29 : seeded(i + 400) * 0.35;
  return {
    id: `DEMO-${String(1040 - i).padStart(5, "0")}`,
    step: 1 + Math.floor(seeded(i + 500) * 740),
    type,
    amount,
    oldbalanceOrg: old,
    newbalanceOrig: isFraud ? 0 : Math.max(0, old - amount),
    oldbalanceDest: Math.round(seeded(i + 600) * 300000),
    newbalanceDest: Math.round(seeded(i + 700) * 300000),
    prediction: isFraud ? 1 : 0,
    probability: Math.round(p * 1000) / 1000,
    timestamp: new Date(Date.UTC(2026, 9, 9, 20) - i * 3600_000 * 3.7).toISOString(),
  };
});

export const demoStats: Stats = {
  total: 1240,
  fraud: 37,
  legitimate: 1203,
  daily: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((d, i) => ({
    date: d,
    total: 140 + Math.round(seeded(i + 900) * 80),
    fraud: 2 + Math.round(seeded(i + 950) * 8),
  })),
};
