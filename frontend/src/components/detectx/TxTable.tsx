import type { TransactionRecord } from "@/lib/api";
import { riskLevel } from "@/lib/api";
import { PredictionPill, RiskPill, fmtMoney } from "./ui-bits";

export function TxTable({ rows, compact }: { rows: TransactionRecord[]; compact?: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted-foreground">
            <th className="py-3 pr-4 font-medium">ID</th>
            <th className="py-3 pr-4 font-medium">Type</th>
            <th className="py-3 pr-4 text-right font-medium">Amount</th>
            {!compact && (
              <th className="py-3 pr-4 text-right font-medium">Origin bal. (before → after)</th>
            )}
            <th className="py-3 pr-4 font-medium">Prediction</th>
            <th className="py-3 pr-4 font-medium">Risk</th>
            <th className="py-3 font-medium">Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b border-border/50 hover:bg-muted/30">
              <td className="py-3 pr-4 font-mono text-xs">{r.id}</td>
              <td className="py-3 pr-4 font-mono text-xs">{r.type}</td>
              <td className="py-3 pr-4 text-right font-mono">{fmtMoney(r.amount)}</td>
              {!compact && (
                <td className="py-3 pr-4 text-right font-mono text-xs text-muted-foreground">
                  {fmtMoney(r.oldbalanceOrg)} → {fmtMoney(r.newbalanceOrig)}
                </td>
              )}
              <td className="py-3 pr-4">
                <PredictionPill prediction={r.prediction} />
              </td>
              <td className="py-3 pr-4">
                <RiskPill level={riskLevel(r.probability, r.prediction)} />
              </td>
              <td className="py-3 text-xs text-muted-foreground">
                {new Date(r.timestamp).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
