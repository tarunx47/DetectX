import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, ScanSearch, ShieldAlert, ShieldCheck, WifiOff } from "lucide-react";
import { AppShell } from "@/components/detectx/AppShell";
import { ErrorState, PageHeader, Panel, RiskPill } from "@/components/detectx/ui-bits";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api, riskLevel, TX_TYPES, type TransactionInput, type TxType } from "@/lib/api";
import { useBackendStatus } from "@/hooks/use-backend";

export const Route = createFileRoute("/analyze")({
  head: () => ({
    meta: [
      { title: "Analyze Transaction — DetectX" },
      {
        name: "description",
        content: "Submit a financial transaction to the DetectX model and get a fraud prediction.",
      },
      { property: "og:title", content: "Analyze Transaction — DetectX" },
      {
        property: "og:description",
        content: "Get an ML-powered fraud prediction for any transaction.",
      },
    ],
  }),
  component: Analyze,
});

const numFields: { key: Exclude<keyof TransactionInput, "type">; label: string; hint: string }[] = [
  { key: "step", label: "Step", hint: "Hour of simulation (1 step = 1 hour)" },
  { key: "amount", label: "Amount", hint: "Transaction amount" },
  { key: "oldbalanceOrg", label: "Origin balance before", hint: "Sender balance before" },
  { key: "newbalanceOrig", label: "Origin balance after", hint: "Sender balance after" },
  { key: "oldbalanceDest", label: "Destination balance before", hint: "Recipient balance before" },
  { key: "newbalanceDest", label: "Destination balance after", hint: "Recipient balance after" },
];

function Analyze() {
  const health = useBackendStatus();
  const qc = useQueryClient();
  const [type, setType] = useState<TxType>("TRANSFER");
  const [values, setValues] = useState<Record<string, string>>({
    step: "1",
    amount: "181.00",
    oldbalanceOrg: "181.00",
    newbalanceOrig: "0.00",
    oldbalanceDest: "0.00",
    newbalanceDest: "0.00",
  });
  const [validationError, setValidationError] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: api.predict,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["transactions"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    setValidationError(null);

    // Validate 7 required features
    const stepVal = parseInt(values.step ?? "", 10);
    if (isNaN(stepVal) || stepVal < 1) {
      setValidationError("Step must be a valid positive integer (hour >= 1).");
      return;
    }

    const tx: TransactionInput = {
      step: stepVal,
      type,
      amount: Number(values.amount ?? 0),
      oldbalanceOrg: Number(values.oldbalanceOrg ?? 0),
      newbalanceOrig: Number(values.newbalanceOrig ?? 0),
      oldbalanceDest: Number(values.oldbalanceDest ?? 0),
      newbalanceDest: Number(values.newbalanceDest ?? 0),
    };

    for (const f of numFields) {
      if (f.key !== "step") {
        const val = tx[f.key];
        if (isNaN(val) || val < 0) {
          setValidationError(`${f.label} must be a valid non-negative number.`);
          return;
        }
      }
    }

    mut.mutate(tx);
  };

  const r = mut.data;
  const prob = r?.probability;

  return (
    <AppShell>
      <PageHeader
        title="Transaction Analysis"
        subtitle="Enter transaction details to get a real-time ML fraud prediction from DetectX."
      />
      <div className="grid gap-6 lg:grid-cols-5">
        <Panel className="lg:col-span-3">
          <form onSubmit={submit} className="grid gap-5 sm:grid-cols-2">
            <div className="space-y-2 sm:col-span-2">
              <Label>Transaction type</Label>
              <Select value={type} onValueChange={(v) => setType(v as TxType)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TX_TYPES.map((t) => (
                    <SelectItem key={t} value={t}>
                      {t}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {numFields.map((f) => (
              <div key={f.key} className="space-y-2">
                <Label htmlFor={f.key}>{f.label}</Label>
                <Input
                  id={f.key}
                  type="number"
                  step="any"
                  min={f.key === "step" ? 1 : 0}
                  required
                  className="font-mono"
                  value={values[f.key] ?? ""}
                  onChange={(e) => {
                    setValues({ ...values, [f.key]: e.target.value });
                    if (validationError) setValidationError(null);
                  }}
                  placeholder="0"
                />
                <p className="text-xs text-muted-foreground">{f.hint}</p>
              </div>
            ))}

            {validationError && (
              <div className="sm:col-span-2">
                <ErrorState message={validationError} />
              </div>
            )}

            <div className="sm:col-span-2">
              <Button
                type="submit"
                size="lg"
                className="w-full"
                disabled={mut.isPending || !health.isSuccess}
              >
                {mut.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <ScanSearch className="h-4 w-4" />
                )}
                Analyze Transaction
              </Button>
              {!health.isSuccess && !health.isLoading && (
                <p className="mt-2 flex items-center gap-1.5 text-xs text-warning">
                  <WifiOff className="h-3.5 w-3.5" /> Connect the FastAPI backend (top-right) to run
                  real predictions.
                </p>
              )}
            </div>
          </form>
        </Panel>

        <Panel className="lg:col-span-2">
          <h2 className="mb-4 text-sm font-medium">Model Prediction Result</h2>
          {mut.isPending ? (
            <div className="flex flex-col items-center gap-3 py-16 text-sm text-muted-foreground">
              <Loader2 className="h-8 w-8 animate-spin text-primary" /> Running Random Forest
              pipeline…
            </div>
          ) : mut.error ? (
            <ErrorState message={`Prediction failed: ${(mut.error as Error).message}`} />
          ) : r ? (
            <div className="space-y-5">
              <div
                className={`flex items-center gap-4 rounded-xl border p-5 ${r.prediction === 1 ? "border-destructive/40 bg-destructive/10" : "border-success/40 bg-success/10"}`}
              >
                {r.prediction === 1 ? (
                  <ShieldAlert className="h-10 w-10 text-destructive" />
                ) : (
                  <ShieldCheck className="h-10 w-10 text-success" />
                )}
                <div>
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    Predicted class
                  </p>
                  <p className="font-display text-2xl font-semibold">
                    {r.label ?? (r.prediction === 1 ? "Fraudulent" : "Legitimate")}
                  </p>
                </div>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Assessed Risk Level</span>
                <RiskPill level={riskLevel(prob, r.prediction)} />
              </div>
              {prob != null ? (
                <div>
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="text-muted-foreground">Fraud probability</span>
                    <span className="font-mono">{(prob * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full risk-gradient"
                      style={{ width: `${Math.min(100, Math.max(0, prob * 100))}%` }}
                    />
                  </div>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  The model did not return a probability.
                </p>
              )}
              {r.id && (
                <p className="text-xs text-muted-foreground font-mono">
                  Session Tracking ID: {r.id}
                </p>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2 py-16 text-center text-sm text-muted-foreground">
              <ScanSearch className="h-8 w-8" /> Submit transaction features on the left to screen
              with the ML model.
            </div>
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
