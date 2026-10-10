import type { ReactNode } from "react";
import { AlertTriangle, FlaskConical, Inbox, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export function Panel({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={cn("panel rounded-xl p-5", className)}>{children}</div>;
}

export function PageHeader({
  title,
  subtitle,
  right,
}: {
  title: string;
  subtitle: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="font-display text-3xl font-semibold tracking-tight text-foreground">
          {title}
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      </div>
      {right}
    </div>
  );
}

export function DemoBadge() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-warning/40 bg-warning/10 px-3 py-1 text-xs font-medium text-warning">
      <FlaskConical className="h-3.5 w-3.5" /> Demo data — backend not connected
    </span>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
      <Loader2 className="h-4 w-4 animate-spin" /> {label}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /> <span>{message}</span>
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 py-12 text-center">
      <Inbox className="h-8 w-8 text-muted-foreground" />
      <p className="text-sm font-medium text-foreground">{title}</p>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export function PredictionPill({ prediction }: { prediction: number }) {
  return prediction === 1 ? (
    <span className="rounded-full bg-destructive/15 px-2.5 py-0.5 text-xs font-semibold text-destructive">
      Fraud
    </span>
  ) : (
    <span className="rounded-full bg-success/15 px-2.5 py-0.5 text-xs font-semibold text-success">
      Legitimate
    </span>
  );
}

export function RiskPill({ level }: { level: "High" | "Medium" | "Low" }) {
  const cls =
    level === "High"
      ? "text-destructive border-destructive/40"
      : level === "Medium"
        ? "text-warning border-warning/40"
        : "text-success border-success/40";
  return (
    <span className={cn("rounded-md border px-2 py-0.5 text-xs font-medium", cls)}>{level}</span>
  );
}

export const fmtMoney = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
