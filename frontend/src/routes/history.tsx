import { useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Search } from "lucide-react";
import { AppShell } from "@/components/detectx/AppShell";
import {
  DemoBadge,
  Empty,
  ErrorState,
  Loading,
  PageHeader,
  Panel,
} from "@/components/detectx/ui-bits";
import { TxTable } from "@/components/detectx/TxTable";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTransactions } from "@/hooks/use-backend";
import { riskLevel, TX_TYPES } from "@/lib/api";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Transaction History — DetectX" },
      { name: "description", content: "Search and filter every transaction screened by DetectX." },
      { property: "og:title", content: "Transaction History — DetectX" },
      {
        property: "og:description",
        content: "Search and filter screened transactions by prediction and risk.",
      },
    ],
  }),
  component: HistoryPage,
});

function HistoryPage() {
  const tx = useTransactions();
  const [q, setQ] = useState("");
  const [type, setType] = useState("all");
  const [pred, setPred] = useState("all");
  const [risk, setRisk] = useState("all");

  const rows = useMemo(
    () =>
      tx.data.filter((r) => {
        if (q && !`${r.id} ${r.type} ${r.amount}`.toLowerCase().includes(q.toLowerCase()))
          return false;
        if (type !== "all" && r.type !== type) return false;
        if (pred !== "all" && String(r.prediction) !== pred) return false;
        if (risk !== "all" && riskLevel(r.probability, r.prediction) !== risk) return false;
        return true;
      }),
    [tx.data, q, type, pred, risk],
  );

  return (
    <AppShell>
      <PageHeader
        title="Transaction History"
        subtitle="Every transaction screened by the model."
        right={tx.isDemo ? <DemoBadge /> : null}
      />
      <Panel>
        <div className="mb-4 grid gap-3 md:grid-cols-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              className="pl-9"
              placeholder="Search ID, type, amount"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          <Select value={type} onValueChange={setType}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All types</SelectItem>
              {TX_TYPES.map((t) => (
                <SelectItem key={t} value={t}>
                  {t}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={pred} onValueChange={setPred}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All predictions</SelectItem>
              <SelectItem value="1">Fraud</SelectItem>
              <SelectItem value="0">Legitimate</SelectItem>
            </SelectContent>
          </Select>
          <Select value={risk} onValueChange={setRisk}>
            <SelectTrigger>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All risk levels</SelectItem>
              <SelectItem value="High">High</SelectItem>
              <SelectItem value="Medium">Medium</SelectItem>
              <SelectItem value="Low">Low</SelectItem>
            </SelectContent>
          </Select>
        </div>
        {tx.isLoading ? (
          <Loading />
        ) : tx.error ? (
          <ErrorState message={tx.error.message} />
        ) : rows.length ? (
          <>
            <TxTable rows={rows} />
            <p className="mt-3 text-xs text-muted-foreground">
              {rows.length} of {tx.data.length} transactions
            </p>
          </>
        ) : (
          <Empty
            title="No matching transactions"
            hint={
              tx.data.length ? "Try clearing filters." : "Analyze a transaction to get started."
            }
          />
        )}
      </Panel>
    </AppShell>
  );
}
