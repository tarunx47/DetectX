import { createFileRoute, Link } from "@tanstack/react-router";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  ShieldAlert,
  ShieldCheck,
  Percent,
  Database,
  Target,
  CheckCircle2,
  AlertTriangle,
  Layers,
} from "lucide-react";
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
import { useDatasetStats, useStats, useTransactions } from "@/hooks/use-backend";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — DetectX Fraud Detection" },
      {
        name: "description",
        content: "Overview of full dataset benchmark and live screened transactions in DetectX.",
      },
      { property: "og:title", content: "Dashboard — DetectX Fraud Detection" },
      {
        property: "og:description",
        content: "Full-dataset benchmark statistics and live session screening overview.",
      },
    ],
  }),
  component: Dashboard,
});

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  fontSize: 12,
};

function Dashboard() {
  const stats = useStats();
  const dataset = useDatasetStats();
  const tx = useTransactions();

  const ds = dataset.data;
  const s = stats.data;
  const liveRate = s && s.total > 0 ? ((s.fraud / s.total) * 100).toFixed(2) + "%" : "—";

  // Prepare full dataset type chart data
  const typeChartData = ds?.by_type
    ? Object.entries(ds.by_type).map(([name, val]) => ({
        type: name,
        transactions: val.transactions,
        actual_fraud: val.actual_fraud,
        predicted_fraud: val.predicted_fraud,
      }))
    : [];

  const liveCards = [
    {
      label: "Session Screened",
      value: s?.total.toLocaleString() ?? "0",
      icon: Activity,
      tone: "text-primary",
    },
    {
      label: "Session Fraud",
      value: s?.fraud.toLocaleString() ?? "0",
      icon: ShieldAlert,
      tone: "text-destructive",
    },
    {
      label: "Session Legitimate",
      value: s?.legitimate.toLocaleString() ?? "0",
      icon: ShieldCheck,
      tone: "text-success",
    },
    {
      label: "Session Fraud Rate",
      value: liveRate,
      icon: Percent,
      tone: "text-warning",
    },
  ];

  return (
    <AppShell>
      <PageHeader
        title="Fraud Intelligence Dashboard"
        subtitle="Benchmark evaluation on full dataset (6.36M transactions) & real-time session screening."
        right={stats.isDemo ? <DemoBadge /> : null}
      />

      {/* SECTION 1: FULL DATASET BENCHMARK EVALUATION */}
      <div className="mb-10 space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
          <div>
            <h2 className="flex items-center gap-2 font-display text-xl font-semibold text-foreground">
              <Database className="h-5 w-5 text-primary" />
              Full Dataset Analysis (6,362,620 Transactions)
            </h2>
            <p className="text-xs text-muted-foreground">
              Comprehensive evaluation metrics computed across all 6.36M records from the PaySim
              dataset.
            </p>
          </div>
          <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
            Static Dataset Benchmark
          </span>
        </div>

        {dataset.isLoading ? (
          <Loading label="Loading full dataset statistics…" />
        ) : dataset.error ? (
          <ErrorState message={`Could not load dataset stats: ${dataset.error.message}`} />
        ) : ds ? (
          <>
            {/* Top Overview Cards */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Panel>
                <div className="flex items-center justify-between">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    Total Transactions
                  </p>
                  <Layers className="h-4 w-4 text-primary" />
                </div>
                <p className="mt-3 font-display text-3xl font-semibold text-foreground">
                  {ds.total_transactions.toLocaleString()}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">100% full dataset scale</p>
              </Panel>

              <Panel>
                <div className="flex items-center justify-between">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    Actual Fraud
                  </p>
                  <AlertTriangle className="h-4 w-4 text-destructive" />
                </div>
                <p className="mt-3 font-display text-3xl font-semibold text-destructive">
                  {ds.actual_fraud.toLocaleString()}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Actual fraud rate: {ds.actual_fraud_rate}%
                </p>
              </Panel>

              <Panel>
                <div className="flex items-center justify-between">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    Predicted Fraud
                  </p>
                  <Target className="h-4 w-4 text-warning" />
                </div>
                <p className="mt-3 font-display text-3xl font-semibold text-warning">
                  {ds.predicted_fraud.toLocaleString()}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Predicted fraud rate: {ds.predicted_fraud_rate}%
                </p>
              </Panel>

              <Panel>
                <div className="flex items-center justify-between">
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    Actual Legitimate
                  </p>
                  <CheckCircle2 className="h-4 w-4 text-success" />
                </div>
                <p className="mt-3 font-display text-3xl font-semibold text-success">
                  {ds.actual_legitimate.toLocaleString()}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  True negatives: {ds.true_negative.toLocaleString()}
                </p>
              </Panel>
            </div>

            {/* Model Evaluation Metrics & Confusion Breakdown */}
            <div className="grid gap-4 sm:grid-cols-3">
              <Panel className="border-l-4 border-l-primary">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">
                  Recall (Sensitivity)
                </p>
                <p className="mt-2 font-display text-2xl font-bold text-foreground">
                  {(ds.evaluation.recall * 100).toFixed(2)}%
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Caught {ds.true_positive.toLocaleString()} of {ds.actual_fraud.toLocaleString()}{" "}
                  frauds (only {ds.false_negative} missed)
                </p>
              </Panel>

              <Panel className="border-l-4 border-l-warning">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">Precision</p>
                <p className="mt-2 font-display text-2xl font-bold text-foreground">
                  {(ds.evaluation.precision * 100).toFixed(2)}%
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {ds.true_positive.toLocaleString()} true positives vs{" "}
                  {ds.false_positive.toLocaleString()} false alarms
                </p>
              </Panel>

              <Panel className="border-l-4 border-l-success">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">
                  Overall Accuracy
                </p>
                <p className="mt-2 font-display text-2xl font-bold text-foreground">
                  {(ds.evaluation.accuracy * 100).toFixed(2)}%
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  High baseline due to{" "}
                  {((ds.actual_legitimate / ds.total_transactions) * 100).toFixed(2)}% legitimate
                  class
                </p>
              </Panel>
            </div>

            {/* Transaction Type Comparison Charts */}
            <div className="grid gap-4 lg:grid-cols-2">
              <Panel>
                <h3 className="mb-2 text-sm font-medium text-foreground">
                  Actual vs. Predicted Fraud by Transaction Type
                </h3>
                <p className="mb-4 text-xs text-muted-foreground">
                  Comparing ground-truth fraud occurrences with ML model detections across
                  transaction categories.
                </p>
                <div className="h-64">
                  <ResponsiveContainer>
                    <BarChart data={typeChartData}>
                      <CartesianGrid stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="type"
                        stroke="var(--muted-foreground)"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        stroke="var(--muted-foreground)"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--muted)" }} />
                      <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
                      <Bar
                        name="Actual Fraud"
                        dataKey="actual_fraud"
                        fill="var(--destructive)"
                        radius={[4, 4, 0, 0]}
                      />
                      <Bar
                        name="Predicted Fraud"
                        dataKey="predicted_fraud"
                        fill="var(--warning)"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>

              <Panel>
                <h3 className="mb-2 text-sm font-medium text-foreground">
                  Transaction Volume by Type
                </h3>
                <p className="mb-4 text-xs text-muted-foreground">
                  Total transaction volume distribution across all 6.36M PaySim records.
                </p>
                <div className="h-64">
                  <ResponsiveContainer>
                    <BarChart data={typeChartData}>
                      <CartesianGrid stroke="var(--border)" vertical={false} />
                      <XAxis
                        dataKey="type"
                        stroke="var(--muted-foreground)"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        stroke="var(--muted-foreground)"
                        fontSize={12}
                        tickLine={false}
                        axisLine={false}
                        tickFormatter={(v) => `${(v / 1000000).toFixed(1)}M`}
                      />
                      <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--muted)" }} />
                      <Bar
                        name="Total Transactions"
                        dataKey="transactions"
                        fill="var(--primary)"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            </div>
          </>
        ) : (
          <Empty
            title="Dataset statistics unavailable"
            hint="Verify reports/dataset_analysis.json on the backend."
          />
        )}
      </div>

      {/* SECTION 2: LIVE SESSION SCREENING */}
      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
          <div>
            <h2 className="flex items-center gap-2 font-display text-xl font-semibold text-foreground">
              <Activity className="h-5 w-5 text-primary" />
              Live Session Screening
            </h2>
            <p className="text-xs text-muted-foreground">
              Transactions screened dynamically through the FastAPI endpoint in the current
              browser/server session.
            </p>
          </div>
          <span className="rounded-full bg-success/10 px-3 py-1 text-xs font-medium text-success">
            Live Inference Session
          </span>
        </div>

        {stats.error && (
          <div className="mb-4">
            <ErrorState message={`Could not load live stats: ${stats.error.message}`} />
          </div>
        )}

        {/* Live Session Cards */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {liveCards.map(({ label, value, icon: Icon, tone }) => (
            <Panel key={label}>
              <div className="flex items-center justify-between">
                <p className="text-xs uppercase tracking-wider text-muted-foreground">{label}</p>
                <Icon className={`h-4 w-4 ${tone}`} />
              </div>
              <p className="mt-3 font-display text-3xl font-semibold">
                {stats.isLoading ? "…" : value}
              </p>
            </Panel>
          ))}
        </div>

        {/* Live Session Charts */}
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel>
            <h3 className="mb-4 text-sm font-medium">Session Transaction Activity</h3>
            {stats.isLoading ? (
              <Loading />
            ) : s?.daily?.length ? (
              <div className="h-64">
                <ResponsiveContainer>
                  <AreaChart data={s.daily}>
                    <defs>
                      <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.5} />
                        <stop offset="100%" stopColor="var(--primary)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="var(--border)" vertical={false} />
                    <XAxis
                      dataKey="date"
                      stroke="var(--muted-foreground)"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      stroke="var(--muted-foreground)"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Area
                      type="monotone"
                      dataKey="total"
                      stroke="var(--primary)"
                      fill="url(#g1)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <Empty
                title="No session activity yet"
                hint="Analyze transactions to view real-time activity trends."
              />
            )}
          </Panel>

          <Panel>
            <h3 className="mb-4 text-sm font-medium">Session Fraud Trend</h3>
            {stats.isLoading ? (
              <Loading />
            ) : s?.daily?.length ? (
              <div className="h-64">
                <ResponsiveContainer>
                  <BarChart data={s.daily}>
                    <CartesianGrid stroke="var(--border)" vertical={false} />
                    <XAxis
                      dataKey="date"
                      stroke="var(--muted-foreground)"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      stroke="var(--muted-foreground)"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--muted)" }} />
                    <Bar dataKey="fraud" fill="var(--destructive)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <Empty
                title="No session fraud data"
                hint="Analyzed fraudulent transactions will appear here."
              />
            )}
          </Panel>
        </div>

        {/* Live Recent Transactions */}
        <Panel className="mt-6">
          <div className="mb-2 flex items-center justify-between">
            <div>
              <h3 className="text-sm font-medium">Recent Session Transactions</h3>
              <p className="text-xs text-muted-foreground">
                Transactions evaluated during this runtime session
              </p>
            </div>
            <Link to="/history" className="text-xs text-primary hover:underline">
              View all in History →
            </Link>
          </div>
          {tx.isLoading ? (
            <Loading />
          ) : tx.error ? (
            <ErrorState message={tx.error.message} />
          ) : tx.data.length ? (
            <TxTable rows={tx.data.slice(0, 6)} compact />
          ) : (
            <Empty
              title="No session transactions yet"
              hint="Go to the Analyze page to screen a transaction and see live predictions here."
            />
          )}
        </Panel>
      </div>
    </AppShell>
  );
}
