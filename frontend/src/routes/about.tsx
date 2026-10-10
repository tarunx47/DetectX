import { createFileRoute } from "@tanstack/react-router";
import { Brain, Database, Gauge, ShieldAlert } from "lucide-react";
import { AppShell } from "@/components/detectx/AppShell";
import { PageHeader, Panel } from "@/components/detectx/ui-bits";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About DetectX — ML for Fraud Detection" },
      {
        name: "description",
        content:
          "Why financial fraud matters and how machine learning helps flag suspicious transactions.",
      },
      { property: "og:title", content: "About DetectX — ML for Fraud Detection" },
      {
        property: "og:description",
        content: "How DetectX uses machine learning to flag suspicious transactions.",
      },
    ],
  }),
  component: About,
});

const steps = [
  {
    icon: Database,
    title: "Learn from history",
    body: "The model is trained on labeled transactions — type, amount and balance changes on both sides — to learn what fraud looks like.",
  },
  {
    icon: Brain,
    title: "Spot patterns",
    body: "Fraud often hides in subtle signals: accounts drained to zero, transfers followed by cash-outs, balances that don't add up.",
  },
  {
    icon: Gauge,
    title: "Score each transaction",
    body: "Every new transaction gets a prediction and, when available, a probability that drives its risk level.",
  },
  {
    icon: ShieldAlert,
    title: "Support analysts",
    body: "DetectX flags suspicious activity for human review — it assists decisions rather than replacing them.",
  },
];

function About() {
  return (
    <AppShell>
      <PageHeader title="About DetectX" subtitle="AI-powered financial fraud detection." />
      <Panel className="mb-6">
        <h2 className="font-display text-xl font-semibold">The problem</h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          Digital payments move billions every day, and fraudsters exploit that speed. Fraudulent
          transactions are rare — often well under 1% of volume — which makes them hard to catch
          with fixed rules and easy to miss in manual review. Every missed case costs money and
          trust; every false alarm frustrates a genuine customer.
        </p>
        <h2 className="mt-6 font-display text-xl font-semibold">How machine learning helps</h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          Instead of hand-written rules, a classifier learns from past examples which combinations
          of features signal fraud. DetectX's React frontend sends each transaction to a Python
          FastAPI service hosting the trained model, and displays the prediction in seconds.
        </p>
      </Panel>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {steps.map(({ icon: Icon, title, body }) => (
          <Panel key={title}>
            <Icon className="h-6 w-6 text-primary" />
            <h3 className="mt-3 font-medium">{title}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{body}</p>
          </Panel>
        ))}
      </div>
    </AppShell>
  );
}
