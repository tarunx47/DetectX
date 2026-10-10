import { useState, type ReactNode } from "react";
import { Link } from "@tanstack/react-router";
import { useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  History,
  Info,
  LayoutDashboard,
  Menu,
  Settings2,
  ShieldCheck,
  X,
} from "lucide-react";
import { useBackendStatus } from "@/hooks/use-backend";
import { getApiBaseUrl, setApiBaseUrl } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

const nav = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analyze", label: "Analyze", icon: Activity },
  { to: "/history", label: "History", icon: History },
  { to: "/about", label: "About", icon: Info },
] as const;

function BackendStatus() {
  const { isSuccess, isLoading } = useBackendStatus();
  const qc = useQueryClient();
  const [url, setUrl] = useState("");
  return (
    <Popover onOpenChange={(o) => o && setUrl(getApiBaseUrl())}>
      <PopoverTrigger asChild>
        <button className="flex items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground">
          <span
            className={`h-2 w-2 rounded-full ${isLoading ? "bg-muted-foreground" : isSuccess ? "bg-success shadow-[0_0_8px_var(--success)]" : "bg-warning"}`}
          />
          {isLoading ? "Checking API" : isSuccess ? "API connected" : "API offline"}
          <Settings2 className="h-3.5 w-3.5" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80">
        <p className="text-sm font-medium">FastAPI base URL</p>
        <p className="mb-3 text-xs text-muted-foreground">
          Expects /health, /predict, /transactions, /stats.
        </p>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            setApiBaseUrl(url);
            qc.invalidateQueries();
          }}
        >
          <Input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="http://localhost:8000"
          />
          <Button type="submit" size="sm">
            Save
          </Button>
        </form>
      </PopoverContent>
    </Popover>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const links = nav.map(({ to, label, icon: Icon }) => (
    <Link
      key={to}
      to={to}
      onClick={() => setOpen(false)}
      activeOptions={{ exact: to === "/" }}
      className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
      activeProps={{ className: "bg-primary/15 !text-primary" }}
    >
      <Icon className="h-4 w-4" /> {label}
    </Link>
  ));
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-6 px-4 sm:px-6">
          <Link to="/" className="flex items-center gap-2">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-primary/20 text-primary">
              <ShieldCheck className="h-5 w-5" />
            </span>
            <span className="font-display text-lg font-semibold tracking-tight">
              Detect<span className="text-primary">X</span>
            </span>
          </Link>
          <nav className="hidden items-center gap-1 md:flex">{links}</nav>
          <div className="ml-auto flex items-center gap-2">
            <BackendStatus />
            <button className="md:hidden" onClick={() => setOpen(!open)} aria-label="Menu">
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
        {open && (
          <nav className="flex flex-col gap-1 border-t border-border p-3 md:hidden">{links}</nav>
        )}
      </header>
      <main className="mx-auto max-w-7xl px-4 py-10 sm:px-6">{children}</main>
      <footer className="border-t border-border py-6 text-center text-xs text-muted-foreground">
        DetectX · AI-powered financial fraud detection
      </footer>
    </div>
  );
}
