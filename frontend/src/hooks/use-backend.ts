import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { demoStats, demoTransactions } from "@/lib/demo-data";

export function useBackendStatus() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    retry: false,
    refetchInterval: 15000,
  });
}

/** Returns real data when backend is online; labeled demo data otherwise. */
export function useTransactions() {
  const health = useBackendStatus();
  const online = health.isSuccess;
  const q = useQuery({ queryKey: ["transactions"], queryFn: api.transactions, enabled: online });
  return {
    checking: health.isLoading,
    online,
    isDemo: !online && !health.isLoading,
    isLoading: health.isLoading || (online && q.isLoading),
    error: q.error as Error | null,
    data: online ? (q.data ?? []) : health.isLoading ? [] : demoTransactions,
  };
}

export function useStats() {
  const health = useBackendStatus();
  const online = health.isSuccess;
  const q = useQuery({ queryKey: ["stats"], queryFn: api.stats, enabled: online });
  return {
    online,
    isDemo: !online && !health.isLoading,
    isLoading: health.isLoading || (online && q.isLoading),
    error: q.error as Error | null,
    data: online ? q.data : health.isLoading ? undefined : demoStats,
  };
}

export function useDatasetStats() {
  const health = useBackendStatus();
  const online = health.isSuccess;
  const q = useQuery({ queryKey: ["dataset-stats"], queryFn: api.datasetStats, enabled: online });
  return {
    online,
    isLoading: health.isLoading || (online && q.isLoading),
    error: q.error as Error | null,
    data: q.data,
  };
}
