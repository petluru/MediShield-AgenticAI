import type { CaseStatusValue, Decision } from "@/lib/types";

const STATUS_STYLES: Record<CaseStatusValue, string> = {
  RECEIVED: "bg-slate-100 text-slate-700 ring-slate-300",
  CLASSIFIED: "bg-sky-100 text-sky-700 ring-sky-300",
  PROCESSING: "bg-sky-100 text-sky-700 ring-sky-300",
  FRAUD_CHECK: "bg-amber-100 text-amber-800 ring-amber-300",
  AGGREGATED: "bg-amber-100 text-amber-800 ring-amber-300",
  AWAITING_REVIEW: "bg-orange-100 text-orange-800 ring-orange-300 animate-pulse",
  DECIDED: "bg-slate-100 text-slate-700 ring-slate-300",
};

export function StatusBadge({ status }: { status: CaseStatusValue | null }) {
  if (!status) return null;
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${STATUS_STYLES[status]}`}
    >
      {status.replace("_", " ")}
    </span>
  );
}

const DECISION_STYLES: Record<Decision, string> = {
  APPROVE: "bg-emerald-100 text-emerald-800 ring-emerald-300",
  REJECT: "bg-rose-100 text-rose-800 ring-rose-300",
  ESCALATE: "bg-orange-100 text-orange-800 ring-orange-300",
};

export function DecisionBadge({ decision }: { decision: Decision | null }) {
  if (!decision) return <span className="text-xs text-slate-400">—</span>;
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ring-inset ${DECISION_STYLES[decision]}`}
    >
      {decision}
    </span>
  );
}
