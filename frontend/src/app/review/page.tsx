import Link from "next/link";
import { listCases } from "@/lib/api";
import { DecisionBadge } from "@/components/StatusBadge";

export const dynamic = "force-dynamic";

export default async function ReviewQueuePage() {
  let cases: Awaited<ReturnType<typeof listCases>> = [];
  let loadError: string | null = null;
  try {
    cases = await listCases(200);
  } catch (err) {
    loadError = err instanceof Error ? err.message : "Failed to load cases";
  }

  const pending = cases.filter((c) => c.status === "AWAITING_REVIEW");

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <h1 className="text-2xl font-semibold text-slate-900">Review Queue</h1>
      <p className="mt-1 text-sm text-slate-500">
        Cases the orchestrator escalated for human sign-off before a final decision is recorded.
      </p>

      {loadError && (
        <div className="mt-6 rounded-md bg-rose-50 px-4 py-3 text-sm text-rose-700 ring-1 ring-inset ring-rose-200">
          Could not reach the MediShield API: {loadError}
        </div>
      )}

      {!loadError && pending.length === 0 && (
        <div className="mt-8 rounded-lg border border-dashed border-slate-300 bg-white px-6 py-10 text-center text-slate-400">
          Nothing awaiting review right now.
        </div>
      )}

      <ul className="mt-6 space-y-3">
        {pending.map((c) => (
          <li key={c.case_id} className="rounded-lg border border-orange-200 bg-orange-50 px-5 py-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs text-slate-700">{c.case_id}</span>
              <DecisionBadge decision={c.decision} />
            </div>
            <div className="mt-2 flex items-center justify-between">
              <span className="text-xs text-slate-500">
                Last updated {new Date(c.updated_at).toLocaleString()}
              </span>
              <Link
                href={`/cases/${encodeURIComponent(c.case_id)}`}
                className="text-sm font-medium text-orange-800 hover:underline"
              >
                Review →
              </Link>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
