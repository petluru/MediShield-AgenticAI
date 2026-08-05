import Link from "next/link";
import { listCases } from "@/lib/api";
import { DecisionBadge, StatusBadge } from "@/components/StatusBadge";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  let cases: Awaited<ReturnType<typeof listCases>> = [];
  let loadError: string | null = null;
  try {
    cases = await listCases(200);
  } catch (err) {
    loadError = err instanceof Error ? err.message : "Failed to load cases";
  }

  const awaitingReview = cases.filter((c) => c.status === "AWAITING_REVIEW").length;

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-900">Case Dashboard</h1>
          <p className="mt-1 text-sm text-slate-500">
            {cases.length} case{cases.length === 1 ? "" : "s"} on record
            {awaitingReview > 0 && (
              <>
                {" "}
                ·{" "}
                <Link href="/review" className="font-medium text-orange-700 hover:underline">
                  {awaitingReview} awaiting human review
                </Link>
              </>
            )}
          </p>
        </div>
        <Link
          href="/upload"
          className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
        >
          Upload document
        </Link>
      </div>

      {loadError && (
        <div className="mb-6 rounded-md bg-rose-50 px-4 py-3 text-sm text-rose-700 ring-1 ring-inset ring-rose-200">
          Could not reach the MediShield API: {loadError}. Is the backend running (
          <code className="rounded bg-rose-100 px-1">uv run uvicorn backend.api.app:app</code>)?
        </div>
      )}

      <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
        <table className="min-w-full divide-y divide-slate-200 text-sm">
          <thead className="bg-slate-50">
            <tr>
              <th className="px-4 py-3 text-left font-medium text-slate-500">Case ID</th>
              <th className="px-4 py-3 text-left font-medium text-slate-500">Status</th>
              <th className="px-4 py-3 text-left font-medium text-slate-500">Decision</th>
              <th className="px-4 py-3 text-left font-medium text-slate-500">Updated</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {cases.map((c) => (
              <tr key={c.case_id} className="hover:bg-slate-50">
                <td className="px-4 py-3 font-mono text-xs text-slate-700">{c.case_id}</td>
                <td className="px-4 py-3">
                  <StatusBadge status={c.status} />
                </td>
                <td className="px-4 py-3">
                  <DecisionBadge decision={c.decision} />
                </td>
                <td className="px-4 py-3 text-slate-500">{new Date(c.updated_at).toLocaleString()}</td>
                <td className="px-4 py-3 text-right">
                  <Link href={`/cases/${encodeURIComponent(c.case_id)}`} className="text-sky-700 hover:underline">
                    View →
                  </Link>
                </td>
              </tr>
            ))}
            {cases.length === 0 && !loadError && (
              <tr>
                <td colSpan={5} className="px-4 py-10 text-center text-slate-400">
                  No cases yet. Upload a document to get started.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
