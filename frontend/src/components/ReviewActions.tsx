"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { reviewCase } from "@/lib/api";
import type { Decision, PendingReview } from "@/lib/types";

export function ReviewActions({ caseId, pendingReview }: { caseId: string; pendingReview: PendingReview }) {
  const router = useRouter();
  const [mode, setMode] = useState<"idle" | "override">("idle");
  const [overriddenDecision, setOverriddenDecision] = useState<Decision>("APPROVE");
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(outcome: "APPROVED" | "OVERRIDDEN") {
    setSubmitting(true);
    setError(null);
    try {
      await reviewCase(caseId, {
        outcome,
        overridden_decision: outcome === "OVERRIDDEN" ? overriddenDecision : undefined,
        notes,
      });
      router.refresh();
      setMode("idle");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit review");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="rounded-lg border border-orange-200 bg-orange-50 p-5">
      <h2 className="text-sm font-semibold text-orange-900">Awaiting human review</h2>
      <p className="mt-2 text-sm text-orange-900/80">{pendingReview.justification}</p>

      {error && <p className="mt-3 text-sm text-rose-700">{error}</p>}

      {mode === "idle" ? (
        <div className="mt-4 flex gap-3">
          <button
            type="button"
            disabled={submitting}
            onClick={() => submit("APPROVED")}
            className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
          >
            Confirm computed decision ({pendingReview.decision})
          </button>
          <button
            type="button"
            disabled={submitting}
            onClick={() => setMode("override")}
            className="rounded-md border border-orange-300 bg-white px-4 py-2 text-sm font-medium text-orange-800 hover:bg-orange-100 disabled:opacity-50"
          >
            Override decision
          </button>
        </div>
      ) : (
        <div className="mt-4 space-y-3">
          <div className="flex items-center gap-3">
            <label className="text-sm font-medium text-orange-900" htmlFor="override-decision">
              New decision
            </label>
            <select
              id="override-decision"
              value={overriddenDecision}
              onChange={(e) => setOverriddenDecision(e.target.value as Decision)}
              className="rounded-md border border-orange-300 bg-white px-2 py-1 text-sm"
            >
              <option value="APPROVE">APPROVE</option>
              <option value="REJECT">REJECT</option>
              <option value="ESCALATE">ESCALATE</option>
            </select>
          </div>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Reviewer notes (why are you overriding this?)"
            rows={3}
            className="w-full rounded-md border border-orange-300 bg-white px-3 py-2 text-sm"
          />
          <div className="flex gap-3">
            <button
              type="button"
              disabled={submitting}
              onClick={() => submit("OVERRIDDEN")}
              className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
            >
              Submit override
            </button>
            <button
              type="button"
              disabled={submitting}
              onClick={() => setMode("idle")}
              className="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
