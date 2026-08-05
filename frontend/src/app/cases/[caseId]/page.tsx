import Link from "next/link";
import { notFound } from "next/navigation";
import { ApiError, getCase } from "@/lib/api";
import { DecisionBadge, StatusBadge } from "@/components/StatusBadge";
import { ReviewActions } from "@/components/ReviewActions";

export const dynamic = "force-dynamic";

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details open className="rounded-lg border border-slate-200 bg-white">
      <summary className="cursor-pointer select-none px-5 py-3 text-sm font-semibold text-slate-800">
        {title}
      </summary>
      <div className="border-t border-slate-100 px-5 py-4 text-sm text-slate-700">{children}</div>
    </details>
  );
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 py-1">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right text-slate-900">{value}</dd>
    </div>
  );
}

export default async function CaseDetailPage({ params }: { params: Promise<{ caseId: string }> }) {
  const { caseId } = await params;

  let detail;
  try {
    detail = await getCase(caseId);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) notFound();
    throw err;
  }

  const { case: c, pending_review } = detail;

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <Link href="/" className="text-sm text-slate-500 hover:text-slate-800">
        ← Back to dashboard
      </Link>

      <div className="mt-3 mb-8 flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-lg font-semibold text-slate-900">{c.case_id}</h1>
        <StatusBadge status={c.status} />
        {c.decision && <DecisionBadge decision={c.decision.decision} />}
      </div>

      {pending_review && (
        <div className="mb-8">
          <ReviewActions caseId={c.case_id} pendingReview={pending_review} />
        </div>
      )}

      <div className="space-y-4">
        <Panel title="Case Details">
          <dl>
            <Field label="Content type" value={c.content_type} />
            <Field label="Patient ID" value={c.patient_id ?? "—"} />
            <Field label="Policy number" value={c.policy_number ?? "—"} />
            <Field label="File" value={<span className="font-mono text-xs">{c.file_path.split(/[\\/]/).pop()}</span>} />
            <Field label="Created" value={new Date(c.created_at).toLocaleString()} />
            <Field label="Last updated" value={new Date(c.updated_at).toLocaleString()} />
          </dl>
        </Panel>

        {c.decision && (
          <Panel title="Final Decision">
            <div className="mb-3 flex items-center gap-3">
              <DecisionBadge decision={c.decision.decision} />
              <span className="text-slate-500">confidence {(c.decision.confidence * 100).toFixed(0)}%</span>
              {c.decision.escalated_to_opus && (
                <span className="rounded-full bg-purple-100 px-2 py-0.5 text-xs font-medium text-purple-700 ring-1 ring-inset ring-purple-300">
                  escalated to Opus
                </span>
              )}
            </div>
            <p className="leading-relaxed text-slate-700">{c.decision.justification}</p>
          </Panel>
        )}

        {c.human_review && (
          <Panel title="Human Review">
            <dl>
              <Field label="Outcome" value={c.human_review.outcome} />
              {c.human_review.overridden_decision && (
                <Field label="Overridden to" value={<DecisionBadge decision={c.human_review.overridden_decision} />} />
              )}
              <Field label="Reviewed at" value={new Date(c.human_review.reviewed_at).toLocaleString()} />
            </dl>
            {c.human_review.reviewer_notes && (
              <p className="mt-3 rounded-md bg-slate-50 p-3 text-slate-700">{c.human_review.reviewer_notes}</p>
            )}
          </Panel>
        )}

        {c.classifier_result && (
          <Panel title="Classifier Agent">
            <dl>
              <Field label="Document type" value={c.classifier_result.doc_type} />
              <Field label="Confidence" value={`${(c.classifier_result.confidence * 100).toFixed(0)}%`} />
              <Field label="Routing tags" value={c.classifier_result.routing_tags.join(", ") || "—"} />
            </dl>
          </Panel>
        )}

        {c.kyc_result && (
          <Panel title="KYC Agent">
            <dl>
              <Field label="Passed" value={c.kyc_result.kyc_passed ? "Yes" : "No"} />
              <Field label="Confidence" value={`${(c.kyc_result.confidence * 100).toFixed(0)}%`} />
              <Field label="Flags" value={c.kyc_result.flags.join(", ") || "none"} />
            </dl>
          </Panel>
        )}

        {c.claims_result && (
          <Panel title="Claims Agent">
            <dl>
              <Field label="Schema valid" value={c.claims_result.schema_valid ? "Yes" : "No"} />
              <Field label="Confidence" value={`${(c.claims_result.confidence * 100).toFixed(0)}%`} />
              <Field label="Claim amount" value={c.claims_result.extracted_fields.claim_amount ?? "—"} />
              <Field label="CPT codes" value={c.claims_result.extracted_fields.cpt_codes.join(", ") || "—"} />
              <Field label="ICD-10 codes" value={c.claims_result.extracted_fields.icd10_codes.join(", ") || "—"} />
              <Field label="Provider NPI" value={c.claims_result.extracted_fields.provider_npi ?? "—"} />
              <Field label="Service date" value={c.claims_result.extracted_fields.service_date ?? "—"} />
            </dl>
            {c.claims_result.validation_errors.length > 0 && (
              <p className="mt-3 rounded-md bg-rose-50 p-3 text-rose-700">
                {c.claims_result.validation_errors.join("; ")}
              </p>
            )}
          </Panel>
        )}

        {c.policy_result && (
          <Panel title="Policy RAG Agent">
            <dl>
              <Field label="Covered" value={c.policy_result.covered ? "Yes" : "No"} />
              <Field label="Coverage" value={`${c.policy_result.coverage_percentage}%`} />
              <Field label="Confidence" value={`${(c.policy_result.confidence * 100).toFixed(0)}%`} />
              <Field label="Exclusions" value={c.policy_result.exclusions.join(", ") || "none"} />
            </dl>
            <p className="mt-3 rounded-md bg-slate-50 p-3 leading-relaxed text-slate-700">
              {c.policy_result.policy_clause}
            </p>
          </Panel>
        )}

        {c.fraud_result && (
          <Panel title="Fraud Detection Agent">
            <dl>
              <Field label="Fraud score" value={c.fraud_result.fraud_score.toFixed(2)} />
              <Field label="Risk level" value={c.fraud_result.risk_level} />
              {c.fraud_result.escalated_to_opus && <Field label="Escalated to Opus" value="Yes" />}
            </dl>
            {c.fraud_result.anomalies.length > 0 ? (
              <ul className="mt-3 list-inside list-disc space-y-1 text-slate-700">
                {c.fraud_result.anomalies.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-slate-500">No anomalies found.</p>
            )}
          </Panel>
        )}

        {c.decision?.agent_summaries && Object.keys(c.decision.agent_summaries).length > 0 && (
          <Panel title="Audit Trail — Agent Summaries">
            <ol className="space-y-2">
              {Object.entries(c.decision.agent_summaries).map(([agent, summary]) => (
                <li key={agent} className="flex gap-3">
                  <span className="w-24 shrink-0 font-medium capitalize text-slate-500">{agent}</span>
                  <span className="text-slate-700">{summary}</span>
                </li>
              ))}
            </ol>
          </Panel>
        )}

        {c.errors.length > 0 && (
          <Panel title="Security Flags">
            <ul className="list-inside list-disc space-y-1 text-amber-700">
              {c.errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          </Panel>
        )}
      </div>
    </div>
  );
}
