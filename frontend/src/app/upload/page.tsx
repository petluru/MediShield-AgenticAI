"use client";

import { useState, type FormEvent } from "react";
import { useRouter } from "next/navigation";
import { uploadCase } from "@/lib/api";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState("");
  const [policyNumber, setPolicyNumber] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose a file to upload.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const res = await uploadCase(file, patientId || undefined, policyNumber || undefined);
      router.push(`/cases/${encodeURIComponent(res.case_id)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setSubmitting(false);
    }
  }

  return (
    <div className="mx-auto max-w-xl px-6 py-10">
      <h1 className="text-2xl font-semibold text-slate-900">Upload document</h1>
      <p className="mt-1 text-sm text-slate-500">
        Submits a document to the MediShield intake pipeline for classification, KYC, claims
        extraction, policy review, and fraud checks.
      </p>

      <form onSubmit={handleSubmit} className="mt-8 space-y-5 rounded-lg border border-slate-200 bg-white p-6">
        <div>
          <label className="block text-sm font-medium text-slate-700" htmlFor="file">
            Document
          </label>
          <input
            id="file"
            type="file"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="mt-1 block w-full text-sm text-slate-600 file:mr-4 file:rounded-md file:border-0 file:bg-slate-900 file:px-4 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-slate-700"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700" htmlFor="patientId">
            Patient ID <span className="font-normal text-slate-400">(optional)</span>
          </label>
          <input
            id="patientId"
            type="text"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
            placeholder="PT-99733"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-700" htmlFor="policyNumber">
            Policy number <span className="font-normal text-slate-400">(optional)</span>
          </label>
          <input
            id="policyNumber"
            type="text"
            value={policyNumber}
            onChange={(e) => setPolicyNumber(e.target.value)}
            className="mt-1 w-full rounded-md border border-slate-300 px-3 py-2 text-sm"
            placeholder="POL-4821"
          />
        </div>

        {error && <p className="text-sm text-rose-700">{error}</p>}

        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-50"
        >
          {submitting ? "Uploading…" : "Submit for intake"}
        </button>
      </form>
    </div>
  );
}
