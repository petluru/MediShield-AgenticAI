import type { CaseDetail, CaseListItem, ReviewRequest, UploadResponse } from "./types";

// Matches backend/config.py's Settings.api_auth_tokens_list default
// ("dev-local-token") and cors_origins default ("http://localhost:3000",
// this app's own dev port) — override both via .env.local for a real
// deployment, never hardcode a real token in committed code.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const API_TOKEN = process.env.NEXT_PUBLIC_API_TOKEN ?? "dev-local-token";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      Authorization: `Bearer ${API_TOKEN}`,
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new ApiError(res.status, body || res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export function listCases(limit = 100): Promise<CaseListItem[]> {
  return apiFetch<CaseListItem[]>(`/cases?limit=${limit}`);
}

export function getCase(caseId: string): Promise<CaseDetail> {
  return apiFetch<CaseDetail>(`/cases/${encodeURIComponent(caseId)}`);
}

export function reviewCase(
  caseId: string,
  review: ReviewRequest,
): Promise<{ case_id: string; status: string; decision: string }> {
  return apiFetch(`/cases/${encodeURIComponent(caseId)}/review`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(review),
  });
}

export async function uploadCase(
  file: File,
  patientId?: string,
  policyNumber?: string,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (patientId) form.append("patient_id", patientId);
  if (policyNumber) form.append("policy_number", policyNumber);
  return apiFetch<UploadResponse>("/cases", { method: "POST", body: form });
}
