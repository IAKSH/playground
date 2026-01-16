// src/api.js
const BASE = import.meta.env.VITE_API_BASE || '';
let _session_id = null;

export function setSessionId(id) { _session_id = id; }
export function getSessionId() { return _session_id; }
export function clearSessionId() { _session_id = null; }

function ensureSession_id(provided) {
  const sid = provided || _session_id;
  if (!sid) throw Object.assign(new Error('Missing session_id'), { isSessionError: true });
  return sid;
}

function isSessionErrorText(text) {
  if (!text) return false;
  const t = text.toLowerCase();
  return t.includes('invalid session') || t.includes('missing session_id') || t.includes('no session');
}

async function fetchJsonOrThrow(res) {
  if (res.ok) return res.json();
  const txt = await res.text();
  const err = new Error(txt || `HTTP ${res.status}`);
  err.status = res.status;
  err.isSessionError = isSessionErrorText(txt) || res.status === 404;
  throw err;
}

export async function uploadImage(file, session_id = null) {
  const form = new FormData();
  form.append('file', file);
  const url = session_id ? `${BASE}/image/upload?session_id=${encodeURIComponent(session_id)}` : `${BASE}/image/upload`;
  const headers = {};
  if (session_id) headers['X-Session-Id'] = session_id;

  const res = await fetch(url, { method: 'POST', body: form, headers });
  if (!res.ok) {
    const txt = await res.text();
    const err = new Error(txt || 'Upload failed');
    err.isSessionError = isSessionErrorText(txt) || res.status === 404;
    throw err;
  }
  const json = await res.json();
  if (json && json.session_id) _session_id = json.session_id;
  return json;
}

export async function processImage({ session_id = null, threshold, min_area, margin = 10 }) {
  const sid = ensureSession_id(session_id);
  const res = await fetch(`${BASE}/image/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sid, threshold, min_area, margin }),
  });
  return fetchJsonOrThrow(res);
}

export async function getThresholdedPng(session_id = null) {
  const sid = ensureSession_id(session_id);
  const res = await fetch(`${BASE}/image/thresholded?session_id=${encodeURIComponent(sid)}`);
  if (!res.ok) {
    const txt = await res.text();
    const err = new Error(txt || 'Failed to fetch thresholded image');
    err.isSessionError = isSessionErrorText(txt) || res.status === 404;
    throw err;
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function getFinalPng(session_id = null) {
  const sid = ensureSession_id(session_id);
  const res = await fetch(`${BASE}/image/final?session_id=${encodeURIComponent(sid)}`);
  if (!res.ok) {
    const txt = await res.text();
    const err = new Error(txt || 'Failed to fetch final image');
    err.isSessionError = isSessionErrorText(txt) || res.status === 404;
    throw err;
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function getBoundary(session_id = null) {
  const sid = ensureSession_id(session_id);
  const res = await fetch(`${BASE}/image/boundary?session_id=${encodeURIComponent(sid)}`);
  return fetchJsonOrThrow(res);
}

export async function saveCrop({ session_id = null, x, y, w, h }) {
  const sid = ensureSession_id(session_id);
  const res = await fetch(`${BASE}/image/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sid, x, y, w, h }),
  });
  if (!res.ok) {
    const txt = await res.text();
    const err = new Error(txt || 'Save crop failed');
    err.isSessionError = isSessionErrorText(txt) || res.status === 404;
    throw err;
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'crop.png';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function getStatus(session_id = null) {
  const sid = ensureSession_id(session_id);
  const res = await fetch(`${BASE}/image/status?session_id=${encodeURIComponent(sid)}`);
  return fetchJsonOrThrow(res);
}
