// api.js — All API calls in one place, automatically attaches JWT token

const BASE = "https://govprep-backend-pkn9.onrender.com";

function getToken() { return localStorage.getItem("govprep_token"); }
export function setToken(t) { localStorage.setItem("govprep_token", t); }
export function clearToken() { localStorage.removeItem("govprep_token"); localStorage.removeItem("govprep_user"); }
export function getUser() { try { return JSON.parse(localStorage.getItem("govprep_user") || "null"); } catch { return null; } }
export function setUser(u) { localStorage.setItem("govprep_user", JSON.stringify(u)); }

async function req(method, url, body, isForm = false) {
  const headers = {};
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (body && !isForm) headers["Content-Type"] = "application/json";

  const res = await fetch(BASE + url, {
    method,
    headers,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// Auth
export const signup = (name, password, examTarget) => req("POST", "/api/auth/signup", { name, password, examTarget });
export const login = (name, password) => req("POST", "/api/auth/login", { name, password });
export const getMe = () => req("GET", "/api/auth/me");

// PDFs
export const uploadPDF = (formData) => req("POST", "/api/upload", formData, true);
export const getPDFs = () => req("GET", "/api/pdfs");
export const deletePDF = (id) => req("DELETE", `/api/pdfs/${id}`);

// Tests
export const generateQuestions = (body) => req("POST", "/api/generate-questions", body);
export const submitTest = (body) => req("POST", "/api/submit-test", body);
export const getMyTests = () => req("GET", "/api/my-tests");

// Assistant
export const sendChat = (message, context) => req("POST", "/api/assistant/chat", { message, context });
export const getChatHistory = () => req("GET", "/api/assistant/history");
export const clearChatHistory = () => req("DELETE", "/api/assistant/history");
