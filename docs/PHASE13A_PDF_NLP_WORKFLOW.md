# Phase 13A — PDF → NLP → Question Generation

## Request flow

1. React posts the PDF to `POST /api/upload` with `subjectMode=auto` or `subjectMode=manual`.
2. Express uses the existing `pdf-parse` implementation and stores the original PDF path, full extracted text, and page-wise text.
3. In auto mode, Express creates up to eight representative chunks distributed across the PDF pages and calls the configured FastAPI service at the fixed path `/classify/bert`.
4. Express aggregates per-chunk class probabilities for the four currently trained subjects. The stored score is an uncalibrated mean model score; margin and agreement are also checked.
5. A clear result is saved with status `awaiting_confirmation`. The upload UI shows the candidate subject and asks the student to confirm or change it. An unclear result is saved as `manual_required`; the PDF remains available for manual selection.
6. `PATCH /api/pdfs/:id/subject` verifies ownership and saves the confirmed/manual subject. The Generate Test page also exposes confirmation for pending PDFs.
7. `POST /api/generate-questions` refuses an unconfirmed auto-detection with HTTP 409. After confirmation it uses the selected PDF pages and saved subject in the existing Groq prompt, then saves the generated test as before.

## Current scope and honest limitations

- Automatic classification supports only History, Geography, Polity, and Science. Other application subjects can be selected manually.
- The model was trained on labelled question/answer records, not long-document PDF chapters. Chunk sampling and manual confirmation reduce risk but do not establish PDF-domain accuracy. Validate with representative study PDFs before treating this as production-grade automatic labeling.
- A mixed-subject document may be sent for manual selection based on chunk disagreement.
- Scanned/image-only PDFs are not OCRed; the user must provide a text-based PDF or select a subject manually if extraction yields insufficient text.
- Groq remains responsible for question generation. BERT only predicts subject labels.
- The frontend still has no separate NLP Lab; it only presents the upload result and required subject confirmation.

## Local verification

Run the services as usual: FastAPI at `127.0.0.1:8001`, Express at `localhost:5000`, and React at `localhost:3000`. The backend `.env` must include `NLP_SERVICE_URL=http://127.0.0.1:8001`. FastAPI must have the local BERT checkpoint and tokenizer under `nlp_service/artifacts/bert/`. The checkpoint is a large local artifact and is not included in the source patch ZIP.

Backend tests: `cd backend && npm test`. Python NLP tests: `cd nlp_service && python -m pytest -q`. Frontend build: `cd frontend && npm run build`. The new backend integration tests use a local stub of FastAPI for deterministic coverage; a real local upload should also be tested against the actual trained BERT service and representative subject PDFs.
