GovPrep-AI Phase 13A PDF → NLP → Question Generation Patch

This is an OVERLAY PATCH, not a full project backup.
Extract the ZIP into the existing project root:
  D:\Projects\gp2

Allow the listed source/documentation files to be overwritten. The patch does NOT contain or replace:
- backend/.env or any other real .env file
- backend/govprep-db.json / backend/govprep.db
- backend/uploads/
- node_modules or Python virtual environments
- trained model checkpoints (*.pt), including artifacts/bert/subject.pt

Your locally trained BERT checkpoint and tokenizer must remain at:
  nlp_service\artifacts\bert\subject.pt
  nlp_service\artifacts\bert\tokenizer\

The patch changes the existing React upload flow and Generate Test flow, backend PDF metadata handling,
BERT document classification, owner-scoped subject confirmation, tests, and documentation.

After extracting:
1. Keep your existing backend/.env. It should contain:
   NLP_SERVICE_URL=http://127.0.0.1:8001
2. Start FastAPI, backend, and frontend in three terminals.
3. Run backend tests: cd backend; npm test
4. Run frontend build: cd frontend; npm run build
5. Try a text-based PDF. Auto detection is attempted by default; confirm/correct the subject before question generation.

Important limitations:
- The classifier currently supports only History, Geography, Polity, and Science.
- PDF-domain accuracy is not yet validated on a representative held-out PDF corpus.
- Low-confidence/mixed/unavailable cases require manual subject selection.
- OCR for scanned PDFs is not implemented.
