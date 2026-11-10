import React, { useState, useRef } from 'react';
import { uploadPDF, confirmPDFSubject } from '../api';
import styles from './UploadPDF.module.css';

const SUBJECTS = ['History', 'Geography', 'Mathematics', 'Science', 'Reasoning', 'General Awareness', 'English', 'Economics', 'Polity', 'Current Affairs', 'Other'];

const statusLabel = (status) => ({
  awaiting_confirmation: 'Subject detected — confirmation required',
  manual_required: 'Manual subject selection required',
  confirmed: 'Detected subject confirmed',
  manual_confirmed: 'Subject selected manually',
}[status] || 'Subject status unavailable');

const reasonLabel = (reason) => ({
  nlp_service_not_configured: 'The NLP service is not configured. You can still select the subject manually.',
  nlp_service_unavailable: 'The NLP service could not be reached. Your PDF is safe; select the subject manually.',
  nlp_service_timeout: 'Subject detection timed out. Your PDF is safe; select the subject manually.',
  not_trained_yet: 'The BERT model is not available for inference. Select the subject manually.',
  dependency_missing: 'The NLP service is missing a required inference dependency. Select the subject manually.',
  insufficient_extractable_text: 'Not enough selectable text was extracted. This may be a scanned PDF; OCR is not enabled yet.',
  insufficient_classifiable_chunks: 'The PDF did not contain enough readable chunks for a reliable prediction.',
  low_subject_score: 'The model score was below the automatic-confirmation threshold.',
  ambiguous_subject_scores: 'The model could not clearly distinguish between the top subjects.',
  mixed_or_inconsistent_document: 'Different sections appear to cover different subjects.',
  invalid_classifier_response: 'The NLP service returned an unreadable response.',
  classification_failed: 'Subject detection failed. You can select the subject manually.',
}[reason] || 'Automatic subject detection could not be completed. You can select the subject manually.');

export default function UploadPDF({ navigate, onUpload }) {
  const [file, setFile] = useState(null);
  const [subjectMode, setSubjectMode] = useState('auto');
  const [subject, setSubject] = useState('History');
  const [resultSubject, setResultSubject] = useState('Other');
  const [chapter, setChapter] = useState('');
  const [uploading, setUploading] = useState(false);
  const [savingSubject, setSavingSubject] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [drag, setDrag] = useState(false);
  const fileRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    if (f.type !== 'application/pdf' && !f.name.toLowerCase().endsWith('.pdf')) { setError('Only PDF files are allowed.'); return; }
    if (f.size > 50 * 1024 * 1024) { setError('File too large. Max 50MB.'); return; }
    setFile(f);
    setError('');
    if (!chapter) setChapter(f.name.replace(/\.pdf$/i, '').replace(/[_-]/g, ' '));
  };

  const handleUpload = async () => {
    if (!file) { setError('Please select a PDF.'); return; }
    if (!chapter.trim()) { setError('Please enter a chapter/title name.'); return; }
    setUploading(true); setError(''); setProgress(20);
    const formData = new FormData();
    formData.append('pdf', file);
    formData.append('subjectMode', subjectMode);
    if (subjectMode === 'manual') formData.append('subject', subject);
    formData.append('chapter', chapter.trim());
    try {
      setProgress(45);
      const data = await uploadPDF(formData);
      setProgress(100);
      setResult(data.pdf);
      setResultSubject(data.pdf.subjectDetected || data.pdf.subject || 'Other');
      onUpload && onUpload();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleConfirmSubject = async (value) => {
    if (!result?.id) return;
    setSavingSubject(true); setError('');
    try {
      const data = await confirmPDFSubject(result.id, value);
      setResult(data.pdf);
      setResultSubject(data.pdf.subject || value);
      onUpload && onUpload();
    } catch (err) {
      setError(err.message);
    } finally {
      setSavingSubject(false);
    }
  };

  const canGenerate = Boolean(
    result &&
    ['confirmed', 'manual_confirmed'].includes(result.subjectDetectionStatus || 'manual_confirmed')
  );

  if (result) return (
    <div className={styles.page}>
      <div className={styles.successCard}>
        <div className={styles.successIcon}>✅</div>
        <h2 className={styles.successTitle}>PDF Uploaded!</h2>
        <div className={styles.details}>
          {[['File', result.name], ['Subject', result.subject], ['Chapter', result.chapter], ['Total pages', result.totalPages], ['Extracted words', result.wordCount?.toLocaleString()]].map(([k, v]) => (
            <div key={k} className={styles.detailRow}><span>{k}</span><strong>{v}</strong></div>
          ))}
        </div>

        <div className={styles.detectionCard}>
          <div className={styles.detectionHeading}>Automatic subject detection</div>
          <div className={styles.detectionStatus}>{statusLabel(result.subjectDetectionStatus)}</div>
          {result.subjectDetected && (
            <div className={styles.detectedSubject}>
              <span>Suggested subject</span>
              <strong>{result.subjectDetected}</strong>
            </div>
          )}
          {result.subjectDetectionModel && <div className={styles.detectionMeta}>Model: {result.subjectDetectionModel.toUpperCase()}</div>}
          {Number.isFinite(result.subjectDetectionScore) && result.subjectDetectionScore > 0 && (
            <div className={styles.detectionMeta}>Average model score: {(result.subjectDetectionScore * 100).toFixed(1)}% · Not a calibrated probability</div>
          )}
          {result.subjectDetectionReason && result.subjectDetectionStatus === 'manual_required' && (
            <div className={styles.detectionReason}>{reasonLabel(result.subjectDetectionReason)}</div>
          )}
          {['awaiting_confirmation', 'manual_required'].includes(result.subjectDetectionStatus) && (
            <div className={styles.subjectConfirmForm}>
              <label className={styles.label}>{result.subjectDetected ? 'Confirm or change the subject' : 'Select the PDF subject'}</label>
              <select className={styles.select} value={resultSubject} onChange={e => setResultSubject(e.target.value)}>
                {SUBJECTS.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
              <button className={styles.primaryBtn} onClick={() => handleConfirmSubject(resultSubject)} disabled={savingSubject}>
                {savingSubject ? 'Saving…' : result.subjectDetected && resultSubject === result.subjectDetected ? 'Confirm detected subject' : 'Save selected subject'}
              </button>
            </div>
          )}
          {canGenerate && <div className={styles.readyNote}>✓ Subject confirmed. You can now generate questions from this PDF.</div>}
        </div>

        {error && <div className={styles.error}>⚠️ {error}</div>}
        <div className={styles.successActions}>
          <button className={styles.primaryBtn} onClick={() => navigate('generate')} disabled={!canGenerate}>Generate Questions from this PDF →</button>
          {!canGenerate && <div className={styles.fieldNote}>Confirm or select a subject before generating questions.</div>}
          <button className={styles.secondaryBtn} onClick={() => { setFile(null); setResult(null); setChapter(''); setProgress(0); setError(''); setSubjectMode('auto'); setSubject('History'); }}>Upload another PDF</button>
        </div>
      </div>
    </div>
  );

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h2 className={styles.title}>Upload Study Material</h2>
        <p className={styles.sub}>Upload a PDF. GovPrep-AI will extract its text and try to detect the subject before question generation.</p>
      </div>
      <div className={styles.card}>
        <div
          className={`${styles.dropZone} ${drag ? styles.dragging : ''} ${file ? styles.hasFile : ''}`}
          onDragOver={e => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}
          onClick={() => fileRef.current.click()}
        >
          <input ref={fileRef} type="file" accept=".pdf,application/pdf" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />
          {file ? (
            <><div className={styles.fileIcon}>📄</div><div className={styles.fileName}>{file.name}</div><div className={styles.fileSize}>{(file.size / 1024 / 1024).toFixed(2)} MB · Click to change</div></>
          ) : (
            <><div className={styles.uploadIcon}>☁️</div><div className={styles.dropTitle}>Drop PDF here or click to browse</div><div className={styles.dropSub}>PDF files up to 50MB · Text-based PDFs work best</div></>
          )}
        </div>
        <div className={styles.form}>
          <div className={styles.formRow}>
            <label className={styles.label}>Subject selection <span className={styles.req}>*</span></label>
            <select className={styles.select} value={subjectMode} onChange={e => setSubjectMode(e.target.value)}>
              <option value="auto">Auto-detect subject using BERT</option>
              <option value="manual">I'll select the subject manually</option>
            </select>
          </div>
          {subjectMode === 'manual' && (
            <div className={styles.formRow}>
              <label className={styles.label}>Subject</label>
              <select className={styles.select} value={subject} onChange={e => setSubject(e.target.value)}>
                {SUBJECTS.map(s => <option key={s}>{s}</option>)}
              </select>
            </div>
          )}
          <div className={styles.formRow}>
            <label className={styles.label}>Chapter / Title <span className={styles.req}>*</span></label>
            <input className={styles.input} value={chapter} onChange={e => setChapter(e.target.value)} placeholder="e.g. Mughal Empire, Number System, Chapter 5" />
          </div>
        </div>
        {error && <div className={styles.error}>⚠️ {error}</div>}
        {uploading && (
          <div className={styles.progressBox}>
            <div className={styles.progressLabel}>{subjectMode === 'auto' ? 'Uploading PDF, extracting text and detecting subject…' : 'Uploading and extracting PDF text…'}</div>
            <div className={styles.progressTrack}><div className={styles.progressFill} style={{ width: progress + '%' }} /></div>
          </div>
        )}
        <button className={styles.primaryBtn} onClick={handleUpload} disabled={uploading || !file}>
          {uploading ? <><span className="spin">⟳</span> Processing…</> : '↑ Upload PDF'}
        </button>
        <div className={styles.tips}>
          <div className={styles.tipTitle}>💡 Tips</div>
          <ul className={styles.tipList}>
            <li>Auto-detection currently supports History, Geography, Polity and Science.</li>
            <li>Text-based PDFs work best. Scanned PDFs need OCR, which is not enabled yet.</li>
            <li>Mixed-subject or uncertain PDFs will ask you to select a subject manually.</li>
            <li>Questions are generated from the selected PDF pages after subject confirmation.</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
