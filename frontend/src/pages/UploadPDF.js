import React, { useState, useRef } from 'react';
import { uploadPDF } from '../api';
import styles from './UploadPDF.module.css';

const SUBJECTS = ['History','Geography','Mathematics','Science','Reasoning','General Awareness','English','Economics','Polity','Current Affairs','Other'];

export default function UploadPDF({ navigate, onUpload }) {
  const [file, setFile] = useState(null);
  const [subject, setSubject] = useState('History');
  const [chapter, setChapter] = useState('');
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [drag, setDrag] = useState(false);
  const fileRef = useRef();

  const handleFile = (f) => {
    if (!f) return;
    if (f.type !== 'application/pdf') { setError('Only PDF files are allowed.'); return; }
    if (f.size > 50 * 1024 * 1024) { setError('File too large. Max 50MB.'); return; }
    setFile(f);
    setError('');
    if (!chapter) setChapter(f.name.replace('.pdf', '').replace(/[_-]/g, ' '));
  };

  const handleUpload = async () => {
    if (!file) { setError('Please select a PDF.'); return; }
    if (!chapter.trim()) { setError('Please enter a chapter/title name.'); return; }
    setUploading(true); setError(''); setProgress(20);
    const formData = new FormData();
    formData.append('pdf', file);
    formData.append('subject', subject);
    formData.append('chapter', chapter.trim());
    try {
      setProgress(50);
      const data = await uploadPDF(formData);
      setProgress(100);
      setResult(data.pdf);
      onUpload && onUpload();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  if (result) return (
    <div className={styles.page}>
      <div className={styles.successCard}>
        <div className={styles.successIcon}>✅</div>
        <h2 className={styles.successTitle}>PDF Uploaded!</h2>
        <div className={styles.details}>
          {[['File', result.name], ['Subject', result.subject], ['Chapter', result.chapter], ['Total pages', result.totalPages], ['Word count', result.wordCount?.toLocaleString()]].map(([k, v]) => (
            <div key={k} className={styles.detailRow}><span>{k}</span><strong>{v}</strong></div>
          ))}
        </div>
        <div className={styles.successActions}>
          <button className={styles.primaryBtn} onClick={() => navigate('generate')}>Generate Questions from this PDF →</button>
          <button className={styles.secondaryBtn} onClick={() => { setFile(null); setResult(null); setChapter(''); setProgress(0); }}>Upload another PDF</button>
        </div>
      </div>
    </div>
  );

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h2 className={styles.title}>Upload Study Material</h2>
        <p className={styles.sub}>Upload any PDF — textbook chapter, notes, practice set. Then generate questions from specific page ranges.</p>
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
            <><div className={styles.fileIcon}>📄</div><div className={styles.fileName}>{file.name}</div><div className={styles.fileSize}>{(file.size/1024/1024).toFixed(2)} MB · Click to change</div></>
          ) : (
            <><div className={styles.uploadIcon}>☁️</div><div className={styles.dropTitle}>Drop PDF here or click to browse</div><div className={styles.dropSub}>PDF files up to 50MB · Text-based PDFs work best</div></>
          )}
        </div>
        <div className={styles.form}>
          <div className={styles.formRow}>
            <label className={styles.label}>Subject <span className={styles.req}>*</span></label>
            <select className={styles.select} value={subject} onChange={e => setSubject(e.target.value)}>
              {SUBJECTS.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
          <div className={styles.formRow}>
            <label className={styles.label}>Chapter / Title <span className={styles.req}>*</span></label>
            <input className={styles.input} value={chapter} onChange={e => setChapter(e.target.value)} placeholder="e.g. Mughal Empire, Number System, Chapter 5" />
          </div>
        </div>
        {error && <div className={styles.error}>⚠️ {error}</div>}
        {uploading && (
          <div className={styles.progressBox}>
            <div className={styles.progressLabel}>Uploading and parsing PDF…</div>
            <div className={styles.progressTrack}><div className={styles.progressFill} style={{ width: progress + '%' }} /></div>
          </div>
        )}
        <button className={styles.primaryBtn} onClick={handleUpload} disabled={uploading || !file}>
          {uploading ? <><span className="spin">⟳</span> Processing…</> : '↑ Upload PDF'}
        </button>
        <div className={styles.tips}>
          <div className={styles.tipTitle}>💡 Tips</div>
          <ul className={styles.tipList}>
            <li>Text-based PDFs (not scanned images) work best</li>
            <li>Smaller focused chapters (10–30 pages) give better questions</li>
            <li>After upload, you can select any page range to generate from</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
