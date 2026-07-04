import React, { useState, useEffect } from 'react';
import { getMyTests, deletePDF, getPDFs } from '../api';
import styles from './MyTests.module.css';

export default function MyTests({ navigate, pdfs }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMyTests()
      .then(d => setResults(d.results || []))
      .catch(() => setResults([]))
      .finally(() => setLoading(false));
  }, []);

  const fmt = s => `${Math.floor(s / 60)}m ${s % 60}s`;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h2 className={styles.title}>My Tests & PDFs</h2>
        <p className={styles.sub}>Your uploaded PDFs and past test history.</p>
      </div>

      {/* PDFs */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Uploaded PDFs ({pdfs.length})</h3>
          <button className={styles.uploadBtn} onClick={() => navigate('upload')}>+ Upload PDF</button>
        </div>
        {pdfs.length === 0 ? (
          <div className={styles.emptyBox}>
            <div className={styles.emptyIcon}>📂</div>
            <div className={styles.emptyTitle}>No PDFs yet</div>
            <div className={styles.emptySub}>Upload a study PDF to start generating questions</div>
            <button className={styles.primaryBtn} onClick={() => navigate('upload')}>Upload PDF</button>
          </div>
        ) : (
          <div className={styles.tableWrap}>
            <div className={styles.tableHead}>
              <div>File</div><div>Subject</div><div>Pages</div><div>Uploaded</div><div>Action</div>
            </div>
            {pdfs.map(p => (
              <div key={p.id} className={styles.tableRow}>
                <div className={styles.fileCell}>
                  <span className={styles.fileIcon}>📄</span>
                  <div>
                    <div className={styles.fileName}>{p.name.length > 42 ? p.name.slice(0, 42) + '…' : p.name}</div>
                    <div className={styles.fileMeta}>{p.chapter}</div>
                  </div>
                </div>
                <div><span className={styles.subjectBadge}>{p.subject}</span></div>
                <div className={styles.pages}>{p.totalPages}</div>
                <div className={styles.date}>{new Date(p.uploadedAt).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}</div>
                <div>
                  <button className={styles.genBtn} onClick={() => navigate('generate')}>Generate →</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Test history */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Test History</h3>
        {loading ? (
          <div className={styles.loading}><span className="spin">⟳</span> Loading…</div>
        ) : results.length === 0 ? (
          <div className={styles.emptyBox}>
            <div className={styles.emptyIcon}>📝</div>
            <div className={styles.emptyTitle}>No tests yet</div>
            <div className={styles.emptySub}>Generate and complete a test to see your history here</div>
            <button className={styles.primaryBtn} onClick={() => navigate('generate')}>Generate Test</button>
          </div>
        ) : (
          <div className={styles.tableWrap}>
            <div className={styles.tableHeadTest}>
              <div>PDF / Template</div><div>Score</div><div>Correct</div><div>Wrong</div><div>Time</div><div>Date</div>
            </div>
            {results.map(r => {
              const pct = r.percentage;
              const color = pct >= 70 ? '#15803d' : pct >= 50 ? '#92400e' : '#991b1b';
              return (
                <div key={r.id} className={styles.tableRowTest}>
                  <div>
                    <div className={styles.testPdf}>{r.pdf_name || 'Unknown PDF'}</div>
                    <div className={styles.testTemplate}>{r.exam_template} · Pages {r.page_range}</div>
                  </div>
                  <div style={{ fontWeight: 700, color }}>
                    {r.raw_score?.toFixed(1)}<span style={{ color: 'var(--gray-400)', fontWeight: 400 }}>/{r.max_score}</span>
                    <div style={{ fontSize: 11, color }}>{pct}%</div>
                  </div>
                  <div style={{ color: '#15803d', fontWeight: 600 }}>{r.correct}</div>
                  <div style={{ color: '#991b1b', fontWeight: 600 }}>{r.wrong}</div>
                  <div style={{ fontSize: 12, color: 'var(--gray-400)' }}>{fmt(r.time_taken || 0)}</div>
                  <div style={{ fontSize: 12, color: 'var(--gray-400)' }}>
                    {new Date(r.created_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Info cards */}
      <div className={styles.infoSection}>
        {[
          { icon: '📖', title: 'Page-range questions', text: 'When generating a test, specify exact page numbers. AI reads ONLY those pages.' },
          { icon: '✦', title: 'AI-powered accuracy', text: 'Every question includes a page reference and explanation to verify from your PDF.' },
          { icon: '📐', title: 'Exam marking schemes', text: 'Railway, SSC, UPSC, Banking — accurate negative marking on every submission.' },
        ].map(c => (
          <div key={c.title} className={styles.infoCard}>
            <div className={styles.infoIcon}>{c.icon}</div>
            <div className={styles.infoTitle}>{c.title}</div>
            <div className={styles.infoText}>{c.text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
