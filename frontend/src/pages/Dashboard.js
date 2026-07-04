import React from 'react';
import styles from './Dashboard.module.css';

const EXAM_TEMPLATES = [
  { name: 'Railway RRB NTPC', q: 100, time: '90 min', neg: '-0.33', color: '#2563eb' },
  { name: 'SSC CGL Tier 1', q: 100, time: '60 min', neg: '-0.5', color: '#16a34a' },
  { name: 'UPSC Prelims', q: 100, time: '120 min', neg: '-0.66', color: '#7c3aed' },
  { name: 'Banking IBPS PO', q: 100, time: '60 min', neg: '-0.25', color: '#d97706' },
];

export default function Dashboard({ navigate, pdfs, user }) {
  const hour = new Date().getHours();
  const greeting = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';

  return (
    <div className={styles.page}>
      <div className={styles.welcome}>
        <div>
          <h2 className={styles.welcomeTitle}>{greeting}, {user?.name || 'Aspirant'} 👋</h2>
          <p className={styles.welcomeSub}>Target: {user?.examTarget || 'Railway RRB NTPC'} · Keep going, you're doing great!</p>
        </div>
        <div className={styles.welcomeTag}>🔥 Stay consistent</div>
      </div>

      {/* Stats */}
      <div className={styles.statsRow}>
        {[
          { label: 'PDFs Uploaded', value: pdfs.length, sub: 'Ready to generate', color: 'var(--green)' },
          { label: 'Questions/Test', value: '5–20', sub: 'Per generation', color: 'var(--blue)' },
          { label: 'Exam Templates', value: '4', sub: 'Railway, SSC, UPSC, Bank', color: 'var(--purple)' },
          { label: 'AI Assistant', value: '24/7', sub: 'Ask anything', color: 'var(--amber)' },
        ].map(s => (
          <div key={s.label} className={styles.statCard}>
            <div className={styles.statVal} style={{ color: s.color }}>{s.value}</div>
            <div className={styles.statLabel}>{s.label}</div>
            <div className={styles.statSub}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Quick actions */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Quick Actions</h3>
        <div className={styles.actions}>
          <button className={styles.actionBtn} onClick={() => navigate('upload')}>
            <span className={styles.actionIcon}>📄</span>
            <div>
              <div className={styles.actionLabel}>Upload PDF</div>
              <div className={styles.actionSub}>Add study material</div>
            </div>
          </button>
          <button className={styles.actionBtn} onClick={() => navigate('generate')}>
            <span className={styles.actionIcon}>✦</span>
            <div>
              <div className={styles.actionLabel}>Generate Test</div>
              <div className={styles.actionSub}>From your PDFs with page range</div>
            </div>
          </button>
          <button className={styles.actionBtn} onClick={() => navigate('my-tests')}>
            <span className={styles.actionIcon}>📊</span>
            <div>
              <div className={styles.actionLabel}>My Tests</div>
              <div className={styles.actionSub}>View past results</div>
            </div>
          </button>
        </div>
      </div>

      {/* PDFs */}
      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <h3 className={styles.sectionTitle}>Your PDFs ({pdfs.length})</h3>
          <button className={styles.linkBtn} onClick={() => navigate('upload')}>Upload new →</button>
        </div>
        {pdfs.length === 0 ? (
          <div className={styles.emptyBox}>
            <div className={styles.emptyIcon}>📂</div>
            <div className={styles.emptyTitle}>No PDFs yet</div>
            <div className={styles.emptySub}>Upload a PDF to start generating questions from specific page ranges</div>
            <button className={styles.primaryBtn} onClick={() => navigate('upload')}>Upload PDF</button>
          </div>
        ) : (
          <div className={styles.pdfGrid}>
            {pdfs.map(p => (
              <div key={p.id} className={styles.pdfCard}>
                <div className={styles.pdfIcon}>📄</div>
                <div className={styles.pdfInfo}>
                  <div className={styles.pdfName}>{p.name.length > 38 ? p.name.slice(0, 38) + '…' : p.name}</div>
                  <div className={styles.pdfMeta}>{p.subject} · {p.chapter} · {p.totalPages} pages</div>
                </div>
                <button className={styles.genBtn} onClick={() => navigate('generate')}>Generate →</button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Exam templates */}
      <div className={styles.section}>
        <h3 className={styles.sectionTitle}>Exam Templates</h3>
        <div className={styles.templateGrid}>
          {EXAM_TEMPLATES.map(t => (
            <div key={t.name} className={styles.templateCard}>
              <div className={styles.templateDot} style={{ background: t.color }} />
              <div className={styles.templateName}>{t.name}</div>
              <div className={styles.templateStats}>
                <span>{t.q} questions</span>
                <span>{t.time}</span>
                <span style={{ color: 'var(--red)' }}>{t.neg} wrong</span>
              </div>
              <button className={styles.startBtn} onClick={() => navigate('generate')} style={{ borderColor: t.color, color: t.color }}>
                Start →
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* How it works */}
      <div className={styles.howItWorks}>
        <h3 className={styles.sectionTitle}>How Page-Range Questions Work</h3>
        <div className={styles.steps}>
          {[
            { n: '1', title: 'Upload PDF', desc: 'Upload any textbook chapter, notes, or practice material.' },
            { n: '2', title: 'Choose Pages', desc: 'Set page range (e.g. 5–20). AI reads ONLY those pages.' },
            { n: '3', title: 'AI Generates', desc: 'Claude/Groq AI creates MCQs strictly from your selected content.' },
            { n: '4', title: 'Test & Review', desc: 'Take timed test, get scorecard with marking scheme applied.' },
          ].map(s => (
            <div key={s.n} className={styles.step}>
              <div className={styles.stepNum}>{s.n}</div>
              <div className={styles.stepTitle}>{s.title}</div>
              <div className={styles.stepDesc}>{s.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* AI Assistant hint */}
      <div className={styles.assistantHint}>
        <span className={styles.assistantIcon}>💬</span>
        <div>
          <div className={styles.assistantTitle}>AI Study Assistant is ready!</div>
          <div className={styles.assistantSub}>Click the green chat button (bottom-right) to ask doubts, get exam tips, or explain any concept.</div>
        </div>
      </div>
    </div>
  );
}
