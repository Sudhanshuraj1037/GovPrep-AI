import React, { useState, useEffect, useCallback, useRef } from 'react';
import { submitTest } from '../api';
import styles from './TakeTest.module.css';

export default function TakeTest({ navigate, testData }) {
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState({});
  const [flagged, setFlagged] = useState({});
  const [timeLeft, setTimeLeft] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const submittedRef = useRef(false);

  const questions = testData?.questions || [];
  const totalTime = Math.max(questions.length * 90, 600);

  useEffect(() => { if (testData) setTimeLeft(totalTime); }, [testData, totalTime]);

  const doSubmit = useCallback(async () => {
    if (submittedRef.current) return;
    submittedRef.current = true;
    setSubmitting(true);
    const timeTaken = totalTime - (timeLeft || 0);
    try {
      const data = await submitTest({ testId: testData.testId, answers, timeTaken, examTemplate: testData.examTemplate });
      navigate('results', data.scorecard);
    } catch (err) {
      alert('Submission error: ' + err.message);
      submittedRef.current = false;
      setSubmitting(false);
    }
  }, [answers, timeLeft, totalTime, testData, navigate]);

  useEffect(() => {
    if (timeLeft === null) return;
    if (timeLeft <= 0) { doSubmit(); return; }
    const t = setTimeout(() => setTimeLeft(p => p - 1), 1000);
    return () => clearTimeout(t);
  }, [timeLeft, doSubmit]);

  if (!testData || !questions.length) return (
    <div className={styles.empty}>
      <p>No test data. Please generate a test first.</p>
      <button className={styles.btn} onClick={() => navigate('generate')}>Generate Test</button>
    </div>
  );

  const q = questions[current];
  const answered = Object.keys(answers).length;
  const flaggedCount = Object.values(flagged).filter(Boolean).length;
  const fmt = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;

  return (
    <div className={styles.page}>
      {/* Top bar */}
      <div className={styles.bar}>
        <div>
          <div className={styles.barTitle}>{testData.examTemplate}</div>
          <div className={styles.barSub}>📄 {testData.meta?.pdfName} · Pages {testData.meta?.pageRange}</div>
        </div>
        <span className={styles.progress}>{answered}/{questions.length} answered</span>
        <div className={`${styles.timer} ${timeLeft !== null && timeLeft < 300 ? styles.timerRed : timeLeft < 600 ? styles.timerAmber : ''}`}>
          ⏱ {timeLeft !== null ? fmt(timeLeft) : '--:--'}
        </div>
      </div>

      <div className={styles.layout}>
        {/* Question panel */}
        <div className={styles.qPanel}>
          <div className={styles.qMeta}>
            <span className={styles.qNum}>Q{current + 1} / {questions.length}</span>
            <span className={`${styles.diff} ${styles['diff_' + q.difficulty]}`}>{q.difficulty}</span>
            <span className={styles.src}>📄 Pg {q.pageRef}</span>
            <button className={`${styles.flagBtn} ${flagged[q.id] ? styles.flagged : ''}`} onClick={() => setFlagged(p => ({ ...p, [q.id]: !p[q.id] }))}>
              {flagged[q.id] ? '🚩 Flagged' : '⚑ Flag'}
            </button>
          </div>
          <div className={styles.qText}>{q.question}</div>
          <div className={styles.options}>
            {q.options.map((opt, oi) => (
              <button key={oi} className={`${styles.option} ${answers[q.id] === oi ? styles.selected : ''}`} onClick={() => setAnswers(p => ({ ...p, [q.id]: oi }))}>
                <span className={styles.optKey}>{['A','B','C','D'][oi]}</span>
                <span>{opt}</span>
                {answers[q.id] === oi && <span className={styles.tick}>✓</span>}
              </button>
            ))}
          </div>
          <div className={styles.navBtns}>
            <button className={styles.navBtn} disabled={current === 0} onClick={() => setCurrent(c => c - 1)}>← Prev</button>
            <button className={styles.clearBtn} disabled={answers[q.id] === undefined} onClick={() => setAnswers(p => { const n = {...p}; delete n[q.id]; return n; })}>Clear</button>
            {current < questions.length - 1
              ? <button className={styles.nextBtn} onClick={() => setCurrent(c => c + 1)}>Next →</button>
              : <button className={styles.submitBtn} onClick={() => setShowConfirm(true)}>Submit ✓</button>}
          </div>
        </div>

        {/* Side panel */}
        <div className={styles.side}>
          <div className={styles.navPanel}>
            <div className={styles.navTitle}>Navigator</div>
            <div className={styles.navGrid}>
              {questions.map((_, i) => {
                const qid = questions[i].id;
                return <button key={i} className={`${styles.navQ} ${i === current ? styles.navCurrent : flagged[qid] ? styles.navFlagged : answers[qid] !== undefined ? styles.navAnswered : ''}`} onClick={() => setCurrent(i)}>{i+1}</button>;
              })}
            </div>
            <div className={styles.legend}>
              <span><span className={styles.dot} style={{background:'var(--green)'}}/>Answered</span>
              <span><span className={styles.dot} style={{background:'var(--amber)'}}/>Flagged</span>
            </div>
          </div>
          <div className={styles.summaryPanel}>
            <div className={styles.sumRow}><span>Answered</span><strong style={{color:'var(--green)'}}>{answered}</strong></div>
            <div className={styles.sumRow}><span>Remaining</span><strong>{questions.length - answered}</strong></div>
            <div className={styles.sumRow}><span>Flagged</span><strong style={{color:'var(--amber)'}}>{flaggedCount}</strong></div>
            <button className={styles.submitFull} onClick={() => setShowConfirm(true)} disabled={submitting}>
              {submitting ? 'Submitting…' : 'Submit Test'}
            </button>
          </div>
        </div>
      </div>

      {showConfirm && (
        <div className={styles.overlay} onClick={() => setShowConfirm(false)}>
          <div className={styles.modal} onClick={e => e.stopPropagation()}>
            <div className={styles.modalTitle}>Submit Test?</div>
            <p>You've answered <strong>{answered}</strong> of <strong>{questions.length}</strong> questions.</p>
            {questions.length - answered > 0 && <p className={styles.warn}>⚠️ {questions.length - answered} unattempted.</p>}
            <div className={styles.modalBtns}>
              <button className={styles.cancelBtn} onClick={() => setShowConfirm(false)}>Continue</button>
              <button className={styles.confirmBtn} onClick={() => { setShowConfirm(false); doSubmit(); }}>Yes, Submit</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
