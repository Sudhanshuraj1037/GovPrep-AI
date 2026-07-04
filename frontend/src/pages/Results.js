import React, { useState } from 'react';
import styles from './Results.module.css';

export default function Results({ navigate, scorecard }) {
  const [showReview, setShowReview] = useState(false);
  if (!scorecard) return <div className={styles.empty}><div>No results found.</div><button className={styles.btn} onClick={() => navigate('generate')}>Start a test</button></div>;

  const { examTemplate, correct, wrong, unattempted, rawScore, maxScore, percentage, timeTaken, scheme, topicBreakdown, questionResults, recommendation, pdfName, pageRange } = scorecard;
  const fmt = s => `${Math.floor(s/60)}m ${s%60}s`;
  const grade = percentage >= 80 ? { label: 'Excellent 🏆', color: '#15803d', bg: '#dcfce7' } : percentage >= 65 ? { label: 'Good 👍', color: '#1d4ed8', bg: '#dbeafe' } : percentage >= 50 ? { label: 'Average 📚', color: '#92400e', bg: '#fef3c7' } : { label: 'Needs Work 💪', color: '#991b1b', bg: '#fee2e2' };

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.gradeTag} style={{ background: grade.bg, color: grade.color }}>{grade.label}</div>
        <div className={styles.examName}>{examTemplate}</div>
        <div className={styles.pdfRef}>📄 {pdfName} · Pages {pageRange}</div>
        <div className={styles.scoreBig}><span className={styles.scoreNum} style={{ color: grade.color }}>{Math.max(0, rawScore).toFixed(1)}</span><span className={styles.scoreDenom}>/ {maxScore}</span></div>
        <div className={styles.pct} style={{ color: grade.color }}>{percentage}% Score</div>
      </div>

      <div className={styles.statsRow}>
        {[['Correct ✓', correct, '#15803d', `+${scheme?.correct || 1} each`], ['Wrong ✗', wrong, '#991b1b', `${scheme?.wrong} each`], ['Skipped', unattempted, 'var(--gray-400)', 'No penalty'], ['Time', fmt(timeTaken), 'var(--gray-700)', 'Total']].map(([l, v, c, s]) => (
          <div key={l} className={styles.statBox}>
            <div className={styles.statVal} style={{ color: c }}>{v}</div>
            <div className={styles.statLabel}>{l}</div>
            <div className={styles.statSub}>{s}</div>
          </div>
        ))}
      </div>

      {/* Marking scheme */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>📐 Marking Scheme</div>
        <div className={styles.scheme}>
          <div className={styles.schemeRow}><span>Correct</span><span>{correct} × +{scheme?.correct}</span><span className={styles.pos}>+{(correct*(scheme?.correct||1)).toFixed(2)}</span></div>
          <div className={styles.schemeRow}><span>Wrong</span><span>{wrong} × {scheme?.wrong}</span><span className={styles.neg}>{(wrong*(scheme?.wrong||0)).toFixed(2)}</span></div>
          <div className={styles.schemeRow}><span>Unattempted</span><span>{unattempted} × 0</span><span>0</span></div>
          <div className={styles.schemeDivider}/>
          <div className={`${styles.schemeRow} ${styles.schemeTotal}`}><span>Raw Score</span><span/><span style={{color:grade.color}}>{Math.max(0,rawScore).toFixed(2)} / {maxScore}</span></div>
        </div>
      </div>

      {/* Topic breakdown */}
      {topicBreakdown?.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardTitle}>📚 Topic Performance</div>
          {topicBreakdown.map(t => {
            const c = t.percentage >= 70 ? 'var(--green)' : t.percentage >= 50 ? 'var(--amber)' : 'var(--red)';
            return <div key={t.topic} className={styles.topicRow}><div className={styles.topicName}>{t.topic}</div><div className={styles.topicBar}><div className={styles.topicFill} style={{width:t.percentage+'%',background:c}}/></div><div className={styles.topicSc} style={{color:c}}>{t.correct}/{t.total}</div><div className={styles.topicPct} style={{color:c}}>{t.percentage}%</div></div>;
          })}
        </div>
      )}

      {/* Recommendation */}
      <div className={styles.rec}><div className={styles.recTitle}>🤖 AI Recommendation</div><p className={styles.recText}>{recommendation}</p><div className={styles.recHint}>💬 Ask the AI Assistant (bottom-right) to explain any concept or wrong answer in detail.</div></div>

      <div className={styles.actions}>
        <button className={styles.primaryBtn} onClick={() => navigate('generate')}>🔄 New Test</button>
        <button className={styles.reviewBtn} onClick={() => setShowReview(v => !v)}>{showReview ? 'Hide' : '📋 Review'} Questions</button>
        <button className={styles.backBtn} onClick={() => navigate('dashboard')}>← Dashboard</button>
      </div>

      {showReview && questionResults && (
        <div className={styles.reviewSection}>
          <div className={styles.reviewTitle}>Question Review</div>
          {questionResults.map((q, i) => (
            <div key={q.id} className={`${styles.qCard} ${q.status === 'correct' ? styles.qCorrect : q.status === 'wrong' ? styles.qWrong : styles.qSkip}`}>
              <div className={styles.qCardHeader}>
                <span className={styles.qNum}>Q{i+1}</span>
                <span className={`${styles.badge} ${q.status === 'correct' ? styles.bCorrect : q.status === 'wrong' ? styles.bWrong : styles.bSkip}`}>{q.status === 'correct' ? '✓ Correct' : q.status === 'wrong' ? '✗ Wrong' : '– Skipped'}</span>
                <span className={styles.qtopic}>{q.topic}</span>
                <span className={styles.qpage}>Pg {q.pageRef}</span>
              </div>
              <div className={styles.qText}>{q.question}</div>
              <div className={styles.opts}>
                {q.options.map((o, oi) => (
                  <div key={oi} className={`${styles.opt} ${oi === q.correct ? styles.optCorrect : oi === q.userAnswer && oi !== q.correct ? styles.optWrong : ''}`}>
                    <span className={styles.optKey}>{['A','B','C','D'][oi]}</span>
                    <span className={styles.optTxt}>{o}</span>
                    {oi === q.correct && <span className={styles.correctMark}>✓ Correct</span>}
                    {oi === q.userAnswer && oi !== q.correct && <span className={styles.wrongMark}>✗ Yours</span>}
                  </div>
                ))}
              </div>
              {q.explanation && <div className={styles.expl}><strong>💡</strong> {q.explanation}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
