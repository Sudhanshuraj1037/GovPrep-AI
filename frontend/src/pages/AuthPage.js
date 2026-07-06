import React, { useState } from 'react';
import { login, signup } from '../api';
import styles from './AuthPage.module.css';

const EXAM_TARGETS = ['Railway RRB NTPC', 'SSC CGL Tier 1', 'UPSC Prelims', 'Banking IBPS PO', 'State PSC'];

export default function AuthPage({ onLogin, theme, toggleTheme }) {
  const [mode, setMode] = useState('login');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [examTarget, setExamTarget] = useState('Railway RRB NTPC');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPass, setShowPass] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const data = mode === 'login'
        ? await login(name.trim(), password)
        : await signup(name.trim(), password, examTarget);
      onLogin(data.token, data.user);
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        {/* Logo row + theme toggle */}
        <div className={styles.topBar}>
          <div className={styles.logo}>
            <div className={styles.logoIcon}>G</div>
            <div>
              <div className={styles.logoName}>GovPrep AI</div>
              <div className={styles.logoSub}>Exam Intelligence Platform</div>
            </div>
          </div>
          {toggleTheme && (
            <button className={styles.themeToggle} onClick={toggleTheme} title="Toggle theme">
              {theme === 'light' ? '🌙 Dark' : '☀️ Light'}
            </button>
          )}
        </div>

        {/* Tabs */}
        <div className={styles.tabs}>
          <button className={`${styles.tab} ${mode === 'login' ? styles.activeTab : ''}`} onClick={() => { setMode('login'); setError(''); }}>Login</button>
          <button className={`${styles.tab} ${mode === 'signup' ? styles.activeTab : ''}`} onClick={() => { setMode('signup'); setError(''); }}>Sign Up</button>
        </div>

        <form onSubmit={handleSubmit} className={styles.form}>
          <div className={styles.field}>
            <label className={styles.label}>Your Name</label>
            <input className={styles.input} type="text" placeholder="Enter your name (e.g. Priya)" value={name} onChange={e => setName(e.target.value)} required autoFocus />
          </div>
          <div className={styles.field}>
            <label className={styles.label}>Password</label>
            <div className={styles.passWrap}>
              <input className={styles.input} type={showPass ? 'text' : 'password'} placeholder={mode === 'signup' ? 'At least 4 characters' : 'Your password'} value={password} onChange={e => setPassword(e.target.value)} required />
              <button type="button" className={styles.eyeBtn} onClick={() => setShowPass(v => !v)}>{showPass ? '🙈' : '👁️'}</button>
            </div>
          </div>
          {mode === 'signup' && (
            <div className={styles.field}>
              <label className={styles.label}>Target Exam</label>
              <select className={styles.select} value={examTarget} onChange={e => setExamTarget(e.target.value)}>
                {EXAM_TARGETS.map(t => <option key={t}>{t}</option>)}
              </select>
            </div>
          )}
          {error && <div className={styles.error}>⚠️ {error}</div>}
          <button type="submit" className={styles.submitBtn} disabled={loading}>
            {loading ? <><span className="spin">⟳</span> {mode === 'login' ? 'Logging in…' : 'Creating account…'}</> : mode === 'login' ? '→ Login' : '✓ Create Account'}
          </button>
        </form>

        <div className={styles.switchMode}>
          {mode === 'login'
            ? <>Don't have an account? <button onClick={() => { setMode('signup'); setError(''); }}>Sign up free</button></>
            : <>Already have an account? <button onClick={() => { setMode('login'); setError(''); }}>Login here</button></>}
        </div>

        <div className={styles.features}>
          {['Upload PDFs & generate questions from specific pages', 'Railway, SSC, UPSC, Banking exam templates', 'AI assistant for doubt solving & study help', 'Hindi & English question generation'].map(f => (
            <div key={f} className={styles.feature}><span className={styles.featureTick}>✓</span>{f}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
