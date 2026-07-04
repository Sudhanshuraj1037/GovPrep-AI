import React, { useState, useEffect, useCallback } from 'react';
import { getUser, getMe, clearToken, setToken, setUser } from './api';
import AuthPage from './pages/AuthPage';
import Dashboard from './pages/Dashboard';
import UploadPDF from './pages/UploadPDF';
import GenerateTest from './pages/GenerateTest';
import TakeTest from './pages/TakeTest';
import Results from './pages/Results';
import MyTests from './pages/MyTests';
import Sidebar from './components/Sidebar';
import AssistantWidget from './components/AssistantWidget';
import styles from './App.module.css';

export default function App() {
  const [user, setUserState] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [page, setPage] = useState('dashboard');
  const [testData, setTestData] = useState(null);
  const [scorecard, setScorecard] = useState(null);
  const [pdfs, setPdfs] = useState([]);

  // On mount: restore session from localStorage
  useEffect(() => {
    const stored = getUser();
    if (stored) {
      setUserState(stored);
      // Verify token is still valid
      getMe().then(d => { setUserState(d.user); setUser(d.user); }).catch(() => { clearToken(); setUserState(null); });
    }
    setAuthChecked(true);
  }, []);

  const handleLogin = (token, userData) => {
    setToken(token);
    setUser(userData);
    setUserState(userData);
    refreshPdfs();
  };

  const handleLogout = () => {
    clearToken();
    setUserState(null);
    setPdfs([]);
    setPage('dashboard');
  };

  const refreshPdfs = useCallback(async () => {
    try {
      const { getPDFs } = await import('./api');
      const data = await getPDFs();
      setPdfs(data.pdfs || []);
    } catch (e) {
      console.error('Failed to load PDFs:', e);
    }
  }, []);

  useEffect(() => { if (user) refreshPdfs(); }, [user, refreshPdfs]);

  const navigate = (p, data) => {
    setPage(p);
    if (p === 'take-test' && data) setTestData(data);
    if (p === 'results' && data) setScorecard(data);
    window.scrollTo(0, 0);
  };

  if (!authChecked) return <div className={styles.loading}><span className="spin">⟳</span> Loading…</div>;

  if (!user) return <AuthPage onLogin={handleLogin} />;

  const pageTitles = { dashboard: 'Dashboard', upload: 'Upload PDF', generate: 'Generate Test', 'take-test': 'Mock Test', results: 'Results', 'my-tests': 'My Tests' };

  const renderPage = () => {
    switch (page) {
      case 'upload': return <UploadPDF navigate={navigate} onUpload={refreshPdfs} />;
      case 'generate': return <GenerateTest navigate={navigate} pdfs={pdfs} refreshPdfs={refreshPdfs} />;
      case 'take-test': return <TakeTest navigate={navigate} testData={testData} />;
      case 'results': return <Results navigate={navigate} scorecard={scorecard} />;
      case 'my-tests': return <MyTests navigate={navigate} pdfs={pdfs} />;
      default: return <Dashboard navigate={navigate} pdfs={pdfs} user={user} />;
    }
  };

  return (
    <div className={styles.app}>
      <Sidebar page={page} navigate={navigate} user={user} onLogout={handleLogout} />
      <div className={styles.main}>
        <header className={styles.topbar}>
          <h1 className={styles.pageTitle}>{pageTitles[page] || 'Dashboard'}</h1>
          <div className={styles.topbarRight}>
            <span className={styles.userBadge}>👤 {user.name}</span>
            <button className={styles.btnPrimary} onClick={() => navigate('generate')}>+ New Test</button>
          </div>
        </header>
        <main className={styles.content}>{renderPage()}</main>
      </div>
      {/* Floating AI Assistant */}
      <AssistantWidget user={user} />
    </div>
  );
}
