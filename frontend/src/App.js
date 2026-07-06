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
import InstallBanner from './components/InstallBanner';
import styles from './App.module.css';

// Register PWA service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {});
  });
}

export default function App() {
  const [user, setUserState] = useState(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [page, setPage] = useState('dashboard');
  const [testData, setTestData] = useState(null);
  const [scorecard, setScorecard] = useState(null);
  const [pdfs, setPdfs] = useState([]);

  // Mobile sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Theme (preserved from GP2)
  const [theme, setTheme] = useState(() => localStorage.getItem('govprep_theme') || 'light');
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('govprep_theme', theme);
  }, [theme]);
  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light');

  // PWA install prompt
  const [installPrompt, setInstallPrompt] = useState(null);
  const [showInstallBanner, setShowInstallBanner] = useState(false);
  useEffect(() => {
    const handler = (e) => {
      e.preventDefault();
      setInstallPrompt(e);
      if (!localStorage.getItem('govprep_install_dismissed')) {
        setShowInstallBanner(true);
      }
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    const { outcome } = await installPrompt.userChoice;
    if (outcome === 'accepted') { setShowInstallBanner(false); setInstallPrompt(null); }
  };

  const dismissInstall = () => {
    setShowInstallBanner(false);
    localStorage.setItem('govprep_install_dismissed', '1');
  };

  // Auth (same as GP2)
  useEffect(() => {
    const stored = getUser();
    if (stored) {
      setUserState(stored);
      getMe().then(d => { setUserState(d.user); setUser(d.user); })
             .catch(() => { clearToken(); setUserState(null); });
    }
    setAuthChecked(true);
  }, []);

  const handleLogin = (token, userData) => {
    setToken(token); setUser(userData); setUserState(userData); refreshPdfs();
  };

  const handleLogout = () => {
    clearToken(); setUserState(null); setPdfs([]); setPage('dashboard');
  };

  const refreshPdfs = useCallback(async () => {
    try {
      const { getPDFs } = await import('./api');
      const data = await getPDFs();
      setPdfs(data.pdfs || []);
    } catch (e) { console.error('Failed to load PDFs:', e); }
  }, []);

  useEffect(() => { if (user) refreshPdfs(); }, [user, refreshPdfs]);

  const navigate = (p, data) => {
    setPage(p);
    if (p === 'take-test' && data) setTestData(data);
    if (p === 'results' && data) setScorecard(data);
    setSidebarOpen(false); // close mobile sidebar on navigate
    window.scrollTo(0, 0);
  };

  if (!authChecked) return <div className={styles.loading}><span className="spin">⟳</span> Loading…</div>;

  if (!user) return <AuthPage onLogin={handleLogin} theme={theme} toggleTheme={toggleTheme} />;

  const pageTitles = {
    dashboard:'Dashboard', upload:'Upload PDF', generate:'Generate Test',
    'take-test':'Mock Test', results:'Results', 'my-tests':'My Tests',
  };

  const renderPage = () => {
    switch (page) {
      case 'upload':    return <UploadPDF navigate={navigate} onUpload={refreshPdfs} />;
      case 'generate':  return <GenerateTest navigate={navigate} pdfs={pdfs} refreshPdfs={refreshPdfs} />;
      case 'take-test': return <TakeTest navigate={navigate} testData={testData} />;
      case 'results':   return <Results navigate={navigate} scorecard={scorecard} />;
      case 'my-tests':  return <MyTests navigate={navigate} pdfs={pdfs} />;
      default:          return <Dashboard navigate={navigate} pdfs={pdfs} user={user} />;
    }
  };

  return (
    <div className={styles.app}>
      {/* Mobile sidebar overlay */}
      {sidebarOpen && <div className={styles.overlay} onClick={() => setSidebarOpen(false)} />}

      <Sidebar
        page={page}
        navigate={navigate}
        user={user}
        onLogout={handleLogout}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className={styles.main}>
        {/* PWA install banner */}
        {showInstallBanner && (
          <InstallBanner onInstall={handleInstall} onDismiss={dismissInstall} />
        )}

        <header className={styles.topbar}>
          {/* Hamburger — mobile only */}
          <button className={styles.hamburger} onClick={() => setSidebarOpen(v => !v)} aria-label="Open menu">
            ☰
          </button>
          <h1 className={styles.pageTitle}>{pageTitles[page] || 'Dashboard'}</h1>
          <div className={styles.topbarRight}>
            {/* Theme toggle */}
            <button className={styles.themeBtn} onClick={toggleTheme} title="Toggle theme">
              {theme === 'light' ? '🌙' : '☀️'}
              <span className={styles.themeBtnLabel}>{theme === 'light' ? 'Dark' : 'Light'}</span>
            </button>
            <span className={styles.userBadge}>👤 {user.name.split(' ')[0]}</span>
            <button className={styles.btnPrimary} onClick={() => navigate('generate')}>+ Test</button>
          </div>
        </header>

        <main className={styles.content}>{renderPage()}</main>
      </div>

      <AssistantWidget user={user} />
    </div>
  );
}
