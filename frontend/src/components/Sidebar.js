import React from 'react';
import styles from './Sidebar.module.css';

const navItems = [
  { id: 'dashboard', icon: '⊞', label: 'Dashboard' },
  { id: 'upload', icon: '↑', label: 'Upload PDF' },
  { id: 'generate', icon: '✦', label: 'Generate Test' },
  { id: 'my-tests', icon: '☑', label: 'My Tests' },
];

export default function Sidebar({ page, navigate, user, onLogout }) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>G</div>
        <div>
          <div className={styles.logoName}>GovPrep AI</div>
          <div className={styles.logoSub}>Exam Intelligence</div>
        </div>
      </div>

      <nav className={styles.nav}>
        <div className={styles.navSection}>Menu</div>
        {navItems.map(item => (
          <button
            key={item.id}
            className={`${styles.navItem} ${page === item.id ? styles.active : ''}`}
            onClick={() => navigate(item.id)}
          >
            <span className={styles.navIcon}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className={styles.examSection}>
        <div className={styles.navSection}>Exam Types</div>
        {[
          { label: 'Railway', color: '#2563eb' },
          { label: 'SSC CGL', color: '#16a34a' },
          { label: 'UPSC', color: '#7c3aed' },
          { label: 'Banking', color: '#d97706' },
        ].map(e => (
          <div key={e.label} className={styles.examBadge}>
            <span className={styles.examDot} style={{ background: e.color }} />
            {e.label}
          </div>
        ))}
      </div>

      <div className={styles.userArea}>
        <div className={styles.avatar}>{user?.name?.slice(0, 2).toUpperCase() || 'US'}</div>
        <div className={styles.userInfo}>
          <div className={styles.userName}>{user?.name || 'User'}</div>
          <div className={styles.userTarget}>{user?.examTarget || 'Aspirant'}</div>
        </div>
        <button className={styles.logoutBtn} onClick={onLogout} title="Logout">⏻</button>
      </div>
    </aside>
  );
}
