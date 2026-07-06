import React from 'react';
import styles from './InstallBanner.module.css';

export default function InstallBanner({ onInstall, onDismiss }) {
  return (
    <div className={styles.banner}>
      <div className={styles.left}>
        <div className={styles.icon}>📱</div>
        <div>
          <div className={styles.title}>Install GovPrep AI as an App</div>
          <div className={styles.sub}>Works offline · Faster · No browser bar</div>
        </div>
      </div>
      <div className={styles.actions}>
        <button className={styles.installBtn} onClick={onInstall}>Install</button>
        <button className={styles.dismissBtn} onClick={onDismiss} aria-label="Dismiss">✕</button>
      </div>
    </div>
  );
}
