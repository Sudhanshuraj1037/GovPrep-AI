import React, { useState, useEffect, useRef } from 'react';
import { sendChat, getChatHistory, clearChatHistory } from '../api';
import styles from './AssistantWidget.module.css';

const QUICK_QUESTIONS = [
  'How should I prepare for Railway RRB NTPC?',
  'What is negative marking in SSC CGL?',
  'Give me tips to improve my score',
  'How to manage time in mock tests?',
];

export default function AssistantWidget({ user }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [historyLoaded, setHistoryLoaded] = useState(false);
  const bottomRef = useRef();

  // Load chat history when opened first time
  useEffect(() => {
    if (open && !historyLoaded) {
      getChatHistory().then(d => {
        if (d.messages && d.messages.length > 0) {
          setMessages(d.messages.map(m => ({ role: m.role, content: m.content })));
        } else {
          setMessages([{ role: 'assistant', content: `Hi ${user?.name || 'there'}! 👋 I'm your GovPrep AI assistant.\n\nI can help you with:\n• Exam preparation tips & strategy\n• Doubt solving from your PDFs\n• Explaining wrong answers from your tests\n• Subject concepts & current affairs\n\nWhat would you like help with today?` }]);
        }
        setHistoryLoaded(true);
      }).catch(() => {
        setMessages([{ role: 'assistant', content: `Hi ${user?.name || 'there'}! 👋 I'm your GovPrep AI assistant. How can I help you today?` }]);
        setHistoryLoaded(true);
      });
    }
  }, [open, historyLoaded, user]);

  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, open]);

  const send = async (text) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: msg }]);
    setLoading(true);
    try {
      const data = await sendChat(msg);
      setMessages(prev => [...prev, { role: 'assistant', content: data.reply }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Sorry, I encountered an error: ${err.message}. Please try again.` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!window.confirm('Clear chat history?')) return;
    await clearChatHistory();
    setMessages([{ role: 'assistant', content: `Chat cleared! How can I help you, ${user?.name || 'there'}?` }]);
  };

  return (
    <>
      {/* Floating button */}
      <button className={`${styles.fab} ${open ? styles.fabOpen : ''}`} onClick={() => setOpen(v => !v)} aria-label="AI Assistant">
        {open ? '✕' : '💬'}
        {!open && <span className={styles.fabLabel}>How can I help?</span>}
      </button>

      {/* Chat window */}
      {open && (
        <div className={styles.window}>
          {/* Header */}
          <div className={styles.header}>
            <div className={styles.headerLeft}>
              <div className={styles.headerIcon}>🤖</div>
              <div>
                <div className={styles.headerTitle}>GovPrep Assistant</div>
                <div className={styles.headerSub}>Powered by Groq AI · Always ready</div>
              </div>
            </div>
            <div className={styles.headerActions}>
              <button className={styles.clearBtn} onClick={handleClear} title="Clear chat">🗑️</button>
              <button className={styles.closeBtn} onClick={() => setOpen(false)}>✕</button>
            </div>
          </div>

          {/* Messages */}
          <div className={styles.messages}>
            {messages.map((m, i) => (
              <div key={i} className={`${styles.msg} ${m.role === 'user' ? styles.userMsg : styles.aiMsg}`}>
                {m.role === 'assistant' && <div className={styles.msgIcon}>🤖</div>}
                <div className={styles.msgBubble}>
                  {m.content.split('\n').map((line, j) => (
                    <span key={j}>{line}{j < m.content.split('\n').length - 1 && <br />}</span>
                  ))}
                </div>
              </div>
            ))}
            {loading && (
              <div className={`${styles.msg} ${styles.aiMsg}`}>
                <div className={styles.msgIcon}>🤖</div>
                <div className={`${styles.msgBubble} ${styles.typing}`}>
                  <span className={styles.dot} /><span className={styles.dot} /><span className={styles.dot} />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Quick questions (shown when only greeting) */}
          {messages.length <= 1 && (
            <div className={styles.quickWrap}>
              {QUICK_QUESTIONS.map(q => (
                <button key={q} className={styles.quickBtn} onClick={() => send(q)}>{q}</button>
              ))}
            </div>
          )}

          {/* Input */}
          <div className={styles.inputWrap}>
            <input
              className={styles.input}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
              placeholder="Ask anything about your exam…"
              disabled={loading}
            />
            <button className={styles.sendBtn} onClick={() => send()} disabled={!input.trim() || loading}>
              {loading ? <span className="spin">⟳</span> : '➤'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
