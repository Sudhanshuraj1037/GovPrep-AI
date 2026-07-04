import React, { useState } from 'react';
import { generateQuestions } from '../api';
import styles from './GenerateTest.module.css';

const TEMPLATES = [
  { name: 'Railway RRB NTPC', neg: -0.33, marksPerQ: 1 },
  { name: 'SSC CGL Tier 1', neg: -0.5, marksPerQ: 2 },
  { name: 'UPSC Prelims', neg: -0.66, marksPerQ: 2 },
  { name: 'Banking IBPS PO', neg: -0.25, marksPerQ: 1 },
  { name: 'Custom', neg: -0.25, marksPerQ: 1 },
];

export default function GenerateTest({ navigate, pdfs, refreshPdfs }) {
  const [selectedPdf, setSelectedPdf] = useState('');
  const [pageStart, setPageStart] = useState(1);
  const [pageEnd, setPageEnd] = useState('');
  const [questionCount, setQuestionCount] = useState(10);
  const [difficulty, setDifficulty] = useState('mixed');
  const [template, setTemplate] = useState('Railway RRB NTPC');
  const [language, setLanguage] = useState('english');
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [step, setStep] = useState('');

  const selected = pdfs.find(p => p.id === selectedPdf);

  const handleGenerate = async () => {
    if (!selectedPdf) { setError('Please select a PDF.'); return; }
    const ps = parseInt(pageStart) || 1;
    const pe = parseInt(pageEnd) || selected?.totalPages || 999;
    if (ps < 1) { setError('Page start must be at least 1.'); return; }
    if (pe < ps) { setError('Page end must be greater than page start.'); return; }
    if (selected && pe > selected.totalPages) { setError(`This PDF only has ${selected.totalPages} pages. Set page end ≤ ${selected.totalPages}.`); return; }

    setGenerating(true); setError('');
    setStep(`Reading PDF content from pages ${ps}–${pe}…`);
    try {
      setTimeout(() => setStep('Claude AI analyzing content and creating questions…'), 1500);
      setTimeout(() => setStep('Verifying questions against source material…'), 4000);
      const data = await generateQuestions({ pdfId: selectedPdf, pageStart: ps, pageEnd: pe, questionCount: parseInt(questionCount) || 10, difficulty, examTemplate: template, questionTypes: ['MCQ'], language });
      setGenerating(false);
      navigate('take-test', { testId: data.testId, questions: data.questions, examTemplate: template, meta: data.meta, templateConfig: TEMPLATES.find(t => t.name === template) || TEMPLATES[0] });
    } catch (err) {
      setGenerating(false); setStep('');
      setError(err.message);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h2 className={styles.title}>Generate Test from PDF</h2>
        <p className={styles.sub}>Select your PDF, choose a page range, and AI will generate questions only from that content.</p>
      </div>

      {pdfs.length === 0 ? (
        <div className={styles.noPdfs}>
          <div className={styles.noPdfsIcon}>📂</div>
          <div className={styles.noPdfsTitle}>No PDFs uploaded yet</div>
          <div className={styles.noPdfsSub}>Upload a PDF first before generating questions.</div>
          <button className={styles.primaryBtn} onClick={() => navigate('upload')}>Upload a PDF</button>
        </div>
      ) : (
        <div className={styles.formCard}>

          {/* Step 1: Select PDF */}
          <div className={styles.step}>
            <div className={styles.stepHeader}><div className={styles.stepNum}>1</div><div className={styles.stepTitle}>Select PDF</div></div>
            <div className={styles.pdfList}>
              {pdfs.map(p => (
                <div key={p.id} className={`${styles.pdfOption} ${selectedPdf === p.id ? styles.selected : ''}`}
                  onClick={() => { setSelectedPdf(p.id); setPageEnd(p.totalPages); setError(''); }}>
                  <div className={styles.pdfOptionIcon}>📄</div>
                  <div className={styles.pdfOptionInfo}>
                    <div className={styles.pdfOptionName}>{p.name.length > 50 ? p.name.slice(0, 50) + '…' : p.name}</div>
                    <div className={styles.pdfOptionMeta}>{p.subject} · {p.chapter} · <strong>{p.totalPages} pages</strong></div>
                  </div>
                  {selectedPdf === p.id && <div className={styles.checkmark}>✓</div>}
                </div>
              ))}
            </div>
            <button className={styles.uploadMoreBtn} onClick={() => navigate('upload')}>+ Upload another PDF</button>
          </div>

          {/* Step 2: Page Range */}
          {selected && (
            <div className={styles.step}>
              <div className={styles.stepHeader}><div className={styles.stepNum}>2</div><div className={styles.stepTitle}>Choose Page Range</div></div>
              <div className={styles.pageRangeBox}>
                <div className={styles.pageRangeNote}>
                  📖 This PDF has <strong>{selected.totalPages} pages</strong>. Questions will be generated <strong>only from the pages you select below</strong>.
                </div>
                <div className={styles.pageRangeInputs}>
                  <div className={styles.pageField}>
                    <label className={styles.label}>From page</label>
                    <input type="number" className={styles.pageInput} value={pageStart} min={1} max={selected.totalPages} onChange={e => setPageStart(e.target.value)} />
                  </div>
                  <div className={styles.pageSep}>to</div>
                  <div className={styles.pageField}>
                    <label className={styles.label}>To page</label>
                    <input type="number" className={styles.pageInput} value={pageEnd} min={1} max={selected.totalPages} onChange={e => setPageEnd(e.target.value)} />
                  </div>
                  <div className={styles.pageQuickBtns}>
                    <button className={styles.quickBtn} onClick={() => { setPageStart(1); setPageEnd(Math.min(10, selected.totalPages)); }}>First 10</button>
                    <button className={styles.quickBtn} onClick={() => { setPageStart(1); setPageEnd(Math.min(20, selected.totalPages)); }}>First 20</button>
                    <button className={styles.quickBtn} onClick={() => { setPageStart(1); setPageEnd(selected.totalPages); }}>All pages</button>
                  </div>
                </div>
                <div className={styles.pageRangePreview}>
                  Will generate from pages <strong>{pageStart || 1}</strong> to <strong>{pageEnd || selected.totalPages}</strong>
                  {' '}({Math.max(0, (parseInt(pageEnd) || selected.totalPages) - (parseInt(pageStart) || 1) + 1)} pages selected)
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Settings */}
          <div className={styles.step}>
            <div className={styles.stepHeader}><div className={styles.stepNum}>{selected ? '3' : '2'}</div><div className={styles.stepTitle}>Test Settings / परीक्षा सेटिंग</div></div>
            <div className={styles.settingsGrid}>
              <div className={styles.formField}>
                <label className={styles.label}>Exam template</label>
                <select className={styles.select} value={template} onChange={e => setTemplate(e.target.value)}>
                  {TEMPLATES.map(t => <option key={t.name}>{t.name}</option>)}
                </select>
                {(() => { const t = TEMPLATES.find(tt => tt.name === template); return t ? <div className={styles.templateInfo}>+{t.marksPerQ} correct · {t.neg} wrong · Negative marking applies</div> : null; })()}
              </div>
              <div className={styles.formField}>
                <label className={styles.label}>Number of questions</label>
                <select className={styles.select} value={questionCount} onChange={e => setQuestionCount(e.target.value)}>
                  {[5, 10, 15, 20].map(n => <option key={n} value={n}>{n} questions</option>)}
                </select>
                <div className={styles.fieldNote}>Max 20 per generation to ensure quality</div>
              </div>
              <div className={styles.formField}>
                <label className={styles.label}>Difficulty</label>
                <select className={styles.select} value={difficulty} onChange={e => setDifficulty(e.target.value)}>
                  <option value="mixed">Mixed (Easy 30% + Medium 50% + Hard 20%)</option>
                  <option value="easy">Easy only</option>
                  <option value="medium">Medium only</option>
                  <option value="hard">Hard only</option>
                </select>
              </div>
            </div>
          </div>

          {/* Step 4: Language Selection */}
          <div className={styles.step}>
            <div className={styles.stepHeader}>
              <div className={styles.stepNum}>{selected ? '4' : '3'}</div>
              <div className={styles.stepTitle}>Exam Language / परीक्षा की भाषा</div>
            </div>
            <div className={styles.langGrid}>
              <div
                className={`${styles.langCard} ${language === 'english' ? styles.langSelected : ''}`}
                onClick={() => setLanguage('english')}
              >
                <div className={styles.langFlag}>🇬🇧</div>
                <div className={styles.langName}>English</div>
                <div className={styles.langDesc}>Questions & answers in English</div>
                {language === 'english' && <div className={styles.langCheck}>✓</div>}
              </div>
              <div
                className={`${styles.langCard} ${language === 'hindi' ? styles.langSelected : ''}`}
                onClick={() => setLanguage('hindi')}
              >
                <div className={styles.langFlag}>🇮🇳</div>
                <div className={styles.langName}>हिंदी (Hindi)</div>
                <div className={styles.langDesc}>प्रश्न और उत्तर हिंदी में होंगे</div>
                {language === 'hindi' && <div className={styles.langCheck}>✓</div>}
              </div>
            </div>
            {language === 'hindi' && (
              <div className={styles.langNote}>
                ✅ AI सभी प्रश्न, विकल्प और स्पष्टीकरण हिंदी में लिखेगा। PDF का content English में हो तो भी questions Hindi में आएंगे।
              </div>
            )}
          </div>

          {error && <div className={styles.error}>⚠️ {error}</div>}

          {generating ? (
            <div className={styles.generatingBox}>
              <div className={styles.genSpinner + ' spin'} style={{ fontSize: 32, display: 'block', marginBottom: 12 }}>⟳</div>
              <div className={styles.genText}>{step || 'Generating questions…'}</div>
              <div className={styles.genSub}>This takes 10–30 seconds. Please wait…</div>
            </div>
          ) : (
            <button className={styles.generateBtn} onClick={handleGenerate} disabled={!selectedPdf}>
              ✦ Generate Questions & Start Test
            </button>
          )}
        </div>
      )}
    </div>
  );
}
