import React, { useEffect, useMemo, useState } from 'react'
import {
  Activity, ArrowRight, ArrowLeft, BarChart3, BrainCircuit, Check, CheckCircle2,
  ChevronRight, ChevronLeft, CircleAlert, CloudUpload, Database, Download,
  FileSpreadsheet, Gauge, Layers3, Play, RefreshCw, Rocket, Settings2,
  SlidersHorizontal, Table2, Target, WandSparkles, TrendingDown,
  PanelLeftClose, PanelLeftOpen, Search, Lock, Zap, RotateCcw, FolderUp
} from 'lucide-react'
import {
  ResponsiveContainer, BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, Legend,
  ScatterChart, Scatter, Line
} from 'recharts'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

const stages = [
  { id: 'overview', step: '01', name: 'Overview', icon: Database, desc: 'Inspect records, feature schema & missing value rates' },
  { id: 'preprocess', step: '02', name: 'Preprocessing', icon: WandSparkles, desc: 'Target configuration, missing-value strategy & deduplication' },
  { id: 'features', step: '03', name: 'Features', icon: Layers3, desc: 'Temporal derivation & logarithmic skew normalization' },
  { id: 'train', step: '04', name: 'Training', icon: BrainCircuit, desc: 'Train baseline algorithms with stratified cross-validation' },
  { id: 'evaluate', step: '05', name: 'Evaluation', icon: BarChart3, desc: 'Held-out validation, metric rankings & confusion matrices' },
  { id: 'tune', step: '06', name: 'Tuning', icon: SlidersHorizontal, desc: 'RandomizedSearchCV hyperparameter optimization' },
  { id: 'predict', step: '07', name: 'Inference', icon: Rocket, desc: 'Single-sample prediction testbench & pipeline export' },
]

const fmt = (n, digits = 3) => n == null || Number.isNaN(Number(n)) ? '—' : Number(n).toLocaleString(undefined, { maximumFractionDigits: digits })
const pct = n => n == null ? '—' : `${(Number(n) * 100).toFixed(1)}%`

function MetricCard({ label, value, sub, icon: Icon }) {
  return (
    <div className="metric-card">
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        <div className="metric-icon-badge"><Icon size={15} /></div>
      </div>
      <div className="metric-value">{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

function Panel({ title, subtitle, action, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-head">
        <div className="panel-head-titles">
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action && <div className="panel-action">{action}</div>}
      </div>
      {children}
    </section>
  )
}

/* ── Clean Engineering Heatmap Confusion Matrix ───────────────── */
function ConfusionMatrix({ matrix, classes }) {
  if (!matrix || !classes) return null
  const n = classes.length
  const isBinary = n === 2
  const rowSums = matrix.map(row => row.reduce((a, b) => a + b, 0))
  const heat = matrix.map((row, i) => row.map(v => rowSums[i] > 0 ? v / rowSums[i] : 0))

  const cellStyle = (r, c, val, h) => {
    const isDiag = r === c
    const base = isDiag ? 'rgba(212, 175, 55,' : 'rgba(239, 68, 68,'
    const alpha = Math.max(0.08, h * 0.6)
    return {
      background: `${base} ${alpha.toFixed(2)})`,
      color: isDiag ? '#f7ebd0' : '#f87171',
      fontWeight: 600,
      border: isDiag ? '1px solid rgba(212, 175, 55, 0.35)' : '1px solid rgba(239, 68, 68, 0.25)',
    }
  }

  const labelFor = (r, c) => {
    if (!isBinary) return null
    if (r === 0 && c === 0) return 'TN'
    if (r === 0 && c === 1) return 'FP'
    if (r === 1 && c === 0) return 'FN'
    return 'TP'
  }

  return (
    <div className="cm-outer">
      <div className="cm-y-label">Actual Class</div>
      <div style={{ flex: 1, minWidth: 320 }}>
        <div className="cm-x-label">Predicted Class</div>
        <div className="cm-grid" style={{ gridTemplateColumns: `90px repeat(${n}, 1fr)` }}>
          <div className="cm-cell cm-corner" />
          {classes.map((cls, c) => (
            <div key={c} className="cm-cell cm-header">{String(cls)}</div>
          ))}
          {matrix.map((row, r) => (
            <React.Fragment key={r}>
              <div className="cm-cell cm-row-header">{String(classes[r])}</div>
              {row.map((val, c) => (
                <div key={c} className="cm-cell" style={cellStyle(r, c, val, heat[r][c])}>
                  <span className="cm-val">{val.toLocaleString()}</span>
                  {isBinary && <span className="cm-tag">{labelFor(r, c)}</span>}
                  <span className="cm-pct">{rowSums[r] > 0 ? `${(heat[r][c] * 100).toFixed(1)}%` : ''}</span>
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
        {isBinary && (
          <div className="cm-legend">
            <span className="cm-leg-item cm-leg-tn">TN · True Negative</span>
            <span className="cm-leg-item cm-leg-tp">TP · True Positive</span>
            <span className="cm-leg-item cm-leg-fp">FP · False Positive</span>
            <span className="cm-leg-item cm-leg-fn">FN · False Negative</span>
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Premium Golden Loss Charts ───────────────────────────────── */
const LOSS_COLORS = { MAE: '#d4af37', MSE: '#f59e0b', RMSE: '#f3d999' }

function RegressionLossCharts({ results, bestModel }) {
  const rows = (results || []).filter(r => !r.error && r.mae != null)
  if (!rows.length) return null

  const barData = rows.map(r => ({
    name: r.model.replace(' Regressor', '\nReg'),
    MAE: r.mae != null ? +r.mae.toFixed(4) : null,
    MSE: r.mse != null ? +r.mse.toFixed(4) : null,
    RMSE: r.rmse != null ? +r.rmse.toFixed(4) : null,
  }))

  const best = rows.find(r => r.model === bestModel) || rows[0]
  const scatterData = useMemo(() => {
    if (!best?.y_test) return []
    const step = Math.max(1, Math.floor(best.y_test.length / 300))
    return best.y_test
      .filter((_, i) => i % step === 0)
      .map((actual, i) => ({ actual, predicted: best.y_pred[i * step] }))
  }, [best?.model, best?.y_test, best?.y_pred])

  const axMin = scatterData.length ? Math.min(...scatterData.map(d => Math.min(d.actual, d.predicted))) : 0
  const axMax = scatterData.length ? Math.max(...scatterData.map(d => Math.max(d.actual, d.predicted))) : 1
  const ref = [{ x: axMin, y: axMin }, { x: axMax, y: axMax }]

  const CustomTooltipBar = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{ background: '#121216', border: '1px solid #363642', borderRadius: 6, padding: '8px 12px', fontSize: 11, boxShadow: '0 4px 14px rgba(0,0,0,0.5)' }}>
        <p style={{ margin: '0 0 4px', color: '#f3d999', fontWeight: 700 }}>{label}</p>
        {payload.map(p => (
          <p key={p.name} style={{ margin: '2px 0', color: LOSS_COLORS[p.name], display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span>{p.name}:</span> <b style={{ fontFamily: 'monospace' }}>{p.value}</b>
          </p>
        ))}
      </div>
    )
  }

  const CustomTooltipScatter = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0]?.payload
    return (
      <div style={{ background: '#121216', border: '1px solid #363642', borderRadius: 6, padding: '8px 12px', fontSize: 11, boxShadow: '0 4px 14px rgba(0,0,0,0.5)' }}>
        <p style={{ margin: '2px 0', color: '#a3a3ad' }}>Actual: <b style={{ color: '#fff', fontFamily: 'monospace' }}>{d?.actual?.toFixed(4)}</b></p>
        <p style={{ margin: '2px 0', color: '#d4af37' }}>Predicted: <b style={{ color: '#f3d999', fontFamily: 'monospace' }}>{d?.predicted?.toFixed(4)}</b></p>
      </div>
    )
  }

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <Panel title="Loss Metrics" subtitle="MAE · MSE · RMSE across candidate regressors (lower is better)">
        <div className="chart-wrap" style={{ height: 220 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ left: 8, right: 12, top: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="#1f1e24" />
              <XAxis dataKey="name" stroke="#52525e" fontSize={11} tick={{ whiteSpace: 'pre' }} />
              <YAxis stroke="#52525e" fontSize={11} />
              <Tooltip content={<CustomTooltipBar />} />
              <Legend wrapperStyle={{ fontSize: 11, color: '#d4af37' }} />
              <Bar dataKey="MAE" fill={LOSS_COLORS.MAE} radius={[2, 2, 0, 0]} />
              <Bar dataKey="MSE" fill={LOSS_COLORS.MSE} radius={[2, 2, 0, 0]} />
              <Bar dataKey="RMSE" fill={LOSS_COLORS.RMSE} radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>
      {scatterData.length > 0 && (
        <Panel title="Actual vs Predicted Values" subtitle={`${best?.model} · 300 test points · Diagonal denotes ideal alignment`}>
          <div className="chart-wrap" style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="#1f1e24" />
                <XAxis dataKey="actual" name="Actual" stroke="#52525e" fontSize={11} />
                <YAxis dataKey="predicted" name="Predicted" stroke="#52525e" fontSize={11} />
                <Tooltip content={<CustomTooltipScatter />} />
                <Scatter data={scatterData} fill="#d4af37" fillOpacity={0.65} r={3} />
                <Line data={ref} dataKey="y" type="linear" dot={false} stroke="#e5c158" strokeDasharray="4 2" strokeWidth={1.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="callout">
            <TrendingDown size={16} />
            <div>
              <strong>Residual Analysis</strong>
              <span>Tight concentration on the diagonal reflects high accuracy. Dispersion illustrates prediction variance and potential systematic bias.</span>
            </div>
          </div>
        </Panel>
      )}
    </div>
  )
}

/* ── Main Application Component ───────────────────────────────── */
export default function App() {
  const [active, setActive] = useState('overview')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [summary, setSummary] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [target, setTarget] = useState('Class')
  const [task, setTask] = useState('classification')
  const [imputation, setImputation] = useState('median')
  const [dropDuplicates, setDropDuplicates] = useState(true)
  const [prepared, setPrepared] = useState(false)
  const [prepResult, setPrepResult] = useState(null)
  const [featurePreview, setFeaturePreview] = useState(null)
  const [trainResult, setTrainResult] = useState(null)
  const [tuneResult, setTuneResult] = useState(null)
  const [selectedModel, setSelectedModel] = useState('')
  const [predictionInputs, setPredictionInputs] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [testSize, setTestSize] = useState('0.2')
  const [fileName, setFileName] = useState('')

  // Table search & pagination state
  const [tableSearch, setTableSearch] = useState('')
  const [previewPage, setPreviewPage] = useState(0)
  const pageSize = 8

  const call = async (path, options = {}) => {
    const res = await fetch(`${API}${path}`, options)
    const body = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(body.detail || 'Something went wrong')
    return body
  }

  const refreshSummary = async () => {
    const s = await call('/api/dataset/summary')
    setSummary(s.loaded ? s : null)
    if (s.loaded && s.columns?.length && !s.columns.includes(target)) {
      setTarget(s.columns[s.columns.length - 1])
    }
    return s
  }

  useEffect(() => {
    refreshSummary().catch(() => {})
  }, [])

  const runAction = async (label, fn) => {
    setBusy(label)
    setError('')
    setNotice('')
    try {
      await fn()
    } catch (e) {
      setError(e.message || 'Request failed')
    } finally {
      setBusy('')
    }
  }

  const loadDemo = () => runAction('demo', async () => {
    const result = await call('/api/dataset/demo', { method: 'POST' })
    setSummary(result.summary)
    setTarget('Class')
    setTask('classification')
    setPrepared(false)
    setPrepResult(null)
    setTrainResult(null)
    setTuneResult(null)
    setPrediction(null)
    setFeaturePreview(null)
    setFileName('synthetic_demo.csv')
    setNotice(result.warning || result.message)
    setActive('overview')
  })

  const upload = e => {
    const file = e.target.files?.[0]
    if (!file) return
    runAction('upload', async () => {
      const form = new FormData()
      form.append('file', file)
      const result = await call('/api/dataset/upload', { method: 'POST', body: form })
      setSummary(result.summary)
      setTarget(result.summary.columns[result.summary.columns.length - 1])
      setFileName(file.name)
      setPrepared(false)
      setPrepResult(null)
      setTrainResult(null)
      setTuneResult(null)
      setPrediction(null)
      setFeaturePreview(null)
      setNotice(result.message)
      setActive('overview')
    })
    e.target.value = ''
  }

  const doPreprocess = () => runAction('preprocess', async () => {
    const result = await call('/api/preprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target, task, imputation, drop_duplicates: dropDuplicates })
    })
    const preview = await call('/api/features/preview')
    setPrepResult(result.result)
    setSummary(result.summary)
    setFeaturePreview(preview)
    setPrepared(true)
    setTrainResult(null)
    setTuneResult(null)
    setPrediction(null)
    setNotice('Preprocessing completed. Transformers fitted on training partition.')
    setActive('features')
  })

  const doTrain = () => runAction('train', async () => {
    const result = await call('/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test_size: Number(testSize), random_state: 42 })
    })
    setTrainResult(result)
    setSelectedModel(result.best_model)
    setTuneResult(null)
    setPrediction(null)
    setNotice(`Training complete. Selected baseline model: ${result.best_model}`)
    setActive('evaluate')
  })

  const doTune = () => runAction('tune', async () => {
    const model = selectedModel || trainResult?.best_model
    if (!model) throw new Error('Train models first.')
    const result = await call('/api/tune', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, n_iter: 8 })
    })
    setTuneResult(result)
    setNotice(`Hyperparameter tuning complete for ${model}.`)
  })

  const doPredict = () => runAction('predict', async () => {
    const result = await call('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ values: predictionInputs })
    })
    setPrediction(result)
  })

  const downloadModel = () => window.open(`${API}/api/model/download`, '_blank', 'noopener,noreferrer')

  const autofillSampleInputs = () => {
    const firstRow = summary?.preview?.rows?.[0]
    if (!firstRow) return
    const filled = {}
    numericColumns.forEach(col => {
      if (firstRow[col] != null) {
        filled[col] = Number(firstRow[col])
      }
    })
    setPredictionInputs(filled)
    setNotice('Autofilled fields from preview record #1.')
  }

  const clearPredictionInputs = () => {
    setPredictionInputs({})
    setPrediction(null)
  }

  const columns = summary?.columns || []
  const rows = summary?.preview?.rows || []
  const missingData = useMemo(() => (summary?.columns || []).map(c => ({
    name: c,
    missing: summary?.missing?.[c] || 0
  })).filter(x => x.missing > 0).sort((a, b) => b.missing - a.missing).slice(0, 10), [summary])

  const resultRows = trainResult?.results?.filter(r => !r.error) || []
  const bestMetrics = resultRows.find(r => r.model === trainResult?.best_model) || resultRows[0]
  const featureColumns = columns.filter(c => c !== target && c !== 'Class')
  const numericColumns = useMemo(() => {
    const dtypes = summary?.dtypes || {}
    return featureColumns.filter(c => /int|float|bool/.test(dtypes[c] || ''))
  }, [featureColumns, summary])

  const filteredColumns = useMemo(() => {
    if (!tableSearch) return columns
    return columns.filter(c => c.toLowerCase().includes(tableSearch.toLowerCase()))
  }, [columns, tableSearch])

  const isStageComplete = (index) => {
    if (index === 0) return !!summary
    if (index === 1) return prepared
    if (index === 2) return prepared
    if (index === 3) return !!trainResult
    if (index === 4) return !!trainResult
    if (index === 5) return !!tuneResult
    if (index === 6) return !!prediction
    return false
  }

  const isStageUnlocked = (id) => {
    if (id === 'overview') return true
    if (id === 'preprocess') return !!summary
    if (id === 'features') return prepared
    if (id === 'train') return prepared
    if (id === 'evaluate') return !!trainResult
    if (id === 'tune') return !!trainResult
    if (id === 'predict') return !!trainResult
    return false
  }

  const go = id => {
    if (['preprocess', 'features', 'train', 'evaluate', 'tune', 'predict'].includes(id) && !summary) {
      setError('Load or upload a dataset first to begin.')
      setActive('overview')
      return
    }
    if (['features', 'train', 'evaluate', 'tune', 'predict'].includes(id) && !prepared) {
      setError('Run preprocessing before advancing.')
      setActive('preprocess')
      return
    }
    if (['evaluate', 'tune', 'predict'].includes(id) && !trainResult) {
      setError('Train candidate models first.')
      setActive('train')
      return
    }
    setError('')
    setActive(id)
  }

  const currentStageIndex = stages.findIndex(s => s.id === active)
  const prevStage = currentStageIndex > 0 ? stages[currentStageIndex - 1] : null
  const nextStage = currentStageIndex < stages.length - 1 ? stages[currentStageIndex + 1] : null

  return (
    <div className="app-shell">
      {/* ── Sidebar ─────────────────────────────────────────────── */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="brand-section">
          <div className="brand-left">
            <div className="brand-mark">
              <Activity size={18} />
            </div>
            {!sidebarCollapsed && (
              <div className="brand-text">
                <strong>MLFlow Studio</strong>
                <span>PREMIUM WORKBENCH</span>
              </div>
            )}
          </div>
          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={15} /> : <PanelLeftClose size={15} />}
          </button>
        </div>

        {!sidebarCollapsed && <div className="nav-label">Lifecycle Stages</div>}

        <nav className="nav-list">
          {stages.map((s, idx) => {
            const Icon = s.icon
            const complete = isStageComplete(idx)
            const unlocked = isStageUnlocked(s.id)
            return (
              <button
                key={s.id}
                className={`nav-item ${active === s.id ? 'active' : ''}`}
                onClick={() => go(s.id)}
                title={sidebarCollapsed ? s.name : undefined}
              >
                <span className="nav-item-icon">
                  <Icon size={16} />
                </span>
                {!sidebarCollapsed && (
                  <>
                    <span className="nav-item-label">{s.name}</span>
                    <span className="nav-item-badge">
                      {complete ? <Check size={13} /> : (!unlocked ? <Lock size={12} style={{ color: '#52525e' }} /> : null)}
                    </span>
                  </>
                )}
              </button>
            )
          })}
        </nav>

        <div className="sidebar-footer">
          <div className={`status-dot ${summary ? 'active' : ''}`} />
          {!sidebarCollapsed && (
            <>
              <span className="footer-text">
                {busy ? 'Processing' : (summary ? 'Dataset loaded' : 'Standby')}
              </span>
              <span className="version-tag">PRO v1.2</span>
            </>
          )}
        </div>
      </aside>

      {/* ── Main Viewport ───────────────────────────────────────── */}
      <div className="main-viewport">
        {/* Top Header */}
        <header className="topbar">
          <div className="topbar-left">
            <div className="breadcrumb-trail">
              <span>Workbench</span>
              <ChevronRight size={13} />
              <span className="breadcrumb-stage-badge">STAGE {stages.find(s => s.id === active)?.step}</span>
              <span className="breadcrumb-stage">{stages.find(s => s.id === active)?.name}</span>
            </div>
          </div>
          <div className="topbar-right">
            <div className={`api-pill ${summary ? 'connected' : ''}`}>
              <div className="status-bulb" />
              <span>{busy ? 'Working...' : (summary ? 'API Active' : 'API Idle')}</span>
            </div>
            <button className="btn btn-quiet" onClick={loadDemo} disabled={!!busy}>
              <RefreshCw size={13} className={busy === 'demo' ? 'spinner-icon' : ''} />
              <span>Load Demo</span>
            </button>
            <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
              <CloudUpload size={14} />
              <span>Upload CSV</span>
              <input type="file" accept=".csv,text/csv" onChange={upload} hidden disabled={!!busy} />
            </label>
          </div>
        </header>

        {/* ── Segmented Pipeline Stepper ─────────────────────────── */}
        <div className="pipeline-stepper-wrap">
          <div className="pipeline-stepper">
            {stages.map((s, idx) => {
              const activeStep = active === s.id
              const completed = isStageComplete(idx)
              const unlocked = isStageUnlocked(s.id)
              const isLast = idx === stages.length - 1

              return (
                <React.Fragment key={s.id}>
                  <div
                    className={`stepper-item ${activeStep ? 'active' : ''} ${completed ? 'completed' : ''} ${!unlocked ? 'locked' : ''}`}
                    onClick={() => unlocked && go(s.id)}
                  >
                    <div className="stepper-bubble">
                      {completed ? <Check size={11} /> : (!unlocked ? <Lock size={10} /> : s.step)}
                    </div>
                    <div className="stepper-label-group">
                      <span className="stepper-step-num">{s.step}</span>
                      <span className="stepper-title">{s.name}</span>
                    </div>
                  </div>
                  {!isLast && (
                    <div className={`stepper-connector ${completed ? 'passed' : ''}`} />
                  )}
                </React.Fragment>
              )
            })}
          </div>
        </div>

        {/* ── Active Content Area ───────────────────────────────── */}
        <main className="content-area">
          {/* Stage Hero Title */}
          <div className="stage-hero">
            <div>
              <div className="stage-hero-meta">
                <span>Stage {stages.find(s => s.id === active)?.step} of 07</span>
              </div>
              <h1 className="stage-hero-title">
                {stages.find(s => s.id === active)?.name}
              </h1>
              <p className="stage-hero-desc">
                {stages.find(s => s.id === active)?.desc}
              </p>
            </div>
            <div className="stage-hero-actions">
              {summary && (
                <button
                  className="btn btn-secondary"
                  onClick={() => runAction('refresh', refreshSummary)}
                  disabled={!!busy}
                  title="Reload dataset summary"
                >
                  <RefreshCw size={13} className={busy === 'refresh' ? 'spinner-icon' : ''} />
                  <span>Refresh stats</span>
                </button>
              )}
            </div>
          </div>

          {/* Alert Banners */}
          {error && (
            <div className="alert-banner error">
              <CircleAlert size={16} />
              <span>{error}</span>
              <button className="alert-close-btn" onClick={() => setError('')}>×</button>
            </div>
          )}

          {notice && (
            <div className="alert-banner notice">
              <CheckCircle2 size={16} />
              <span>{notice}</span>
              <button className="alert-close-btn" onClick={() => setNotice('')}>×</button>
            </div>
          )}

          {busy && (
            <div className="loading-bar">
              <RefreshCw size={14} className="spinner-icon" />
              <span>
                {({
                  demo: 'Generating synthetic benchmark dataset...',
                  upload: 'Parsing & validating uploaded CSV...',
                  preprocess: 'Fitting imputer and encoders on training partition...',
                  train: 'Training candidate algorithms with stratified cross-validation...',
                  tune: 'Running RandomizedSearchCV parameter exploration...',
                  predict: 'Computing model pipeline prediction...'
                })[busy] || 'Processing request...'}
              </span>
            </div>
          )}

          {/* ── Empty State Dropzone ───────────────────────────── */}
          {!summary ? (
            <div className="empty-state-canvas">
              <div className="empty-state-icon">
                <FolderUp size={22} />
              </div>
              <h2>Load dataset to initiate ML pipeline</h2>
              <p>
                Upload a structured CSV file (e.g. credit card fraud detection) or generate a reproducible synthetic demo partition to test candidate models.
              </p>
              <div className="empty-state-actions">
                <button className="btn btn-primary" onClick={loadDemo} disabled={!!busy}>
                  <Play size={14} />
                  <span>Load Demo Dataset (2,500 rows)</span>
                </button>
                <label className="btn btn-secondary" style={{ cursor: 'pointer' }}>
                  <CloudUpload size={14} />
                  <span>Select CSV File</span>
                  <input type="file" accept=".csv,text/csv" onChange={upload} hidden disabled={!!busy} />
                </label>
              </div>
              <span className="empty-state-hint">
                Standard format: CSV · Max 80 MB · Missing values handled via imputer
              </span>
            </div>
          ) : (
            <>
              {/* Dataset Active Header Banner */}
              <div className="dataset-banner">
                <div className="dataset-banner-icon">
                  <FileSpreadsheet size={18} />
                </div>
                <div className="dataset-banner-info">
                  <div className="dataset-banner-title">
                    <strong>{fileName || 'Active Dataset'}</strong>
                    <span className="banner-status-badge">
                      <span /> Ready
                    </span>
                  </div>
                  <div className="dataset-banner-sub">
                    <span>{fmt(summary.rows, 0)} records</span>
                    <span className="dot-sep">·</span>
                    <span>{summary.columns_count} columns</span>
                    <span className="dot-sep">·</span>
                    <span>{fmt(summary.missing_total, 0)} missing values</span>
                    <span className="dot-sep">·</span>
                    <span>Target: <b style={{ color: '#f3d999' }}>{target}</b></span>
                  </div>
                </div>
                <button
                  className="btn-icon"
                  title="Refresh metadata"
                  onClick={() => runAction('refresh', refreshSummary)}
                >
                  <RefreshCw size={14} />
                </button>
              </div>

              {/* ──────────────── STAGE 1: OVERVIEW ────────────────── */}
              {active === 'overview' && (
                <div style={{ display: 'grid', gap: 16 }}>
                  <div className="metric-grid">
                    <MetricCard
                      label="Total Records"
                      value={fmt(summary.rows, 0)}
                      sub="Rows in active dataframe"
                      icon={Database}
                    />
                    <MetricCard
                      label="Feature Columns"
                      value={Math.max(0, summary.columns_count - 1)}
                      sub={`Excluding target (${target})`}
                      icon={Layers3}
                    />
                    <MetricCard
                      label="Missing Cells"
                      value={fmt(summary.missing_total, 0)}
                      sub={summary.missing_total ? 'Configurable imputation handles these' : 'Clean: zero missing cells'}
                      icon={CircleAlert}
                    />
                    <MetricCard
                      label="Duplicate Rows"
                      value={fmt(summary.duplicate_rows, 0)}
                      sub="Exact identical rows"
                      icon={Table2}
                    />
                  </div>

                  <div className="two-col">
                    <Panel title="Dataset Profile" subtitle="Column data types and target setup">
                      <div className="profile-list">
                        <div>
                          <span>Shape (rows × columns)</span>
                          <strong style={{ color: '#f3d999' }}>{fmt(summary.rows, 0)} × {summary.columns_count}</strong>
                        </div>
                        <div>
                          <span>Numeric Features</span>
                          <strong>{Object.values(summary.dtypes || {}).filter(v => /int|float/.test(v)).length}</strong>
                        </div>
                        <div>
                          <span>Categorical Features</span>
                          <strong>{Object.values(summary.dtypes || {}).filter(v => !/int|float/.test(v)).length}</strong>
                        </div>
                        <div>
                          <span>Selected Target</span>
                          <strong style={{ color: '#d4af37' }}>{target || 'None'}</strong>
                        </div>
                      </div>
                      <div className="callout">
                        <Gauge size={16} />
                        <div>
                          <strong>Validation Strategy</strong>
                          <span>Examine class balance and verify feature representations before executing transformations.</span>
                        </div>
                      </div>
                    </Panel>

                    <Panel title="Missing Value Rates" subtitle="Columns with greatest missing cell counts">
                      {missingData.length ? (
                        <div className="chart-wrap">
                          <ResponsiveContainer width="100%" height={210}>
                            <BarChart data={missingData} layout="vertical" margin={{ left: 14, right: 14, top: 4, bottom: 4 }}>
                              <CartesianGrid strokeDasharray="2 2" horizontal={false} stroke="#1f1e24" />
                              <XAxis type="number" stroke="#52525e" fontSize={10} />
                              <YAxis dataKey="name" type="category" width={52} stroke="#52525e" fontSize={10} />
                              <Tooltip
                                contentStyle={{ background: '#121216', border: '1px solid #363642', borderRadius: 6, fontSize: 11 }}
                              />
                              <Bar dataKey="missing" fill="#d4af37" radius={[0, 2, 2, 0]} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      ) : (
                        <div className="chart-empty">
                          <CheckCircle2 size={22} style={{ color: '#d4af37' }} />
                          <strong>No Missing Values Detected</strong>
                          <span>All columns in the dataset are fully populated.</span>
                        </div>
                      )}
                    </Panel>
                  </div>

                  {/* Data Preview Table */}
                  <Panel
                    title="Data Preview"
                    subtitle="First rows and schema data types"
                    action={
                      <div className="table-toolbar" style={{ margin: 0 }}>
                        <div className="table-search-box">
                          <Search size={13} style={{ color: '#52525e' }} />
                          <input
                            type="text"
                            placeholder="Filter column names..."
                            value={tableSearch}
                            onChange={e => setTableSearch(e.target.value)}
                          />
                        </div>
                      </div>
                    }
                  >
                    <div className="table-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th style={{ width: 40 }}>#</th>
                            {filteredColumns.map(c => {
                              const dtype = summary.dtypes?.[c] || ''
                              const isNum = /int|float/.test(dtype)
                              const isTarget = c === target
                              return (
                                <th key={c}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                                    <span>{c}</span>
                                    {isTarget ? (
                                      <span className="best-tag" style={{ margin: 0 }}>TARGET</span>
                                    ) : (
                                      <span className="type-pill">
                                        {isNum ? 'NUM' : 'CAT'}
                                      </span>
                                    )}
                                  </div>
                                  <small>{dtype}</small>
                                </th>
                              )
                            })}
                          </tr>
                        </thead>
                        <tbody>
                          {rows.slice(previewPage * pageSize, (previewPage + 1) * pageSize).map((r, rIdx) => (
                            <tr key={rIdx}>
                              <td style={{ color: '#52525e', fontFamily: 'monospace' }}>
                                {previewPage * pageSize + rIdx + 1}
                              </td>
                              {filteredColumns.map(c => (
                                <td key={c}>
                                  {r[c] == null ? (
                                    <span className="null-cell">null</span>
                                  ) : typeof r[c] === 'number' ? (
                                    fmt(r[c], 4)
                                  ) : (
                                    String(r[c]).slice(0, 36)
                                  )}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="panel-footer">
                      <span>Showing {Math.min(rows.length, (previewPage + 1) * pageSize)} of {rows.length} preview records</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <button
                          className="btn btn-quiet"
                          onClick={() => setPreviewPage(Math.max(0, previewPage - 1))}
                          disabled={previewPage === 0}
                        >
                          <ChevronLeft size={13} /> Prev
                        </button>
                        <span style={{ fontSize: 11, color: '#d4af37', fontFamily: 'monospace' }}>
                          Page {previewPage + 1} of {Math.ceil(rows.length / pageSize) || 1}
                        </span>
                        <button
                          className="btn btn-quiet"
                          onClick={() => setPreviewPage(previewPage + 1)}
                          disabled={(previewPage + 1) * pageSize >= rows.length}
                        >
                          Next <ChevronRight size={13} />
                        </button>
                      </div>
                    </div>
                  </Panel>
                </div>
              )}

              {/* ──────────────── STAGE 2: PREPROCESSING ────────────── */}
              {active === 'preprocess' && (
                <div style={{ display: 'grid', gap: 16 }}>
                  <div className="two-col">
                    <Panel title="Configuration" subtitle="Target column, task type and imputation strategy">
                      <div className="form-stack">
                        <div className="form-group">
                          <label>Target Column</label>
                          <select value={target} onChange={e => setTarget(e.target.value)}>
                            {columns.map(c => <option key={c}>{c}</option>)}
                          </select>
                          <small>The dependent variable the pipeline will learn to predict.</small>
                        </div>

                        <div className="form-group">
                          <label>Problem Type</label>
                          <select value={task} onChange={e => setTask(e.target.value)}>
                            <option value="classification">Classification (e.g. Fraud detection, binary or multiclass)</option>
                            <option value="regression">Regression (Continuous values)</option>
                          </select>
                          <small>Determines candidate models, splitting strategy, and evaluation metrics.</small>
                        </div>

                        <div className="form-group">
                          <label>Numeric Missing-Value Strategy</label>
                          <select value={imputation} onChange={e => setImputation(e.target.value)}>
                            <option value="median">Median Imputation (Robust to outliers)</option>
                            <option value="mean">Mean Imputation</option>
                          </select>
                          <small>Fitted strictly on training folds to prevent data leakage.</small>
                        </div>

                        <label className="checkbox-toggle">
                          <input
                            type="checkbox"
                            checked={dropDuplicates}
                            onChange={e => setDropDuplicates(e.target.checked)}
                          />
                          <span>Remove exact duplicate rows</span>
                        </label>

                        <button
                          className="btn btn-primary full-width"
                          onClick={doPreprocess}
                          disabled={!!busy}
                          style={{ marginTop: 6 }}
                        >
                          <WandSparkles size={14} />
                          <span>Execute Preprocessing</span>
                          <ArrowRight size={14} />
                        </button>
                      </div>
                    </Panel>

                    <Panel title="Pipeline Transformers" subtitle="Transformers applied during training and inference">
                      <div className="process-list">
                        <div className="process-item">
                          <div className="process-icon"><Check size={13} /></div>
                          <div className="process-text">
                            <strong>Training Fold Isolation</strong>
                            <p>Transformers (StandardScaler, SimpleImputer) fit exclusively on the train split and transform test partitions.</p>
                          </div>
                        </div>
                        <div className="process-item">
                          <div className="process-icon"><Check size={13} /></div>
                          <div className="process-text">
                            <strong>Target Validation</strong>
                            <p>Rows missing the target label are removed prior to training.</p>
                          </div>
                        </div>
                        <div className="process-item">
                          <div className="process-icon"><Check size={13} /></div>
                          <div className="process-text">
                            <strong>One-Hot Categorical Encoding</strong>
                            <p>Encodes string/category columns while handling unseen categories safely during inference.</p>
                          </div>
                        </div>
                      </div>
                      <div className="callout">
                        <Activity size={16} />
                        <div>
                          <strong>Pipeline Serialization</strong>
                          <span>All fitted transformers are packaged with the estimator into an exportable Scikit-learn Pipeline artifact.</span>
                        </div>
                      </div>
                    </Panel>
                  </div>

                  {prepResult && (
                    <Panel title="Preprocessing Summary" subtitle="Dataset dimensions following transformation">
                      <div className="metric-grid compact">
                        <MetricCard label="Rows Before" value={fmt(prepResult.rows_before, 0)} icon={Database} />
                        <MetricCard label="Rows After" value={fmt(prepResult.rows_after, 0)} icon={Check} />
                        <MetricCard label="Feature Count" value={prepResult.feature_count} icon={Layers3} />
                        <MetricCard label="Target Column" value={prepResult.target} icon={Target} />
                      </div>
                    </Panel>
                  )}
                </div>
              )}

              {/* ──────────────── STAGE 3: FEATURES ─────────────────── */}
              {active === 'features' && (
                <div style={{ display: 'grid', gap: 16 }}>
                  <div className="three-col">
                    <div className="feature-card">
                      <div className="feature-card-icon">
                        <Layers3 size={17} />
                      </div>
                      <h3>Time-Derived Features</h3>
                      <p>Extracts cyclical hour-of-day and elapsed-day features when a numeric <code>Time</code> column is present.</p>
                      <span className="feature-card-badge">
                        {columns.includes('Time') ? '✓ Applicable to this dataset' : 'Requires Time column'}
                      </span>
                    </div>

                    <div className="feature-card">
                      <div className="feature-card-icon">
                        <Activity size={17} />
                      </div>
                      <h3>Amount Transformation</h3>
                      <p>Computes logarithmic transformation <code>log1p(Amount)</code> to normalize heavy-tailed transaction balances.</p>
                      <span className="feature-card-badge">
                        {columns.includes('Amount') ? '✓ Applicable to this dataset' : 'Requires Amount column'}
                      </span>
                    </div>

                    <div className="feature-card">
                      <div className="feature-card-icon">
                        <Settings2 size={17} />
                      </div>
                      <h3>Standardized Scaling</h3>
                      <p>Scales continuous features to zero mean and unit variance for distance-sensitive algorithms.</p>
                      <span className="feature-card-badge">Active in pipeline</span>
                    </div>
                  </div>

                  <Panel
                    title="Feature Inventory"
                    subtitle={`Combined feature set targeting: ${target}`}
                  >
                    <div style={{ fontSize: 11.5, fontWeight: 600, color: '#a3a3ad', marginBottom: 6 }}>
                      Base Features ({featureColumns.length})
                    </div>
                    <div className="feature-chips">
                      {featureColumns.map(c => (
                        <span key={c} className="feature-chip">
                          {c}
                          <small>{summary.dtypes?.[c]}</small>
                        </span>
                      ))}
                    </div>

                    {featurePreview?.generated_columns?.length > 0 && (
                      <div style={{ marginTop: 14 }}>
                        <div style={{ fontSize: 11.5, fontWeight: 600, color: '#f3d999', marginBottom: 6 }}>
                          Engineered Features ({featurePreview.generated_columns.length})
                        </div>
                        <div className="feature-chips">
                          {featurePreview.generated_columns.map(c => (
                            <span key={c} className="feature-chip generated">
                              {c}
                              <small>ENGINEERED</small>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </Panel>
                </div>
              )}

              {/* ──────────────── STAGE 4: TRAINING ─────────────────── */}
              {active === 'train' && (
                <div style={{ display: 'grid', gap: 16 }}>
                  <div className="training-config-bar">
                    <div className="form-group config-split-select">
                      <label>Test Split Ratio</label>
                      <select value={testSize} onChange={e => setTestSize(e.target.value)}>
                        <option value="0.2">20% Holdout (Recommended)</option>
                        <option value="0.25">25% Holdout</option>
                        <option value="0.3">30% Holdout</option>
                      </select>
                    </div>

                    <div className="config-note-box">
                      <div className="status-dot active" />
                      <div className="config-note-text">
                        <strong>
                          {task === 'classification' ? 'Candidate Classifiers' : 'Candidate Regressors'}
                        </strong>
                        <span>
                          {task === 'classification'
                            ? 'Logistic Regression · Decision Tree · XGBoost · Random Forest · SVM · Naive Bayes'
                            : 'Linear Regression · Multiple Linear Regression · SVR · Random Forest Regressor · Gradient Boosting Regressor'}
                        </span>
                      </div>
                    </div>

                    <button
                      className="btn btn-primary"
                      onClick={doTrain}
                      disabled={!!busy}
                    >
                      <Play size={14} />
                      <span>Train & Compare Models</span>
                    </button>
                  </div>

                  <div className="three-col">
                    <div className="feature-card">
                      <BrainCircuit size={18} style={{ color: '#d4af37', marginBottom: 8 }} />
                      <h3>Multiple Model Families</h3>
                      <p>Benchmarks linear, Bayesian, and non-linear ensemble models under identical training conditions.</p>
                    </div>
                    <div className="feature-card">
                      <Target size={18} style={{ color: '#d4af37', marginBottom: 8 }} />
                      <h3>Stratified Holdout</h3>
                      <p>Maintains target class proportions across training and test splits to preserve rare positive-class rates.</p>
                    </div>
                    <div className="feature-card">
                      <Gauge size={18} style={{ color: '#d4af37', marginBottom: 8 }} />
                      <h3>Comparable Validation</h3>
                      <p>Computes accuracy, precision, recall, F1, and PR-AUC simultaneously on the held-out test split.</p>
                    </div>
                  </div>

                  {trainResult && (
                    <Panel title="Training Complete" subtitle={`Evaluated on ${trainResult.test_rows.toLocaleString()} test records`}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                        <div>
                          <span style={{ fontSize: 12, color: '#a3a3ad' }}>Selected Best Baseline Model:</span>
                          <div style={{ fontSize: 16, fontWeight: 700, color: '#f3d999', marginTop: 2 }}>
                            {trainResult.best_model}
                          </div>
                        </div>
                        <button className="btn btn-primary" onClick={() => go('evaluate')}>
                          <span>View Leaderboard</span>
                          <ArrowRight size={14} />
                        </button>
                      </div>
                    </Panel>
                  )}
                </div>
              )}

              {/* ──────────────── STAGE 5: EVALUATION ───────────────── */}
              {active === 'evaluate' && (
                <div style={{ display: 'grid', gap: 16 }}>
                  {!trainResult ? (
                    <div className="empty-state-canvas">
                      <BrainCircuit size={26} style={{ color: '#d4af37', marginBottom: 12 }} />
                      <h2>No evaluation data available</h2>
                      <p>Train candidate models first to generate the leaderboard rankings and confusion matrices.</p>
                      <button className="btn btn-primary" onClick={() => go('train')} style={{ marginTop: 16 }}>
                        <span>Go to Model Training</span>
                        <ArrowRight size={14} />
                      </button>
                    </div>
                  ) : (
                    <>
                      {/* Confusion Matrices come first: the source values for classification metrics. */}
                      {task === 'classification' && resultRows.some(r => r.confusion_matrix) && (
                        <div style={{ display: 'grid', gap: 16 }}>
                          {resultRows.filter(r => r.confusion_matrix).map(r => (
                            <Panel
                              key={r.model}
                              title={
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <span>Confusion Matrix · {r.model}</span>
                                  {r.model === trainResult.best_model && (
                                    <span className="best-tag">BEST</span>
                                  )}
                                </div>
                              }
                              subtitle="Review true/false positives and negatives before the calculated evaluation metrics."
                            >
                              <ConfusionMatrix matrix={r.confusion_matrix} classes={r.classes} modelName={r.model} />
                            </Panel>
                          ))}
                        </div>
                      )}

                      {/* Top Metric Cards */}
                      <div className="metric-grid">
                        {task === 'classification' ? (
                          <>
                            <MetricCard label="Accuracy" value={pct(bestMetrics?.accuracy)} sub="Correct predictions" icon={Gauge} />
                            <MetricCard label="Precision" value={pct(bestMetrics?.precision)} sub="Predicted positives that are positive" icon={Target} />
                            <MetricCard label="Recall" value={pct(bestMetrics?.recall)} sub="Actual positives captured" icon={Activity} />
                            <MetricCard label="F1-Score" value={pct(bestMetrics?.f1)} sub="Precision and recall harmonic mean" icon={BarChart3} />
                          </>
                        ) : (
                          <>
                            <MetricCard label="Best R² Score" value={fmt(bestMetrics?.r2)} sub="Explained variance" icon={Gauge} />
                            <MetricCard label="MAE" value={fmt(bestMetrics?.mae)} sub="Mean Absolute Error" icon={TrendingDown} />
                            <MetricCard label="MSE" value={fmt(bestMetrics?.mse)} sub="Mean Squared Error" icon={CircleAlert} />
                            <MetricCard label="MAPE" value={pct(bestMetrics?.mape)} sub="Mean Absolute Percentage Error" icon={BarChart3} />
                          </>
                        )}
                      </div>

                      {/* Leaderboard Table */}
                      <Panel
                        title="Model Leaderboard"
                        subtitle={`Ranked by ${trainResult.selection_metric} on held-out test split`}
                        action={
                          <button className="btn btn-quiet" onClick={downloadModel}>
                            <Download size={13} />
                            <span>Export Artifact (.joblib)</span>
                          </button>
                        }
                      >
                        <div className="table-scroll">
                          <table>
                            <thead>
                              <tr>
                                <th>Rank & Model</th>
                                {task === 'classification' ? (
                                  <>
                                    <th>Accuracy</th>
                                    <th>Precision</th>
                                    <th>Recall</th>
                                    <th>F1 Score</th>
                                    <th>Specificity</th>
                                    <th>ROC-AUC</th>
                                    <th>PR-AUC</th>
                                    <th>Log Loss</th>
                                    <th>MCC</th>
                                    <th>Balanced Accuracy</th>
                                  </>
                                ) : (
                                  <>
                                    <th>MAE</th>
                                    <th>MSE</th>
                                    <th>RMSE</th>
                                    <th>R²</th>
                                    <th>Adjusted R²</th>
                                    <th>MAPE</th>
                                    <th>MSLE</th>
                                    <th>RMSLE</th>
                                    <th>Median AE</th>
                                    <th>Explained Variance</th>
                                  </>
                                )}
                              </tr>
                            </thead>
                            <tbody>
                              {[...resultRows]
                                .sort((a, b) => (b[trainResult.selection_metric] ?? -Infinity) - (a[trainResult.selection_metric] ?? -Infinity))
                                .map((r, rankIdx) => {
                                  const isBest = r.model === trainResult.best_model
                                  return (
                                    <tr key={r.model} className={isBest ? 'best-row' : ''}>
                                      <td>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                          <span style={{ fontFamily: 'monospace', color: isBest ? '#d4af37' : '#737380', fontSize: 11, fontWeight: 700 }}>
                                            #{rankIdx + 1}
                                          </span>
                                          <strong>{r.model}</strong>
                                          {isBest && (
                                            <span className="best-tag">BEST</span>
                                          )}
                                        </div>
                                      </td>
                                      {task === 'classification' ? (
                                        <>
                                          <td>{pct(r.accuracy)}</td>
                                          <td>{pct(r.precision)}</td>
                                          <td>{pct(r.recall)}</td>
                                          <td>{pct(r.f1)}</td>
                                          <td>{pct(r.specificity)}</td>
                                          <td>{fmt(r.roc_auc)}</td>
                                          <td><b style={{ color: isBest ? '#f3d999' : 'inherit' }}>{fmt(r.pr_auc)}</b></td>
                                          <td>{fmt(r.log_loss)}</td>
                                          <td>{fmt(r.mcc)}</td>
                                          <td>{pct(r.balanced_accuracy)}</td>
                                        </>
                                      ) : (
                                        <>
                                          <td>{fmt(r.mae)}</td>
                                          <td>{fmt(r.mse)}</td>
                                          <td>{fmt(r.rmse)}</td>
                                          <td><b style={{ color: isBest ? '#f3d999' : 'inherit' }}>{fmt(r.r2)}</b></td>
                                          <td>{fmt(r.adjusted_r2)}</td>
                                          <td>{pct(r.mape)}</td>
                                          <td>{fmt(r.msle)}</td>
                                          <td>{fmt(r.rmsle)}</td>
                                          <td>{fmt(r.median_absolute_error)}</td>
                                          <td>{fmt(r.explained_variance)}</td>
                                        </>
                                      )}
                                    </tr>
                                  )
                                })}
                            </tbody>
                          </table>
                        </div>
                      </Panel>

                      {/* Loss Charts (Regression) */}
                      {task === 'regression' && (
                        <RegressionLossCharts results={resultRows} bestModel={trainResult.best_model} />
                      )}
                    </>
                  )}
                </div>
              )}

              {/* ──────────────── STAGE 6: TUNING ───────────────────── */}
              {active === 'tune' && (
                <div style={{ display: 'grid', gap: 16 }}>
                  <Panel title="Hyperparameter Tuning" subtitle="Randomized parameter search with 3-fold cross-validation">
                    <div className="form-stack">
                      <div className="form-group">
                        <label>Select Target Model</label>
                        <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                          {resultRows.map(r => (
                            <option key={r.model} value={r.model}>
                              {r.model} {r.model === trainResult?.best_model ? '(Recommended Best)' : ''}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div className="callout">
                        <SlidersHorizontal size={16} />
                        <div>
                          <strong>8 Iterations · 3-Fold Stratified Cross-Validation</strong>
                          <span>RandomizedSearchCV samples parameter distributions to identify optimal configurations without exhaustive grid overhead.</span>
                        </div>
                      </div>

                      <button
                        className="btn btn-primary"
                        onClick={doTune}
                        disabled={!!busy || !trainResult}
                        style={{ alignSelf: 'flex-start', marginTop: 4 }}
                      >
                        <WandSparkles size={14} />
                        <span>Run Parameter Search</span>
                      </button>
                    </div>
                  </Panel>

                  {tuneResult && (
                    <div className="two-col">
                      <Panel title="Optimal Parameters" subtitle="Identified via cross-validation">
                        <div className="parameter-list">
                          {Object.entries(tuneResult.best_params || {}).map(([k, v]) => (
                            <div key={k}>
                              <span style={{ fontFamily: 'monospace' }}>{k.replace('model__', '')}</span>
                              <strong style={{ fontFamily: 'monospace', color: '#f3d999' }}>{String(v)}</strong>
                            </div>
                          ))}
                        </div>
                        <div style={{ marginTop: 14, padding: '10px 14px', background: '#141318', border: '1px solid var(--border-gold)', borderRadius: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: 12, color: '#f3d999' }}>CV Best Score ({tuneResult.scoring}):</span>
                          <strong style={{ fontSize: 15, color: '#fff', fontFamily: 'monospace' }}>{fmt(tuneResult.cv_best_score, 4)}</strong>
                        </div>
                      </Panel>

                      <Panel title="Active Tuned Pipeline" subtitle="In-memory artifact updated">
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <div className="process-icon"><Check size={14} /></div>
                            <div>
                              <strong style={{ fontSize: 13, color: '#f3d999' }}>{tuneResult.best_model}</strong>
                              <p style={{ fontSize: 11.5, color: '#a3a3ad', margin: '2px 0 0' }}>The inference pipeline is now configured with the tuned estimator parameters.</p>
                            </div>
                          </div>
                          <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                            <button className="btn btn-secondary" onClick={downloadModel}>
                              <Download size={13} />
                              <span>Download .joblib</span>
                            </button>
                            <button className="btn btn-primary" onClick={() => go('predict')}>
                              <span>Test Predictions</span>
                              <ArrowRight size={13} />
                            </button>
                          </div>
                        </div>
                      </Panel>
                    </div>
                  )}
                </div>
              )}

              {/* ──────────────── STAGE 7: INFERENCE ────────────────── */}
              {active === 'predict' && (
                <div style={{ display: 'grid', gap: 16 }}>
                  <div className="two-col">
                    <Panel
                      title="Feature Input Simulation"
                      subtitle="Enter values or autofill from dataset preview records"
                      action={
                        <div style={{ display: 'flex', gap: 6 }}>
                          <button
                            className="btn btn-quiet"
                            onClick={autofillSampleInputs}
                            title="Populate inputs with values from sample row #1"
                          >
                            <Zap size={13} style={{ color: '#d4af37' }} />
                            <span>Autofill Sample</span>
                          </button>
                          <button
                            className="btn btn-quiet"
                            onClick={clearPredictionInputs}
                            title="Clear all fields"
                          >
                            <RotateCcw size={13} />
                            <span>Clear</span>
                          </button>
                        </div>
                      }
                    >
                      {numericColumns.length ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                          <div className="prediction-form-grid">
                            {numericColumns.slice(0, 20).map(c => (
                              <div key={c} className="pred-input-group">
                                <label>{c}</label>
                                <input
                                  type="number"
                                  step="any"
                                  placeholder="Auto (median)"
                                  value={predictionInputs[c] ?? ''}
                                  onChange={e => setPredictionInputs(p => ({
                                    ...p,
                                    [c]: e.target.value === '' ? null : Number(e.target.value)
                                  }))}
                                />
                              </div>
                            ))}
                          </div>
                          <button
                            className="btn btn-primary full-width"
                            onClick={doPredict}
                            disabled={!!busy}
                          >
                            <Rocket size={14} />
                            <span>Execute Prediction</span>
                          </button>
                        </div>
                      ) : (
                        <div className="chart-empty">
                          <CircleAlert size={20} />
                          <strong>No Numeric Features Available</strong>
                          <span>Model expects numeric feature columns for this inference testbench.</span>
                        </div>
                      )}
                    </Panel>

                    <Panel title="Inference Result" subtitle="Live response from active pipeline">
                      {prediction ? (
                        <div className="prediction-result-card">
                          <div className={`prediction-hero-badge ${String(prediction.prediction) === '1' ? 'fraud' : 'normal'}`}>
                            <div>
                              <div className="prediction-badge-title">
                                {task === 'classification'
                                  ? (String(prediction.prediction) === '1' ? 'Potential Fraud Case' : 'Normal Transaction')
                                  : 'Predicted Value'}
                              </div>
                              <div className="prediction-badge-value">
                                {task === 'classification'
                                  ? `Class ${prediction.prediction}`
                                  : fmt(prediction.prediction, 4)}
                              </div>
                            </div>
                            <div style={{ fontSize: 13, fontFamily: 'monospace', fontWeight: 700, color: String(prediction.prediction) === '1' ? '#f87171' : '#f3d999' }}>
                              {String(prediction.prediction) === '1' ? 'HIGH RISK' : 'LOW RISK'}
                            </div>
                          </div>

                          <div className="profile-list">
                            <div>
                              <span>Active Estimator</span>
                              <strong>{prediction.model}</strong>
                            </div>
                            {prediction.positive_class_probability != null && (
                              <div>
                                <span>Positive Class Probability</span>
                                <strong style={{ color: '#d4af37', fontFamily: 'monospace' }}>
                                  {pct(prediction.positive_class_probability)}
                                </strong>
                              </div>
                            )}
                          </div>

                          <div className="callout">
                            <Activity size={16} />
                            <div>
                              <strong>Model Advisory</strong>
                              <span>{prediction.probability_note || 'Prediction generated via fitted pipeline transformers.'}</span>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="chart-empty" style={{ minHeight: 240 }}>
                          <Target size={24} style={{ color: '#52525e', marginBottom: 6 }} />
                          <strong>Awaiting Feature Input</strong>
                          <span>Click "Autofill Sample" or specify feature values to test pipeline output.</span>
                        </div>
                      )}
                    </Panel>
                  </div>

                  {/* Export Artifact Panel */}
                  <Panel title="Pipeline Export" subtitle="Download bundled preprocessing and estimator artifact">
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <div className="dataset-banner-icon">
                          <Database size={18} />
                        </div>
                        <div>
                          <strong style={{ fontSize: 13, color: '#f3d999' }}>mlflow_studio_best_model.joblib</strong>
                          <p style={{ fontSize: 11.5, color: '#737380', margin: '2px 0 0' }}>
                            Encapsulates fitted scalers, encoders, and tuned estimator parameters in a Scikit-learn Pipeline.
                          </p>
                        </div>
                      </div>
                      <button className="btn btn-secondary" onClick={downloadModel}>
                        <Download size={13} />
                        <span>Download Artifact</span>
                      </button>
                    </div>
                  </Panel>
                </div>
              )}

              {/* ── Guided Bottom Stepper Bar ──────────────────────── */}
              <div className="pipeline-nav-bar">
                <div className="pipeline-nav-left">
                  {prevStage && (
                    <button
                      className="btn btn-secondary"
                      onClick={() => go(prevStage.id)}
                    >
                      <ArrowLeft size={14} />
                      <span>Back: {prevStage.name}</span>
                    </button>
                  )}
                </div>
                <div className="pipeline-nav-right">
                  <span className="stage-step-tag">STAGE {currentStageIndex + 1} OF {stages.length}</span>
                  {nextStage && (
                    <button
                      className="btn btn-primary"
                      onClick={() => go(nextStage.id)}
                    >
                      <span>Next: {nextStage.name}</span>
                      <ArrowRight size={14} />
                    </button>
                  )}
                </div>
              </div>
            </>
          )}

          {/* Footer */}
          <footer className="footer">
            <span>MLFlow Studio <b>·</b> Executive Machine Learning Workbench</span>
            <span>React 18 <b>·</b> FastAPI <b>·</b> Scikit-learn <b>·</b> Recharts</span>
          </footer>
        </main>
      </div>
    </div>
  )
}
