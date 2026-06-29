import React, { useEffect, useState } from 'react'

const STAGES = [
  { key: 'pretrain', label: '预训练', config: 'configs/pretrain.yaml' },
  { key: 'sft', label: 'SFT', config: 'configs/sft.yaml' },
  { key: 'dpo', label: 'DPO', config: 'configs/dpo.yaml' },
  { key: 'grpo', label: 'GRPO', config: 'configs/grpo.yaml' },
]

export default function TrainingPanel() {
  const [jobs, setJobs] = useState([])
  const [selectedStage, setSelectedStage] = useState('pretrain')
  const [runId, setRunId] = useState('')
  const [overrides, setOverrides] = useState('')
  const [activeLog, setActiveLog] = useState(null)
  const [logLines, setLogLines] = useState([])

  useEffect(() => {
    loadJobs()
    const iv = setInterval(loadJobs, 3000)
    return () => clearInterval(iv)
  }, [])

  const loadJobs = () => {
    fetch('/api/training')
      .then(r => r.json())
      .then(setJobs)
  }

  const start = async () => {
    const stage = STAGES.find(s => s.key === selectedStage)
    const rid = runId || `${selectedStage}_${Date.now()}`
    const ov = {}
    if (overrides) {
      overrides.split(',').forEach(pair => {
        const [k, v] = pair.split('=')
        if (k && v) ov[k.trim()] = v.trim()
      })
    }
    await fetch('/api/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: rid, config_path: stage.config, overrides: ov }),
    })
    loadJobs()
  }

  const stop = async (rid) => {
    await fetch(`/api/training/${rid}/stop`, { method: 'POST' })
    loadJobs()
  }

  const viewLog = async (rid) => {
    setActiveLog(rid)
    const r = await fetch(`/api/training/${rid}/log?tail=200`)
    const data = await r.json()
    setLogLines(data.lines || [])
  }

  return (
    <div style={{ padding: 24, height: '100%', overflowY: 'auto' }}>
      <h2 style={{ marginBottom: 20 }}>训练控制台</h2>

      {/* 启动区 */}
      <div style={{ background: '#fff', borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
          {STAGES.map(s => (
            <button
              key={s.key}
              onClick={() => setSelectedStage(s.key)}
              style={{
                padding: '8px 16px', borderRadius: 8, border: 'none',
                background: selectedStage === s.key ? '#10a37f' : '#e0e0e0',
                color: selectedStage === s.key ? '#fff' : '#333',
                cursor: 'pointer',
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
          <input
            placeholder="run_id (可选)"
            value={runId}
            onChange={e => setRunId(e.target.value)}
            style={{ flex: 1, padding: 10, borderRadius: 8, border: '1px solid #ddd' }}
          />
          <input
            placeholder="覆盖参数,如 max_iters=100,lr=1e-4"
            value={overrides}
            onChange={e => setOverrides(e.target.value)}
            style={{ flex: 2, padding: 10, borderRadius: 8, border: '1px solid #ddd' }}
          />
        </div>
        <button
          onClick={start}
          style={{
            padding: '10px 24px', borderRadius: 8, border: 'none',
            background: '#10a37f', color: '#fff', cursor: 'pointer', fontSize: 14,
          }}
        >
          启动训练
        </button>
      </div>

      {/* 任务列表 */}
      <div style={{ background: '#fff', borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <h3 style={{ marginBottom: 12 }}>训练任务</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid #eee' }}>
              <th style={{ textAlign: 'left', padding: 8 }}>run_id</th>
              <th style={{ textAlign: 'left', padding: 8 }}>阶段</th>
              <th style={{ textAlign: 'left', padding: 8 }}>状态</th>
              <th style={{ textAlign: 'left', padding: 8 }}>进度</th>
              <th style={{ textAlign: 'left', padding: 8 }}>loss</th>
              <th style={{ textAlign: 'left', padding: 8 }}>操作</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map(j => (
              <tr key={j.run_id} style={{ borderBottom: '1px solid #f0f0f0' }}>
                <td style={{ padding: 8 }}>{j.run_id}</td>
                <td style={{ padding: 8 }}>{j.stage}</td>
                <td style={{ padding: 8 }}>
                  <StatusBadge status={j.status} />
                </td>
                <td style={{ padding: 8 }}>
                  {j.max_iter ? `${j.current_iter || 0} / ${j.max_iter}` : '-'}
                </td>
                <td style={{ padding: 8 }}>{j.loss != null ? j.loss.toFixed(4) : '-'}</td>
                <td style={{ padding: 8 }}>
                  {j.status === 'running' && (
                    <button onClick={() => stop(j.run_id)} style={{ marginRight: 8, cursor: 'pointer' }}>停止</button>
                  )}
                  <button onClick={() => viewLog(j.run_id)} style={{ cursor: 'pointer' }}>日志</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 日志面板 */}
      {activeLog && (
        <div style={{ background: '#1e1e1e', color: '#d4d4d4', borderRadius: 12, padding: 16, fontFamily: 'monospace', fontSize: 12, maxHeight: 400, overflowY: 'auto' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span>日志: {activeLog}</span>
            <button onClick={() => setActiveLog(null)} style={{ background: 'transparent', color: '#fff', border: 'none', cursor: 'pointer' }}>关闭</button>
          </div>
          {logLines.map((l, i) => (
            <div key={i} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{l}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }) {
  const colors = {
    pending: '#888',
    running: '#10a37f',
    completed: '#1976d2',
    failed: '#d32f2f',
    stopped: '#ed6c02',
  }
  return (
    <span style={{
      padding: '2px 8px', borderRadius: 10, fontSize: 12,
      background: colors[status] || '#888', color: '#fff',
    }}>
      {status}
    </span>
  )
}
