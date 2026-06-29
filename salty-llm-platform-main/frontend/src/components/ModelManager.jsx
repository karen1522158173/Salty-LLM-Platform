import React, { useEffect, useState } from 'react'

export default function ModelManager() {
  const [checkpoints, setCheckpoints] = useState([])
  const [status, setStatus] = useState({ loaded: false })

  useEffect(() => {
    loadCheckpoints()
    fetch('/api/inference/status').then(r => r.json()).then(setStatus)
  }, [])

  const loadCheckpoints = () => {
    fetch('/api/checkpoints')
      .then(r => r.json())
      .then(setCheckpoints)
  }

  const scan = async () => {
    await fetch('/api/checkpoints/scan', { method: 'POST' })
    loadCheckpoints()
  }

  const loadModel = async (id) => {
    await fetch('/api/inference/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ckpt_id: id }),
    })
    loadCheckpoints()
    fetch('/api/inference/status').then(r => r.json()).then(setStatus)
  }

  const unloadModel = async () => {
    await fetch('/api/inference/unload', { method: 'POST' })
    loadCheckpoints()
    fetch('/api/inference/status').then(r => r.json()).then(setStatus)
  }

  const star = async (id) => {
    await fetch(`/api/checkpoints/${id}/star`, { method: 'POST' })
    loadCheckpoints()
  }

  const deleteCkpt = async (id) => {
    if (!confirm('确定删除?')) return
    await fetch(`/api/checkpoints/${id}`, { method: 'DELETE' })
    loadCheckpoints()
  }

  const stages = ['pretrain', 'sft', 'dpo', 'grpo']

  return (
    <div style={{ padding: 24, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <h2>模型管理</h2>
        <div>
          <button onClick={scan} style={{ marginRight: 12, padding: '8px 16px', borderRadius: 8, border: 'none', background: '#1976d2', color: '#fff', cursor: 'pointer' }}>
            扫描本地
          </button>
          {status.loaded && (
            <button onClick={unloadModel} style={{ padding: '8px 16px', borderRadius: 8, border: 'none', background: '#d32f2f', color: '#fff', cursor: 'pointer' }}>
              卸载模型
            </button>
          )}
        </div>
      </div>

      {stages.map(stage => {
        const cks = checkpoints.filter(c => c.stage === stage)
        if (cks.length === 0) return null
        return (
          <div key={stage} style={{ background: '#fff', borderRadius: 12, padding: 20, marginBottom: 16 }}>
            <h3 style={{ textTransform: 'uppercase', fontSize: 14, color: '#666', marginBottom: 12 }}>{stage}</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
              {cks.map(c => (
                <div key={c.id} style={{
                  border: c.is_current ? '2px solid #10a37f' : '1px solid #e0e0e0',
                  borderRadius: 10, padding: 16,
                  background: c.is_current ? '#f0faf6' : '#fff',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 600, fontSize: 14 }}>{c.name}</span>
                    <span style={{ fontSize: 12, color: '#888' }}>step {c.step ?? '?'}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#666', marginBottom: 12, wordBreak: 'break-all' }}>{c.path}</div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button
                      onClick={() => loadModel(c.id)}
                      disabled={c.is_current}
                      style={{
                        flex: 1, padding: '6px 0', borderRadius: 6, border: 'none',
                        background: c.is_current ? '#ccc' : '#10a37f', color: '#fff',
                        cursor: c.is_current ? 'not-allowed' : 'pointer', fontSize: 12,
                      }}
                    >
                      {c.is_current ? '当前加载' : '加载'}
                    </button>
                    <button
                      onClick={() => star(c.id)}
                      style={{
                        padding: '6px 10px', borderRadius: 6, border: '1px solid #ddd',
                        background: c.is_starred ? '#ffd700' : '#fff',
                        cursor: 'pointer', fontSize: 12,
                      }}
                    >
                      {c.is_starred ? '★' : '☆'}
                    </button>
                    <button
                      onClick={() => deleteCkpt(c.id)}
                      style={{
                        padding: '6px 10px', borderRadius: 6, border: '1px solid #ddd',
                        background: '#fff', color: '#d32f2f', cursor: 'pointer', fontSize: 12,
                      }}
                    >
                      删
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
