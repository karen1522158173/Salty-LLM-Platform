import React, { useEffect, useState } from 'react'

export default function Sidebar({ view, setView, activeSessionId, setActiveSessionId }) {
  const [sessions, setSessions] = useState([])

  useEffect(() => {
    fetch('/api/chat/sessions')
      .then(r => r.json())
      .then(setSessions)
  }, [activeSessionId])

  const createSession = async () => {
    const r = await fetch('/api/chat/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: '新对话' }),
    })
    const s = await r.json()
    setActiveSessionId(s.id)
    setView('chat')
  }

  return (
    <aside style={{
      width: 260, background: '#1a1a2e', color: '#e0e0e0',
      display: 'flex', flexDirection: 'column', padding: '12px 0',
    }}>
      <div style={{ padding: '0 16px', marginBottom: 16 }}>
        <h2 style={{ fontSize: 18, marginBottom: 16 }}>Salty LLM</h2>
        <button onClick={createSession} style={{
          width: '100%', padding: '10px', borderRadius: 8, border: 'none',
          background: '#10a37f', color: '#fff', cursor: 'pointer', fontSize: 14,
        }}>
          + 新建对话
        </button>
      </div>

      <nav style={{ padding: '0 12px', marginBottom: 16 }}>
        <NavBtn label="聊天" active={view === 'chat'} onClick={() => setView('chat')} />
        <NavBtn label="训练" active={view === 'training'} onClick={() => setView('training')} />
        <NavBtn label="模型" active={view === 'models'} onClick={() => setView('models')} />
      </nav>

      <div style={{ flex: 1, overflowY: 'auto', padding: '0 12px' }}>
        <div style={{ fontSize: 12, color: '#888', marginBottom: 8, textTransform: 'uppercase' }}>历史会话</div>
        {sessions.map(s => (
          <div
            key={s.id}
            onClick={() => { setActiveSessionId(s.id); setView('chat') }}
            style={{
              padding: '8px 10px', borderRadius: 6, cursor: 'pointer',
              marginBottom: 4, fontSize: 13,
              background: activeSessionId === s.id ? '#2d2d44' : 'transparent',
            }}
          >
            {s.title || `会话 #${s.id}`}
          </div>
        ))}
      </div>
    </aside>
  )
}

function NavBtn({ label, active, onClick }) {
  return (
    <div onClick={onClick} style={{
      padding: '10px 12px', borderRadius: 6, cursor: 'pointer',
      marginBottom: 4, fontSize: 14,
      background: active ? '#2d2d44' : 'transparent',
      color: active ? '#fff' : '#aaa',
    }}>
      {label}
    </div>
  )
}
