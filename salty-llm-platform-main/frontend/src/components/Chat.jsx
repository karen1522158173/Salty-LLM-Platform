import React, { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function Chat({ sessionId, onSessionChange }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState({ loaded: false })
  const bottomRef = useRef(null)

  useEffect(() => {
    fetch('/api/inference/status').then(r => r.json()).then(setStatus)
  }, [])

  useEffect(() => {
    if (sessionId) {
      fetch(`/api/chat/sessions/${sessionId}/messages`)
        .then(r => r.json())
        .then(setMessages)
    } else {
      setMessages([])
    }
  }, [sessionId])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async () => {
    if (!input.trim() || loading) return
    if (!status.loaded) {
      alert('模型未加载，请先在"模型"页面选择一个 checkpoint')
      return
    }

    const userMsg = { role: 'user', content: input.trim() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    // 如果没有 session，先创建
    let sid = sessionId
    if (!sid) {
      const r = await fetch('/api/chat/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: userMsg.content.slice(0, 20) }),
      })
      const s = await r.json()
      sid = s.id
      onSessionChange(sid)
    }

    // 保存用户消息
    await fetch(`/api/chat/sessions/${sid}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userMsg),
    })

    // 流式请求
    const allMessages = [...messages, userMsg]
    const res = await fetch('/api/inference/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: allMessages.map(m => ({ role: m.role, content: m.content })),
        stream: true,
        temperature: 0.3,
      }),
    })

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let assistantText = ''
    let thinkText = ''
    let inThink = false

    setMessages(prev => [...prev, { role: 'assistant', content: '', think_content: '' }])

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      const chunk = decoder.decode(value)
      const lines = chunk.split('\n')
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6)
        if (data === '[DONE]') continue
        try {
          const parsed = JSON.parse(data)
          const delta = parsed.choices?.[0]?.delta?.content || ''
          if (delta === '<think>') {
            inThink = true
          } else if (delta === '</think>') {
            inThink = false
          } else if (inThink) {
            thinkText += delta
            setMessages(prev => {
              const copy = [...prev]
              copy[copy.length - 1] = { ...copy[copy.length - 1], think_content: thinkText, content: assistantText }
              return copy
            })
          } else {
            assistantText += delta
            setMessages(prev => {
              const copy = [...prev]
              copy[copy.length - 1] = { ...copy[copy.length - 1], content: assistantText, think_content: thinkText }
              return copy
            })
          }
        } catch (e) {}
      }
    }

    // 保存助手消息
    await fetch(`/api/chat/sessions/${sid}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        role: 'assistant',
        content: assistantText,
        think_content: thinkText || null,
      }),
    })

    setLoading(false)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* 顶部状态栏 */}
      <div style={{
        padding: '10px 20px', borderBottom: '1px solid #e0e0e0',
        background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{ fontSize: 14, color: '#333' }}>
          {sessionId ? `会话 #${sessionId}` : '新对话'}
        </span>
        <span style={{
          fontSize: 12, padding: '4px 10px', borderRadius: 12,
          background: status.loaded ? '#d4edda' : '#f8d7da',
          color: status.loaded ? '#155724' : '#721c24',
        }}>
          {status.loaded ? `模型已加载: ${status.current_ckpt?.split('/').pop()}` : '模型未加载'}
        </span>
      </div>

      {/* 消息区域 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '20px 20%' }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: '#888', marginTop: '30%' }}>
            <h2 style={{ marginBottom: 12 }}>Salty LLM</h2>
            <p>开始你的第一个对话</p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{
            display: 'flex',
            justifyContent: m.role === 'user' ? 'flex-end' : 'flex-start',
            marginBottom: 16,
          }}>
            <div style={{
              maxWidth: '80%',
              padding: '12px 16px',
              borderRadius: 16,
              background: m.role === 'user' ? '#10a37f' : '#f0f0f0',
              color: m.role === 'user' ? '#fff' : '#333',
              fontSize: 14,
              lineHeight: 1.6,
            }}>
              {m.think_content && (
                <details style={{ marginBottom: 8, fontSize: 12, opacity: 0.7 }}>
                  <summary>思考过程</summary>
                  <div style={{ marginTop: 6, padding: 8, background: 'rgba(0,0,0,0.05)', borderRadius: 8 }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.think_content}</ReactMarkdown>
                  </div>
                </details>
              )}
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
            </div>
          </div>
        ))}
        {loading && messages[messages.length - 1]?.role === 'user' && (
          <div style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 16 }}>
            <div style={{ padding: '12px 16px', borderRadius: 16, background: '#f0f0f0' }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, background: '#999', borderRadius: '50%', animation: 'pulse 1s infinite' }} />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* 输入区域 */}
      <div style={{
        padding: '16px 20%', borderTop: '1px solid #e0e0e0', background: '#fff',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center',
          background: '#f5f5f5', borderRadius: 12, padding: '4px 4px 4px 16px',
        }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="输入消息..."
            rows={1}
            style={{
              flex: 1, border: 'none', outline: 'none', background: 'transparent',
              resize: 'none', fontSize: 14, padding: '10px 0', maxHeight: 120,
            }}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            style={{
              padding: '8px 16px', borderRadius: 8, border: 'none',
              background: loading || !input.trim() ? '#ccc' : '#10a37f',
              color: '#fff', cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: 14,
            }}
          >
            发送
          </button>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  )
}
