import React, { useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import Chat from './components/Chat.jsx'
import TrainingPanel from './components/TrainingPanel.jsx'
import ModelManager from './components/ModelManager.jsx'

export default function App() {
  const [view, setView] = useState('chat')
  const [activeSessionId, setActiveSessionId] = useState(null)

  return (
    <div style={{ display: 'flex', height: '100vh' }}>
      <Sidebar
        view={view}
        setView={setView}
        activeSessionId={activeSessionId}
        setActiveSessionId={setActiveSessionId}
      />
      <main style={{ flex: 1, overflow: 'hidden' }}>
        {view === 'chat' && <Chat sessionId={activeSessionId} onSessionChange={setActiveSessionId} />}
        {view === 'training' && <TrainingPanel />}
        {view === 'models' && <ModelManager />}
      </main>
    </div>
  )
}
