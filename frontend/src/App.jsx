import { useEffect, useState } from 'react'
import { Menu, Sun, Moon, Wifi, WifiOff } from 'lucide-react'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import MessageInput from './components/MessageInput'
import { useChat } from './hooks/useChat'
import { fetchHealth } from './api/chat'

export default function App() {
  const [dark, setDark] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [apiStatus, setApiStatus] = useState('checking') // 'checking' | 'ok' | 'error'

  const { messages, isStreaming, error, sendMessage, stopGeneration, clearChat } = useChat()

  // Apply dark mode class
  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  // Poll API health on mount
  useEffect(() => {
    fetchHealth()
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'))
  }, [])

  return (
    <div className={`flex h-screen overflow-hidden ${dark ? 'dark bg-chat-bg text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onNewChat={clearChat}
        messageCount={messages.length}
      />

      {/* Main */}
      <div className="flex flex-col flex-1 min-w-0 h-full">
        {/* Top bar */}
        <header className={`flex items-center justify-between px-4 py-3 border-b ${dark ? 'border-gray-700/40 bg-chat-bg' : 'border-gray-200 bg-white'}`}>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen((v) => !v)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/50 transition-colors"
              title="Toggle sidebar"
            >
              <Menu size={18} />
            </button>
            <span className="font-semibold text-sm">NanoGPT</span>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              apiStatus === 'ok' ? 'bg-emerald-900/50 text-emerald-400' :
              apiStatus === 'error' ? 'bg-red-900/50 text-red-400' :
              'bg-gray-700 text-gray-400'
            }`}>
              {apiStatus === 'ok' ? 'GPT-2 ready' : apiStatus === 'error' ? 'API offline' : 'Connecting…'}
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* API indicator */}
            {apiStatus === 'ok'
              ? <Wifi size={14} className="text-emerald-400" />
              : <WifiOff size={14} className="text-red-400" />
            }
            {/* Theme toggle */}
            <button
              onClick={() => setDark((v) => !v)}
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/50 transition-colors"
              title="Toggle theme"
            >
              {dark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="bg-red-900/30 border-b border-red-700/40 px-4 py-2 text-xs text-red-400">
            ⚠️ {error}
          </div>
        )}

        {/* Chat area */}
        <ChatWindow messages={messages} isStreaming={isStreaming} />

        {/* Input */}
        <MessageInput
          onSend={sendMessage}
          onStop={stopGeneration}
          isStreaming={isStreaming}
        />
      </div>
    </div>
  )
}
