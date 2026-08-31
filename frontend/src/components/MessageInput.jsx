import { useRef, useState, useEffect } from 'react'
import { ArrowUp, Square, ChevronDown, ChevronUp, Sliders } from 'lucide-react'

/**
 * MessageInput — textarea + send/stop button + optional settings panel.
 */
export default function MessageInput({ onSend, onStop, isStreaming }) {
  const [text, setText] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [params, setParams] = useState({
    maxNewTokens: 200,
    temperature: 1.0,
    topK: 50,
  })
  const textareaRef = useRef(null)

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [text])

  const handleSend = () => {
    if (!text.trim() || isStreaming) return
    onSend(text.trim(), params)
    setText('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-gray-700/50 bg-chat-bg/95 backdrop-blur px-4 py-4">
      {/* Settings panel */}
      {showSettings && (
        <div className="max-w-3xl mx-auto mb-3 bg-gray-800 rounded-xl p-4 grid grid-cols-3 gap-4">
          <SettingSlider
            label="Max tokens"
            min={10} max={500} step={10}
            value={params.maxNewTokens}
            onChange={(v) => setParams((p) => ({ ...p, maxNewTokens: v }))}
          />
          <SettingSlider
            label={`Temperature: ${params.temperature.toFixed(2)}`}
            min={0.1} max={2.0} step={0.05}
            value={params.temperature}
            onChange={(v) => setParams((p) => ({ ...p, temperature: v }))}
          />
          <SettingSlider
            label={`Top-k: ${params.topK}`}
            min={1} max={200} step={1}
            value={params.topK}
            onChange={(v) => setParams((p) => ({ ...p, topK: v }))}
          />
        </div>
      )}

      {/* Main input row */}
      <div className="max-w-3xl mx-auto relative">
        <div className="flex items-end gap-2 bg-chat-input rounded-2xl border border-gray-600/50 px-4 py-3 focus-within:border-gray-500 transition-colors">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message NanoGPT…"
            rows={1}
            disabled={isStreaming}
            className="flex-1 bg-transparent text-gray-100 placeholder-gray-500 resize-none outline-none text-sm leading-relaxed disabled:opacity-50"
          />
          <div className="flex gap-1.5 pb-0.5">
            <button
              onClick={() => setShowSettings((v) => !v)}
              title="Generation settings"
              className="p-1.5 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700 transition-colors"
            >
              <Sliders size={16} />
            </button>
            {isStreaming ? (
              <button
                onClick={onStop}
                title="Stop generation"
                className="p-1.5 rounded-lg bg-gray-600 hover:bg-gray-500 text-white transition-colors"
              >
                <Square size={16} />
              </button>
            ) : (
              <button
                onClick={handleSend}
                disabled={!text.trim()}
                title="Send (Enter)"
                className="p-1.5 rounded-lg bg-white hover:bg-gray-200 text-gray-900 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                <ArrowUp size={16} />
              </button>
            )}
          </div>
        </div>
        <p className="text-center text-xs text-gray-600 mt-2">
          NanoGPT may produce inaccurate text. GPT-2 with Paged Attention KV Cache.
        </p>
      </div>
    </div>
  )
}

function SettingSlider({ label, min, max, step, value, onChange }) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-emerald-500"
      />
      <span className="text-xs text-gray-500">{value}</span>
    </div>
  )
}
