import { useEffect, useRef } from 'react'
import Message from './Message'
import { Bot } from 'lucide-react'

/**
 * ChatWindow — scrollable message list.
 */
export default function ChatWindow({ messages, isStreaming }) {
  const bottomRef = useRef(null)

  // Auto-scroll to bottom on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center px-4">
        <div className="w-16 h-16 rounded-2xl bg-emerald-600 flex items-center justify-center mb-4">
          <Bot size={32} className="text-white" />
        </div>
        <h1 className="text-2xl font-semibold text-gray-100 mb-2">NanoGPT Chat</h1>
        <p className="text-gray-400 text-sm max-w-sm">
          GPT-2 with Paged Attention KV Cache. Type a message to start a conversation.
        </p>
        <div className="mt-6 grid grid-cols-2 gap-3 max-w-md w-full">
          {SUGGESTIONS.map((s) => (
            <SuggestionCard key={s} text={s} />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="divide-y divide-gray-700/40">
        {messages.map((msg) => (
          <Message key={msg.id} message={msg} />
        ))}
      </div>
      {/* Spacer so last message isn't hidden behind input */}
      <div ref={bottomRef} className="h-32" />
    </div>
  )
}

const SUGGESTIONS = [
  'Once upon a time in a kingdom far away',
  'The future of artificial intelligence is',
  'Write a short poem about autumn',
  'Explain quantum computing in simple terms',
]

function SuggestionCard({ text }) {
  return (
    <div className="border border-gray-600 rounded-xl p-3 text-xs text-gray-400 text-left cursor-default hover:border-gray-500 hover:text-gray-300 transition-colors">
      {text}
    </div>
  )
}
