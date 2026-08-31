import ReactMarkdown from 'react-markdown'
import { Bot, User } from 'lucide-react'

/**
 * Single chat message bubble.
 * Props:
 *   message : { role, content, streaming, error }
 */
export default function Message({ message }) {
  const isUser = message.role === 'user'

  return (
    <div
      className={`w-full py-6 px-4 md:px-0 ${
        isUser ? 'bg-transparent' : 'bg-transparent'
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-4 items-start">
        {/* Avatar */}
        <div
          className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-semibold ${
            isUser
              ? 'bg-violet-600'
              : 'bg-emerald-600'
          }`}
        >
          {isUser ? <User size={16} /> : <Bot size={16} />}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className={`text-xs font-semibold mb-1 ${isUser ? 'text-violet-400' : 'text-emerald-400'}`}>
            {isUser ? 'You' : 'NanoGPT'}
          </p>
          <div
            className={`prose-chat text-sm leading-relaxed ${
              message.error
                ? 'text-red-400'
                : 'text-gray-100 dark:text-gray-100'
            } ${message.streaming ? 'typing-cursor' : ''}`}
          >
            {isUser ? (
              <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <ReactMarkdown>{message.content || (message.streaming ? ' ' : '')}</ReactMarkdown>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
