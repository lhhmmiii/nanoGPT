import { useCallback, useRef, useState } from 'react'
import { streamChat } from '../api/chat'

/**
 * useChat — manages conversation state and streaming generation.
 *
 * Returns:
 *   conversations : Message[]
 *   isStreaming   : bool
 *   error         : string | null
 *   sendMessage   : (text, params) => void
 *   clearChat     : () => void
 */
export function useChat() {
  const [messages, setMessages] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)

  /** Build the full prompt from conversation history */
  const buildPrompt = (history, userText) => {
    const lines = history.map((m) =>
      m.role === 'user' ? `User: ${m.content}` : `Assistant: ${m.content}`
    )
    lines.push(`User: ${userText}`)
    lines.push('Assistant:')
    return lines.join('\n')
  }

  const sendMessage = useCallback(
    (text, { maxNewTokens = 200, temperature = 1.0, topK = 50 } = {}) => {
      if (!text.trim() || isStreaming) return

      // Append user message
      const userMsg = { id: Date.now(), role: 'user', content: text.trim() }
      const assistantId = Date.now() + 1
      const assistantMsg = { id: assistantId, role: 'assistant', content: '', streaming: true }

      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setIsStreaming(true)
      setError(null)

      const prompt = buildPrompt(messages, text.trim())

      abortRef.current = streamChat({
        prompt,
        maxNewTokens,
        temperature,
        topK,
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m
            )
          )
        },
        onDone: () => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, streaming: false } : m
            )
          )
          setIsStreaming(false)
        },
        onError: (err) => {
          setError(err)
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `⚠️ Error: ${err}`, streaming: false, error: true }
                : m
            )
          )
          setIsStreaming(false)
        },
      })
    },
    [messages, isStreaming]
  )

  const stopGeneration = useCallback(() => {
    abortRef.current?.()
    setIsStreaming(false)
    setMessages((prev) =>
      prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
    )
  }, [])

  const clearChat = useCallback(() => {
    abortRef.current?.()
    setMessages([])
    setIsStreaming(false)
    setError(null)
  }, [])

  return { messages, isStreaming, error, sendMessage, stopGeneration, clearChat }
}
