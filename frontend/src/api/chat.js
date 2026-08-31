/**
 * Chat API client
 * Wraps REST + SSE calls to the FastAPI backend.
 */

const BASE = '/api'

/**
 * POST /api/chat — non-streaming, returns full text.
 * @param {object} params
 * @param {string} params.prompt
 * @param {number} params.maxNewTokens
 * @param {number} params.temperature
 * @param {number|null} params.topK
 * @returns {Promise<{generated_text: string, prompt_tokens: number, generated_tokens: number}>}
 */
export async function sendChat({ prompt, maxNewTokens = 200, temperature = 1.0, topK = 50 }) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt,
      max_new_tokens: maxNewTokens,
      temperature,
      top_k: topK,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

/**
 * GET /api/chat/stream — SSE streaming.
 *
 * @param {object} params
 * @param {string}   params.prompt
 * @param {number}   params.maxNewTokens
 * @param {number}   params.temperature
 * @param {number|null} params.topK
 * @param {(token: string) => void} params.onToken   — called for each token
 * @param {(stats: object) => void} params.onDone    — called when done
 * @param {(err: string) => void}   params.onError   — called on error
 * @returns {() => void} abort function
 */
export function streamChat({ prompt, maxNewTokens = 200, temperature = 1.0, topK = 50, onToken, onDone, onError }) {
  const url = new URL(`${window.location.origin}${BASE}/chat/stream`)
  url.searchParams.set('prompt', prompt)
  url.searchParams.set('max_new_tokens', maxNewTokens)
  url.searchParams.set('temperature', temperature)
  if (topK != null) url.searchParams.set('top_k', topK)

  const es = new EventSource(url.toString())

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.error) {
        onError?.(data.error)
        es.close()
      } else if (data.done) {
        onDone?.(data.stats)
        es.close()
      } else if (data.token !== undefined) {
        onToken?.(data.token)
      }
    } catch {
      // ignore malformed
    }
  }

  es.onerror = () => {
    onError?.('Connection lost')
    es.close()
  }

  return () => es.close()
}

/**
 * GET /api/health
 */
export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`)
  if (!res.ok) throw new Error('API not reachable')
  return res.json()
}
