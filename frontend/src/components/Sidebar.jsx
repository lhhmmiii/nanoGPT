import { Plus, MessageSquare, Trash2 } from 'lucide-react'

/**
 * Sidebar — conversation list (currently single session) + new chat button.
 */
export default function Sidebar({ onNewChat, isOpen, messageCount }) {
  if (!isOpen) return null

  return (
    <aside className="w-64 flex-shrink-0 bg-sidebar flex flex-col h-full border-r border-gray-700/40">
      {/* Header */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-gray-300 hover:bg-sidebar-hover hover:text-white transition-colors"
        >
          <Plus size={16} />
          New chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-3 pb-3">
        <p className="text-xs text-gray-500 px-3 py-2 uppercase tracking-wider font-semibold">
          Today
        </p>
        {messageCount > 0 && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-700/50 text-sm text-gray-200 cursor-default">
            <MessageSquare size={14} className="text-gray-400 flex-shrink-0" />
            <span className="truncate">Current conversation</span>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-700/40">
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-gray-500">
          <div className="w-6 h-6 rounded-full bg-violet-600 flex items-center justify-center text-white text-xs font-bold">
            U
          </div>
          <span>User</span>
        </div>
      </div>
    </aside>
  )
}
