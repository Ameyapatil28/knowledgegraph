import React from 'react';
import {
  Film,
  Plus,
  MessageSquare,
  Trash2,
  Activity,
  Layers,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  BarChart3,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { Conversation, HealthResponse } from '../types/api';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
  onSelectQuestion: (question: string) => void;
  isOpen: boolean;
  onToggle: () => void;
  health: HealthResponse | null;
  activeTab: 'chat' | 'insights';
  onSelectTab: (tab: 'chat' | 'insights') => void;
}

const QUICK_SHORTCUTS = [
  'Top 5 Science Fiction movies',
  'Movies directed by Christopher Nolan',
  'Compare Avatar and Titanic',
  'Highest-grossing movies',
];

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  onSelectQuestion,
  isOpen,
  onToggle,
  health,
  activeTab,
  onSelectTab,
}) => {
  const isHealthy = health?.status === 'healthy';
  const isGraphLoaded = health?.graph_loaded ?? false;
  const isGeminiLive = health?.gemini_mode === 'LIVE_GEMINI';

  if (!isOpen) {
    return (
      <div className="w-14 shrink-0 bg-surface border-r border-surface-border flex flex-col items-center py-4 justify-between transition-all">
        <div className="flex flex-col items-center space-y-4">
          <button
            onClick={onToggle}
            className="w-9 h-9 rounded-xl bg-indigo-950/80 border border-indigo-800/60 text-indigo-400 flex items-center justify-center hover:bg-indigo-900 transition-colors"
            title="Expand Sidebar"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
          <button
            onClick={onNewChat}
            className="w-9 h-9 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-700 text-white flex items-center justify-center hover:opacity-90 transition-opacity"
            title="New Chat"
          >
            <Plus className="w-5 h-5" />
          </button>
          <button
            onClick={() => onSelectTab(activeTab === 'chat' ? 'insights' : 'chat')}
            className={`w-9 h-9 rounded-xl flex items-center justify-center transition-colors ${
              activeTab === 'insights'
                ? 'bg-purple-900/60 text-purple-300 border border-purple-700'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
            }`}
            title="Toggle Insights Dashboard"
          >
            <BarChart3 className="w-5 h-5" />
          </button>
        </div>

        <div className="flex flex-col items-center space-y-2">
          <div
            className={`w-3 h-3 rounded-full ${
              isHealthy ? 'bg-emerald-500' : 'bg-amber-500'
            } ring-4 ring-emerald-500/10`}
            title={isHealthy ? 'Backend Connected' : 'Backend Degraded/Connecting'}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="w-72 shrink-0 bg-surface border-r border-surface-border flex flex-col justify-between h-screen transition-all">
      {/* Top Header */}
      <div>
        <div className="flex items-center justify-between p-4 border-b border-surface-border/60">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-md shadow-indigo-950/60">
              <Film className="w-4 h-4" />
            </div>
            <div>
              <div className="font-bold text-sm tracking-tight text-white flex items-center gap-1.5">
                MovieGraph <span className="text-xs bg-indigo-950 text-indigo-300 px-1 py-0.2 rounded border border-indigo-800/60 font-mono">AI</span>
              </div>
              <div className="text-[10px] text-gray-400">TMDB 5000 Knowledge Graph</div>
            </div>
          </div>

          <button
            onClick={onToggle}
            className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
            title="Collapse Sidebar"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Switcher: Chat vs Insights */}
        <div className="p-3 pb-1">
          <div className="grid grid-cols-2 gap-1 p-1 bg-gray-900/80 rounded-xl border border-gray-800">
            <button
              onClick={() => onSelectTab('chat')}
              className={`flex items-center justify-center space-x-1.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'chat'
                  ? 'bg-indigo-600 text-white shadow'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              <span>Chat</span>
            </button>
            <button
              onClick={() => onSelectTab('insights')}
              className={`flex items-center justify-center space-x-1.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === 'insights'
                  ? 'bg-purple-600 text-white shadow'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Insights</span>
            </button>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="px-3 py-2">
          <button
            onClick={onNewChat}
            className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white text-xs font-medium shadow-md shadow-indigo-950/40 active:scale-[0.99] transition-all"
          >
            <Plus className="w-4 h-4" />
            <span>New Chat</span>
          </button>
        </div>

        {/* Conversations List */}
        <div className="px-3 py-2">
          <div className="text-[10px] font-mono uppercase text-gray-500 font-semibold px-2 mb-1.5 tracking-wider">
            Conversations
          </div>
          <div className="max-h-52 overflow-y-auto space-y-1 pr-1">
            {conversations.length === 0 ? (
              <div className="px-2 py-3 text-xs text-gray-500 italic text-center">
                No active conversations
              </div>
            ) : (
              conversations.map((c) => (
                <div
                  key={c.id}
                  onClick={() => onSelectConversation(c.id)}
                  className={`group flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs cursor-pointer transition-all ${
                    c.id === activeId
                      ? 'bg-indigo-950/70 text-indigo-200 border border-indigo-800/60 font-medium'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate">
                    <MessageSquare className="w-3.5 h-3.5 shrink-0 text-indigo-400" />
                    <span className="truncate">{c.title}</span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteConversation(c.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 text-gray-500 hover:text-red-400 transition-opacity"
                    title="Delete Chat"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Suggested Question Shortcuts */}
        <div className="px-3 py-2">
          <div className="text-[10px] font-mono uppercase text-gray-500 font-semibold px-2 mb-1.5 tracking-wider flex items-center gap-1">
            <Sparkles className="w-2.5 h-2.5 text-indigo-400" /> Suggested Queries
          </div>
          <div className="space-y-1">
            {QUICK_SHORTCUTS.map((q, idx) => (
              <button
                key={idx}
                onClick={() => onSelectQuestion(q)}
                className="w-full text-left px-2.5 py-1.5 rounded-lg text-xs text-gray-400 hover:text-indigo-200 hover:bg-indigo-950/40 border border-transparent hover:border-indigo-800/40 truncate transition-all block"
              >
                &ldquo;{q}&rdquo;
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer System Status Panel */}
      <div className="p-3 border-t border-surface-border/60 bg-gray-950/40 text-xs font-mono space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-gray-400 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-indigo-400" /> Backend
          </span>
          {isHealthy ? (
            <span className="text-[10px] text-emerald-400 flex items-center gap-1 font-semibold">
              <CheckCircle2 className="w-3 h-3" /> Online
            </span>
          ) : (
            <span className="text-[10px] text-amber-400 flex items-center gap-1 font-semibold">
              <AlertCircle className="w-3 h-3" /> Degraded
            </span>
          )}
        </div>

        <div className="flex items-center justify-between">
          <span className="text-[11px] text-gray-400 flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-indigo-400" /> Knowledge Graph
          </span>
          <span className="text-[10px] text-gray-300">
            {isGraphLoaded ? '124.7k nodes' : 'Loading...'}
          </span>
        </div>

        <div className="flex items-center justify-between pt-1 border-t border-gray-800/50">
          <span className="text-[10px] text-gray-500">Provider Mode</span>
          <span
            className={`text-[9px] px-1.5 py-0.2 rounded font-semibold ${
              isGeminiLive
                ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                : 'bg-amber-950 text-amber-300 border border-amber-800/60'
            }`}
          >
            {health?.gemini_mode || 'CHECKING...'}
          </span>
        </div>
      </div>
    </div>
  );
};
