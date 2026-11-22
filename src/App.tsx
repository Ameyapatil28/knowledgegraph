import React, { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { WelcomeScreen } from './components/WelcomeScreen';
import { MessageItem } from './components/MessageItem';
import { ChatInput } from './components/ChatInput';
import { InsightsDashboard } from './components/InsightsDashboard';
import { useChat } from './hooks/useChat';
import { getHealth } from './services/api';
import { HealthResponse } from './types/api';
import { Menu, RotateCcw, Trash2, Sparkles, Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<'chat' | 'insights'>('chat');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const {
    conversations,
    activeConversation,
    messages,
    isLoading,
    lastFailedQuestion,
    sendMessage,
    startNewChat,
    selectConversation,
    retryLastQuestion,
    clearCurrentConversation,
    deleteConversation,
  } = useChat();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Poll backend health on startup and every 30 seconds
  useEffect(() => {
    const fetchHealth = () => {
      getHealth()
        .then((data) => setHealth(data))
        .catch(() => {
          setHealth({
            status: 'degraded',
            version: '1.0.0',
            graph_loaded: false,
            gemini_configured: false,
            gemini_model: 'gemini-2.5-flash',
            gemini_mode: 'MOCK',
            pricing_tier: 'free',
          });
        });
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSelectQuestion = (q: string) => {
    setActiveTab('chat');
    sendMessage(q);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-gray-100">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={activeConversation?.id || null}
        onSelectConversation={selectConversation}
        onNewChat={startNewChat}
        onDeleteConversation={deleteConversation}
        onSelectQuestion={handleSelectQuestion}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        health={health}
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />

      {/* Main Workspace */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Navbar */}
        <header className="h-14 shrink-0 border-b border-surface-border px-4 flex items-center justify-between bg-surface/50 backdrop-blur-md z-10">
          <div className="flex items-center space-x-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-1.5 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
                title="Open Sidebar"
              >
                <Menu className="w-5 h-5" />
              </button>
            )}

            <div className="flex items-center space-x-2">
              <span className="font-semibold text-sm text-gray-200">
                {activeTab === 'insights'
                  ? 'Knowledge Graph Insights'
                  : activeConversation?.title || 'New Conversation'}
              </span>
              {activeTab === 'chat' && messages.length > 0 && (
                <span className="text-[11px] text-gray-500 font-mono">
                  ({messages.length} {messages.length === 1 ? 'message' : 'messages'})
                </span>
              )}
            </div>
          </div>

          {/* Right Header Actions */}
          <div className="flex items-center space-x-2">
            {activeTab === 'chat' && messages.length > 0 && (
              <button
                onClick={clearCurrentConversation}
                className="flex items-center space-x-1 px-2.5 py-1 text-xs text-gray-400 hover:text-red-400 hover:bg-gray-800/60 rounded-lg transition-colors border border-transparent hover:border-gray-700"
                title="Clear current messages"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Clear</span>
              </button>
            )}

            {lastFailedQuestion && (
              <button
                onClick={retryLastQuestion}
                className="flex items-center space-x-1 px-2.5 py-1 text-xs text-amber-300 hover:text-amber-200 bg-amber-950/60 hover:bg-amber-900/60 rounded-lg transition-colors border border-amber-800/60"
                title="Retry failed query"
              >
                <RotateCcw className="w-3.5 h-3.5 text-amber-400" />
                <span>Retry</span>
              </button>
            )}
          </div>
        </header>

        {/* Workspace Body */}
        {activeTab === 'insights' ? (
          <InsightsDashboard health={health} />
        ) : (
          <div className="flex-1 flex flex-col h-[calc(100%-3.5rem)] overflow-hidden">
            {/* Messages / Welcome View */}
            <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-4">
              {messages.length === 0 ? (
                <WelcomeScreen onSelectQuestion={handleSelectQuestion} />
              ) : (
                <div className="max-w-4xl mx-auto w-full pt-2">
                  {messages.map((msg) => (
                    <MessageItem key={msg.id} message={msg} />
                  ))}

                  {/* Loading Skeleton */}
                  {isLoading && (
                    <div className="flex justify-start mb-5">
                      <div className="max-w-[85%] w-full flex items-start space-x-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shrink-0 text-white shadow animate-subtle-pulse">
                          <Loader2 className="w-4 h-4 animate-spin" />
                        </div>
                        <div className="flex-1 bg-surface border border-surface-border rounded-2xl rounded-tl-sm p-4 space-y-2.5">
                          <div className="flex items-center space-x-2 text-xs text-indigo-400 font-mono">
                            <Sparkles className="w-3.5 h-3.5 animate-spin" />
                            <span>Translating natural query & executing against Knowledge Graph...</span>
                          </div>
                          <div className="h-3.5 bg-gray-800/80 rounded w-3/4 animate-pulse" />
                          <div className="h-3.5 bg-gray-800/60 rounded w-5/6 animate-pulse" />
                          <div className="h-3.5 bg-gray-800/40 rounded w-1/2 animate-pulse" />
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={messagesEndRef} />
                </div>
              )}
            </div>

            {/* Bottom Input Box */}
            <ChatInput onSendMessage={sendMessage} isLoading={isLoading} />
          </div>
        )}
      </div>
    </div>
  );
};
