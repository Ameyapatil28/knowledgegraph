import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Sparkles } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  isLoading,
  disabled = false,
}) => {
  const [text, setText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [text]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    const clean = text.trim();
    if (!clean || isLoading || disabled) return;
    onSendMessage(clean);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-4 pt-2">
      <div className="relative flex items-end bg-surface border border-surface-border/90 rounded-2xl p-2 shadow-xl shadow-black/40 focus-within:border-indigo-500/80 focus-within:ring-1 focus-within:ring-indigo-500/30 transition-all">
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about 4,803 TMDB movies, actors, directors, box office, or rankings..."
          disabled={isLoading || disabled}
          className="w-full bg-transparent text-gray-100 placeholder-gray-500 text-sm resize-none focus:outline-none px-3 py-1.5 max-h-36 leading-relaxed disabled:opacity-60"
        />

        <button
          onClick={handleSubmit}
          disabled={!text.trim() || isLoading || disabled}
          className="shrink-0 w-9 h-9 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-700 text-white flex items-center justify-center hover:from-indigo-500 hover:to-indigo-600 active:scale-95 transition-all disabled:opacity-40 disabled:hover:from-indigo-600 disabled:hover:to-indigo-700 disabled:cursor-not-allowed shadow-md shadow-indigo-950/40"
          title="Send message (Enter)"
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin text-white" />
          ) : (
            <Send className="w-4 h-4 text-white" />
          )}
        </button>
      </div>

      <div className="flex items-center justify-between text-[11px] text-gray-500 px-3 pt-2">
        <div className="flex items-center space-x-1">
          <Sparkles className="w-3 h-3 text-indigo-400" />
          <span>Grounded with NetworkX Knowledge Graph &middot; Strict Zero-Hallucination Guardrails</span>
        </div>
        <span>Press <kbd className="font-mono text-gray-400 bg-gray-900 px-1 py-0.5 rounded border border-gray-800">Enter</kbd> to send, <kbd className="font-mono text-gray-400 bg-gray-900 px-1 py-0.5 rounded border border-gray-800">Shift+Enter</kbd> for newline</span>
      </div>
    </div>
  );
};
