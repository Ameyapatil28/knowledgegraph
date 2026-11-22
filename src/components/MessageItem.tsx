import React, { useState } from 'react';
import { Copy, Check, User, Bot, AlertCircle, HelpCircle, Ban } from 'lucide-react';
import { ChatMessage } from '../types/api';
import { MetadataPanel } from './MetadataPanel';
import { JsonViewer } from './JsonViewer';

interface MessageItemProps {
  message: ChatMessage;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.sender === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Status badge config
  const renderStatusBadge = () => {
    if (!message.status || message.status === 'SUCCESS') return null;

    if (message.status === 'CLARIFICATION_REQUIRED') {
      return (
        <span className="inline-flex items-center gap-1 text-[11px] bg-blue-950/80 text-blue-300 px-2 py-0.5 rounded border border-blue-800/60 font-medium mb-2">
          <HelpCircle className="w-3 h-3 text-blue-400" /> Clarification Required
        </span>
      );
    }

    if (message.status === 'UNSUPPORTED') {
      return (
        <span className="inline-flex items-center gap-1 text-[11px] bg-amber-950/80 text-amber-300 px-2 py-0.5 rounded border border-amber-800/60 font-medium mb-2">
          <Ban className="w-3 h-3 text-amber-400" /> Query Unsupported
        </span>
      );
    }

    if (message.status === 'NO_RESULTS') {
      return (
        <span className="inline-flex items-center gap-1 text-[11px] bg-gray-800/80 text-gray-300 px-2 py-0.5 rounded border border-gray-700 font-medium mb-2">
          <AlertCircle className="w-3 h-3 text-gray-400" /> No Records Found
        </span>
      );
    }

    if (message.status === 'ERROR') {
      return (
        <span className="inline-flex items-center gap-1 text-[11px] bg-red-950/80 text-red-300 px-2 py-0.5 rounded border border-red-800/60 font-medium mb-2">
          <AlertCircle className="w-3 h-3 text-red-400" /> System Error
        </span>
      );
    }

    return null;
  };

  /**
   * Simple, reliable markdown-like text renderer
   */
  const renderFormattedText = (text: string) => {
    const lines = text.split('\n');
    return (
      <div className="space-y-1.5 leading-relaxed text-gray-200 text-sm">
        {lines.map((line, idx) => {
          if (!line.trim()) {
            return <div key={idx} className="h-1.5" />;
          }

          // Bullet point
          if (line.trim().startsWith('•') || line.trim().startsWith('-')) {
            const content = line.trim().replace(/^[•-]\s*/, '');
            return (
              <div key={idx} className="flex items-start space-x-2 pl-2">
                <span className="text-indigo-400 font-bold">•</span>
                <span>{renderInlineStyles(content)}</span>
              </div>
            );
          }

          // Numbered list item: "1. **Title** ..."
          const numMatch = line.trim().match(/^(\d+)\.\s+(.*)/);
          if (numMatch) {
            return (
              <div key={idx} className="flex items-start space-x-2 pl-2">
                <span className="text-indigo-400 font-semibold font-mono text-xs w-4">
                  {numMatch[1]}.
                </span>
                <span>{renderInlineStyles(numMatch[2])}</span>
              </div>
            );
          }

          return <p key={idx}>{renderInlineStyles(line)}</p>;
        })}
      </div>
    );
  };

  /**
   * Helper to render **bold**, *italic*, and `code` spans
   */
  const renderInlineStyles = (content: string) => {
    // Split by markdown bold markers (**text**)
    const parts = content.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="text-white font-semibold">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('*') && part.endsWith('*')) {
        return (
          <em key={i} className="text-indigo-200 italic">
            {part.slice(1, -1)}
          </em>
        );
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={i} className="px-1.5 py-0.5 rounded bg-gray-900 text-indigo-300 font-mono text-xs border border-gray-800">
            {part.slice(1, -1)}
          </code>
        );
      }
      return part;
    });
  };

  if (isUser) {
    return (
      <div className="flex justify-end mb-4 group">
        <div className="max-w-[85%] sm:max-w-[75%] flex items-start space-x-2">
          <div className="bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 shadow-md shadow-indigo-950/40 text-sm">
            <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
            <div className="text-[10px] text-indigo-200/80 text-right mt-1 font-mono">
              {message.timestamp}
            </div>
          </div>
          <div className="w-8 h-8 rounded-full bg-indigo-700 flex items-center justify-center shrink-0 border border-indigo-500/30 text-white shadow">
            <User className="w-4 h-4" />
          </div>
        </div>
      </div>
    );
  }

  // AI Response Card
  return (
    <div className="flex justify-start mb-5 group">
      <div className="max-w-[95%] sm:max-w-[85%] w-full flex items-start space-x-3">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shrink-0 border border-indigo-400/40 text-white shadow-md shadow-indigo-950/50 mt-1">
          <Bot className="w-4 h-4" />
        </div>

        <div className="flex-1 bg-surface border border-surface-border/70 rounded-2xl rounded-tl-sm p-4 shadow-lg shadow-black/20">
          {renderStatusBadge()}

          <div className="relative">
            {renderFormattedText(message.text)}

            <button
              onClick={handleCopy}
              className="absolute top-0 right-0 p-1.5 rounded-md text-gray-500 hover:text-gray-200 hover:bg-gray-800 transition-colors opacity-0 group-hover:opacity-100"
              title="Copy answer"
            >
              {copied ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
          </div>

          {/* Technical Metadata Panel */}
          <MetadataPanel
            metadata={message.metadata}
            operation={message.operation}
            status={message.status}
          />

          {/* Raw Knowledge Graph Results */}
          {message.results && <JsonViewer data={message.results} />}

          <div className="text-[10px] text-gray-600 text-right mt-2 font-mono">
            {message.timestamp}
          </div>
        </div>
      </div>
    </div>
  );
};
