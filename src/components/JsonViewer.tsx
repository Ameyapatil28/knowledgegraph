import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Copy, Check, Database } from 'lucide-react';

interface JsonViewerProps {
  data: any;
  title?: string;
}

export const JsonViewer: React.FC<JsonViewerProps> = ({
  data,
  title = 'View Knowledge Graph Results',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!data) return null;

  const jsonString = JSON.stringify(data, null, 2);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const recordCount = Array.isArray(data)
    ? data.length
    : typeof data === 'object'
    ? Object.keys(data).length
    : 1;

  return (
    <div className="mt-3 border border-gray-800 rounded-lg overflow-hidden bg-gray-950/60 transition-colors">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-900/50 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <Database className="w-3.5 h-3.5 text-indigo-400" />
          <span className="font-medium text-gray-300">{title}</span>
          <span className="text-[10px] bg-indigo-950/80 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-800/50">
            {recordCount} {recordCount === 1 ? 'record' : 'records'}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-[11px] text-gray-500">
            {isOpen ? 'Click to collapse' : 'Click to expand raw JSON'}
          </span>
          {isOpen ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {isOpen && (
        <div className="border-t border-gray-800/80 p-3 bg-gray-950">
          <div className="flex justify-between items-center mb-2">
            <span className="text-[10px] uppercase font-mono tracking-wider text-gray-500">
              Raw NetworkX Results
            </span>
            <button
              onClick={handleCopy}
              className="flex items-center space-x-1 text-xs text-gray-400 hover:text-white px-2 py-1 rounded bg-gray-900 border border-gray-800 transition-colors"
              title="Copy JSON"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-400 text-[10px]">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span className="text-[10px]">Copy JSON</span>
                </>
              )}
            </button>
          </div>
          <pre className="text-[11px] font-mono text-gray-300 overflow-x-auto max-h-72 p-2.5 rounded bg-gray-900/70 border border-gray-800 leading-relaxed">
            {jsonString}
          </pre>
        </div>
      )}
    </div>
  );
};
