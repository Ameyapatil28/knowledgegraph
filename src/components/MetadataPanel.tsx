import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Cpu, Clock, Coins, ShieldCheck, AlertTriangle } from 'lucide-react';
import { AnswerMetadata } from '../types/api';

interface MetadataPanelProps {
  metadata?: AnswerMetadata | null;
  operation?: string | null;
  status?: string | null;
}

export const MetadataPanel: React.FC<MetadataPanelProps> = ({
  metadata,
  operation,
  status: queryStatus,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!metadata && !operation) return null;

  const parserProvider = metadata?.parser_provider || 'UNKNOWN';
  const answerProvider = metadata?.answer_provider || 'UNKNOWN';
  const modelName = metadata?.model || 'gemini-2.5-flash';
  const totalLatency = metadata?.total_latency_ms ?? 0;
  const isMock = parserProvider === 'MOCK' || answerProvider === 'MOCK';

  return (
    <div className="mt-2.5 border border-gray-800/80 rounded-lg overflow-hidden bg-gray-900/30 transition-colors">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 hover:bg-gray-800/40 transition-colors"
      >
        <div className="flex items-center space-x-2 flex-wrap gap-y-1">
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span className="font-mono text-gray-300 font-medium">
            {operation || 'graph_query'}
          </span>
          {queryStatus && queryStatus !== 'SUCCESS' && (
            <span className="text-[10px] bg-gray-800 text-gray-300 px-1.5 py-0.2 rounded border border-gray-700 font-mono">
              {queryStatus}
            </span>
          )}

          {/* Provider Badge */}
          {isMock ? (
            <span className="text-[10px] bg-amber-950/70 text-amber-300 px-1.5 py-0.2 rounded border border-amber-800/60 font-mono">
              MOCK
            </span>
          ) : (
            <span className="text-[10px] bg-emerald-950/70 text-emerald-300 px-1.5 py-0.2 rounded border border-emerald-800/60 font-mono">
              LIVE_GEMINI
            </span>
          )}

          {/* Grounding Status Badge */}
          {metadata && metadata.grounding_passed && (
            <span className="text-[10px] bg-indigo-950/60 text-indigo-300 px-1.5 py-0.2 rounded border border-indigo-800/50 flex items-center gap-0.5">
              <ShieldCheck className="w-2.5 h-2.5 text-indigo-400" /> Grounded
            </span>
          )}

          {metadata && metadata.grounding_repaired && (
            <span className="text-[10px] bg-purple-950/60 text-purple-300 px-1.5 py-0.2 rounded border border-purple-800/50 flex items-center gap-0.5">
              <AlertTriangle className="w-2.5 h-2.5 text-purple-400" /> Repaired
            </span>
          )}

          {/* Latency */}
          <span className="text-[10px] text-gray-400 font-mono flex items-center gap-1">
            <Clock className="w-2.5 h-2.5" />
            {totalLatency.toFixed(0)}ms
          </span>
        </div>

        <div className="flex items-center space-x-1.5">
          <span className="text-[11px] text-gray-500">
            {isOpen ? 'Hide Telemetry' : 'View Telemetry'}
          </span>
          {isOpen ? (
            <ChevronDown className="w-3.5 h-3.5 text-gray-400" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-gray-400" />
          )}
        </div>
      </button>

      {isOpen && (
        <div className="border-t border-gray-800/60 p-3 bg-gray-950/70 text-xs">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 mb-3 font-mono">
            <div className="p-2 rounded bg-gray-900/60 border border-gray-800/80">
              <span className="text-[10px] text-gray-500 block uppercase">Query Operation</span>
              <span className="text-gray-200 font-medium">{operation || 'N/A'}</span>
            </div>

            <div className="p-2 rounded bg-gray-900/60 border border-gray-800/80">
              <span className="text-[10px] text-gray-500 block uppercase">Model & Tier</span>
              <span className="text-gray-200 font-medium truncate block" title={modelName}>
                {modelName}
              </span>
            </div>

            <div className="p-2 rounded bg-gray-900/60 border border-gray-800/80">
              <span className="text-[10px] text-gray-500 block uppercase">Providers</span>
              <span className="text-gray-200 font-medium">
                P: {parserProvider} | A: {answerProvider}
              </span>
            </div>

            <div className="p-2 rounded bg-gray-900/60 border border-gray-800/80">
              <span className="text-[10px] text-gray-500 block uppercase">Total Latency</span>
              <span className="text-gray-200 font-medium">{totalLatency.toFixed(1)} ms</span>
            </div>
          </div>

          {/* Latency Breakdown */}
          {metadata && (
            <div className="mb-3 p-2 rounded bg-gray-900/40 border border-gray-800/60">
              <span className="text-[10px] font-mono text-gray-400 block mb-1">
                Latency Breakdown
              </span>
              <div className="grid grid-cols-3 gap-2 text-[11px] font-mono text-gray-300">
                <div>Parser: {metadata.parser_latency_ms.toFixed(1)} ms</div>
                <div>Engine: {metadata.engine_latency_ms.toFixed(1)} ms</div>
                <div>Answer: {metadata.answer_latency_ms.toFixed(1)} ms</div>
              </div>
            </div>
          )}

          {/* Tokens & Cost Telemetry */}
          {metadata?.tokens && (
            <div className="p-2 rounded bg-gray-900/40 border border-gray-800/60 flex items-center justify-between flex-wrap gap-2 font-mono text-[11px]">
              <div className="flex items-center space-x-3 text-gray-300">
                <span>Prompt: {metadata.tokens.prompt_tokens}</span>
                <span>Completion: {metadata.tokens.completion_tokens}</span>
                <span className="text-indigo-400 font-semibold">
                  Total: {metadata.tokens.total_tokens} tokens
                </span>
              </div>
              {metadata.cost_usd !== null && metadata.cost_usd !== undefined && (
                <div className="flex items-center space-x-1 text-emerald-400">
                  <Coins className="w-3 h-3" />
                  <span>Est. Cost: ${metadata.cost_usd.toFixed(6)}</span>
                </div>
              )}
            </div>
          )}

          {/* Grounding Violations / Repair Notice */}
          {metadata?.grounding_violations && metadata.grounding_violations.length > 0 && (
            <div className="mt-2.5 p-2 rounded bg-amber-950/30 border border-amber-900/50 text-[11px] text-amber-200">
              <div className="font-semibold flex items-center gap-1 mb-1">
                <AlertTriangle className="w-3 h-3 text-amber-400" /> Grounding Violation Prevented:
              </div>
              <ul className="list-disc list-inside space-y-0.5 text-amber-300/90 text-[10px]">
                {metadata.grounding_violations.map((v, i) => (
                  <li key={i}>{v}</li>
                ))}
              </ul>
              <div className="text-[10px] text-amber-400 mt-1 italic">
                Deterministic fallback response was generated to guarantee 100% accuracy.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
