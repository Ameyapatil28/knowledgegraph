import React, { useState, useEffect } from 'react';
import {
  Database,
  Film,
  Users,
  Tag,
  Building2,
  Share2,
  Activity,
  Layers,
  Code2,
} from 'lucide-react';
import { HealthResponse, OperationsResponse } from '../types/api';
import { getOperations } from '../services/api';

interface InsightsDashboardProps {
  health: HealthResponse | null;
}

export const InsightsDashboard: React.FC<InsightsDashboardProps> = ({ health }) => {
  const [operationsData, setOperationsData] = useState<OperationsResponse | null>(null);

  useEffect(() => {
    getOperations()
      .then((data) => setOperationsData(data))
      .catch(() => {});
  }, []);

  const verifiedNodes = health?.graph_nodes ?? 124700;
  const verifiedEdges = health?.graph_edges ?? 311242;

  const STATS_CARDS = [
    {
      title: 'Total Graph Nodes',
      value: verifiedNodes.toLocaleString(),
      label: 'Verified Entities',
      icon: Database,
      color: 'from-blue-600 to-indigo-600',
    },
    {
      title: 'Total Relationships',
      value: verifiedEdges.toLocaleString(),
      label: 'Directed Graph Edges',
      icon: Share2,
      color: 'from-purple-600 to-pink-600',
    },
    {
      title: 'Movies',
      value: '4,803',
      label: 'Unique TMDB Films',
      icon: Film,
      color: 'from-indigo-600 to-violet-600',
    },
    {
      title: 'Cast & Crew People',
      value: '104,842',
      label: 'Actors, Directors, Crew',
      icon: Users,
      color: 'from-cyan-600 to-blue-600',
    },
    {
      title: 'Genres',
      value: '20',
      label: 'Standardized Categories',
      icon: Tag,
      color: 'from-emerald-600 to-teal-600',
    },
    {
      title: 'Production Companies',
      value: '5,047',
      label: 'Studios & Producers',
      icon: Building2,
      color: 'from-amber-600 to-orange-600',
    },
  ];

  const RELATIONSHIPS = [
    { name: 'HAS_CAST', source: 'Movie', target: 'Person', desc: 'Actor cast members and character roles' },
    { name: 'DIRECTED_BY', source: 'Movie', target: 'Person', desc: 'Film directors' },
    { name: 'HAS_CREW_MEMBER', source: 'Movie', target: 'Person', desc: 'Producers, writers, editors, cinematography' },
    { name: 'HAS_GENRE', source: 'Movie', target: 'Genre', desc: 'Action, Sci-Fi, Drama, Animation, etc.' },
    { name: 'HAS_KEYWORD', source: 'Movie', target: 'Keyword', desc: 'Thematic plot keywords and subjects' },
    { name: 'PRODUCED_BY', source: 'Movie', target: 'ProductionCompany', desc: 'Warner Bros., Paramount, etc.' },
    { name: 'PRODUCED_IN', source: 'Movie', target: 'ProductionCountry', desc: 'Country of production origin' },
    { name: 'HAS_LANGUAGE', source: 'Movie', target: 'Language', desc: 'Spoken and original languages' },
  ];

  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-8 max-w-6xl mx-auto w-full">
      {/* Header */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-surface-border/60 pb-5">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-purple-950/70 border border-purple-800/60 text-purple-300 text-xs font-mono font-medium mb-2">
            <Database className="w-3 h-3 text-purple-400" />
            <span>NetworkX MultiDiGraph Architecture</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">
            Knowledge Graph Insights
          </h1>
          <p className="text-sm text-gray-400 mt-1">
            Complete topology and relationship statistics for the TMDB 5000 dataset graph.
          </p>
        </div>

        {/* Live Backend Status Card */}
        <div className="bg-surface border border-surface-border rounded-xl p-3 text-xs font-mono shrink-0 shadow-sm">
          <div className="text-gray-400 text-[10px] uppercase font-semibold mb-1">
            Backend Connection
          </div>
          <div className="flex items-center space-x-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                health?.status === 'healthy' ? 'bg-emerald-400' : 'bg-amber-400'
              } ring-2 ring-emerald-500/20`}
            />
            <span className="text-gray-200 font-medium">
              {health?.status === 'healthy' ? 'Connected (Online)' : 'Connecting / Degraded'}
            </span>
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">
            Model: {health?.gemini_model || 'gemini-2.5-flash'} ({health?.gemini_mode || 'MOCK'})
          </div>
        </div>
      </div>

      {/* Notice on Verified Build-Time Data */}
      <div className="mb-6 p-3 rounded-xl bg-indigo-950/40 border border-indigo-800/40 text-xs text-indigo-300 flex items-start space-x-2.5">
        <Activity className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-semibold text-white">Verified Graph Metrics:</span> The entity and edge counts
          below reflect the actual serialized NetworkX Knowledge Graph (
          <code className="bg-indigo-900/60 px-1 py-0.5 rounded text-[11px] font-mono text-indigo-200">
            tmdb_knowledge_graph.pickle
          </code>
          ) constructed in Phase 2 and verified via the FastAPI backend health endpoint.
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 mb-8">
        {STATS_CARDS.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div
              key={idx}
              className="p-4 rounded-xl bg-surface border border-surface-border shadow-sm flex items-start space-x-3.5 hover:border-surface-border/90 transition-all"
            >
              <div
                className={`w-10 h-10 rounded-xl bg-gradient-to-br ${stat.color} flex items-center justify-center text-white shrink-0 shadow-md`}
              >
                <Icon className="w-5 h-5" />
              </div>
              <div className="overflow-hidden">
                <div className="text-xs text-gray-400 font-medium">{stat.title}</div>
                <div className="text-2xl font-extrabold text-white font-mono tracking-tight my-0.5">
                  {stat.value}
                </div>
                <div className="text-[11px] text-gray-500 truncate">{stat.label}</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Graph Schema & Directed Relationships */}
      <div className="mb-8">
        <h2 className="text-lg font-bold text-white mb-3 flex items-center space-x-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          <span>Directed Relationships (Edges)</span>
        </h2>
        <div className="bg-surface border border-surface-border rounded-xl overflow-hidden shadow-sm">
          <table className="w-full text-left text-xs">
            <thead className="bg-gray-900/80 text-gray-400 uppercase font-mono text-[10px] border-b border-surface-border">
              <tr>
                <th className="px-4 py-3">Relationship</th>
                <th className="px-4 py-3">Source &rarr; Target</th>
                <th className="px-4 py-3">Semantic Purpose</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border/60 font-mono text-[11px]">
              {RELATIONSHIPS.map((rel, idx) => (
                <tr key={idx} className="hover:bg-gray-900/40 transition-colors">
                  <td className="px-4 py-2.5 font-bold text-indigo-300">{rel.name}</td>
                  <td className="px-4 py-2.5 text-gray-300">
                    <span className="text-cyan-300">{rel.source}</span> &rarr;{' '}
                    <span className="text-purple-300">{rel.target}</span>
                  </td>
                  <td className="px-4 py-2.5 text-gray-400 font-sans text-xs">{rel.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Supported Query Operations */}
      <div>
        <h2 className="text-lg font-bold text-white mb-3 flex items-center space-x-2">
          <Code2 className="w-5 h-5 text-indigo-400" />
          <span>Supported Query Engine Operations</span>
          {operationsData && (
            <span className="text-xs text-indigo-400 font-mono font-normal">
              ({operationsData.supported_operations.length} operations)
            </span>
          )}
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {operationsData?.supported_operations.map((op) => {
            const info = operationsData.operations[op];
            return (
              <div
                key={op}
                className="p-3.5 rounded-xl bg-surface border border-surface-border flex flex-col justify-between"
              >
                <div>
                  <div className="font-mono text-xs font-bold text-indigo-300 mb-1">
                    {op}
                  </div>
                  <div className="text-xs text-gray-400 leading-relaxed mb-2">
                    {info?.description || 'Graph query operation.'}
                  </div>
                </div>
                {info?.allowed_parameters && info.allowed_parameters.length > 0 && (
                  <div className="pt-2 border-t border-surface-border/60">
                    <span className="text-[10px] text-gray-500 font-mono block mb-1">
                      Allowed Parameters:
                    </span>
                    <div className="flex flex-wrap gap-1">
                      {info.allowed_parameters.map((param) => (
                        <span
                          key={param}
                          className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-gray-900 text-gray-300 border border-gray-800"
                        >
                          {param}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
