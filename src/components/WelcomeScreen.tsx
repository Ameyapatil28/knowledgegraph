import React from 'react';
import { Film, Sparkles, Network, TrendingUp, Users, GitCompare } from 'lucide-react';

interface WelcomeScreenProps {
  onSelectQuestion: (question: string) => void;
}

const SUGGESTED_QUESTIONS = [
  {
    icon: Film,
    title: 'Movie Details',
    question: 'Tell me about The Dark Knight',
    description: 'Rating, directors, budget, revenue, and plot overview',
  },
  {
    icon: Sparkles,
    title: 'Ranked Genre Filter',
    question: 'Show me the top 5 highest-rated Science Fiction movies',
    description: 'Deterministic NetworkX ranking with secondary tie-breakers',
  },
  {
    icon: Users,
    title: 'Director Filmography',
    question: 'Which movies were directed by Christopher Nolan?',
    description: 'Explore filmography connected by DIRECTED_BY edges',
  },
  {
    icon: TrendingUp,
    title: 'Box Office Leaders',
    question: 'What are the 10 highest-grossing movies?',
    description: 'Ranked global box office from dataset records',
  },
  {
    icon: Network,
    title: 'Common Cast',
    question: 'Which actors appeared in both The Dark Knight and Inception?',
    description: 'Intersecting cast nodes via HAS_CAST graph traversal',
  },
  {
    icon: GitCompare,
    title: 'Movie Comparison',
    question: 'Compare Avatar and Titanic',
    description: 'Side-by-side comparison of budget, revenue, and ratings',
  },
];

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSelectQuestion }) => {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-4 sm:p-6 max-w-4xl mx-auto w-full text-center my-auto">
      {/* Hero Badge */}
      <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-950/70 border border-indigo-800/60 text-indigo-300 text-xs font-medium mb-4 shadow-sm">
        <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
        <span>Hybrid Graph RAG &middot; NetworkX MultiDiGraph &middot; Gemini 2.5 Flash</span>
      </div>

      {/* Main Title */}
      <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight mb-3">
        MovieGraph <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-purple-400">AI</span>
      </h1>

      <p className="text-base sm:text-lg text-gray-300 max-w-2xl mb-2 font-normal">
        Explore <span className="text-indigo-400 font-semibold">4,803 movies</span> through an AI-powered Knowledge Graph.
      </p>

      <p className="text-xs sm:text-sm text-gray-500 max-w-xl mb-8 leading-relaxed">
        Ask natural language questions. Natural queries are translated into structured graph
        specifications, deterministically executed against 124,700 entities and 311,242 relationships,
        and synthesized into strictly grounded answers.
      </p>

      {/* Question Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full text-left">
        {SUGGESTED_QUESTIONS.map((item, index) => {
          const Icon = item.icon;
          return (
            <button
              key={index}
              onClick={() => onSelectQuestion(item.question)}
              className="p-3.5 rounded-xl bg-surface border border-surface-border/80 hover:border-indigo-500/60 hover:bg-surface-elevated/90 transition-all duration-200 group text-left flex flex-col justify-between shadow-sm hover:shadow-md hover:shadow-indigo-950/30"
            >
              <div>
                <div className="flex items-center space-x-2 mb-1.5">
                  <div className="w-6 h-6 rounded-md bg-indigo-950/80 border border-indigo-800/50 flex items-center justify-center text-indigo-400 group-hover:text-indigo-300 group-hover:bg-indigo-900/90 transition-colors">
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-semibold text-gray-300 group-hover:text-white transition-colors">
                    {item.title}
                  </span>
                </div>
                <div className="text-xs text-gray-200 font-medium group-hover:text-indigo-200 transition-colors">
                  &ldquo;{item.question}&rdquo;
                </div>
              </div>
              <div className="text-[11px] text-gray-500 mt-2 line-clamp-1">
                {item.description}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
