/**
 * TypeScript interfaces matching TMDB Knowledge Graph FastAPI Pydantic schemas.
 */

export type AskStatus =
  | 'SUCCESS'
  | 'NO_RESULTS'
  | 'CLARIFICATION_REQUIRED'
  | 'UNSUPPORTED'
  | 'GENERATION_ERROR'
  | 'GROUNDING_VIOLATION'
  | 'ERROR';

export type ProviderMode = 'LIVE_GEMINI' | 'MOCK' | 'LOCAL_VALIDATION';

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface AnswerMetadata {
  parser_provider: ProviderMode;
  answer_provider: ProviderMode;
  model: string;
  pricing_tier: string;
  total_latency_ms: number;
  parser_latency_ms: number;
  engine_latency_ms: number;
  answer_latency_ms: number;
  tokens?: TokenUsage | null;
  cost_usd?: number | null;
  grounding_passed: boolean;
  grounding_repaired: boolean;
  grounding_violations?: string[];
}

export interface AskQuestionRequest {
  question: string;
}

export interface AskQuestionResponse {
  status: AskStatus;
  question: string;
  answer: string;
  operation?: string | null;
  results?: any | null;
  metadata?: AnswerMetadata | null;
  error?: {
    code: string;
    message: string;
  } | null;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded';
  version: string;
  graph_loaded: boolean;
  graph_nodes?: number | null;
  graph_edges?: number | null;
  gemini_configured: boolean;
  gemini_model: string;
  gemini_mode: ProviderMode;
  pricing_tier: string;
}

export interface OperationInfo {
  description: string;
  allowed_parameters: string[];
}

export interface OperationsResponse {
  supported_operations: string[];
  supported_metrics: string[];
  supported_roles: string[];
  supported_sort_orders: string[];
  operations: Record<string, OperationInfo>;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text: string;
  timestamp: string;
  status?: AskStatus;
  operation?: string | null;
  results?: any | null;
  metadata?: AnswerMetadata | null;
  error?: string | null;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  messages: ChatMessage[];
}

export interface GraphBuildStats {
  totalNodes: number;
  totalRelationships: number;
  movies: number;
  people: number;
  genres: number;
  productionCompanies: number;
  keywords: number;
  countries: number;
  languages: number;
  verifiedBuildDate: string;
}
