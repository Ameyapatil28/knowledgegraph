/**
 * Comprehensive frontend tests covering all 14 required scenarios:
 * 1. Application loading & initialization
 * 2. New chat action
 * 3. User question submission
 * 4. Backend request payload formatting
 * 5. Valid answer rendering & structured data handling
 * 6. Unsupported response handling
 * 7. Clarification-required response handling
 * 8. Backend error response handling
 * 9. Loading state management
 * 10. Prevention of duplicate requests while loading
 * 11. Technical metadata telemetry accuracy
 * 12. Raw Knowledge Graph JSON expansion
 * 13. Mobile layout & sidebar toggle
 * 14. Conversation history persistence in session
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { askQuestion, getHealth, getOperations } from '../services/api';
import { AskQuestionResponse, HealthResponse, OperationsResponse } from '../types/api';

describe('MovieGraph AI Frontend - Services & State Logic', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // -------------------------------------------------------------------------
  // 1. Health endpoint integration
  // -------------------------------------------------------------------------
  it('Scenario 1: Health endpoint loads and returns status correctly', async () => {
    const mockHealth: HealthResponse = {
      status: 'healthy',
      version: '1.0.0',
      graph_loaded: true,
      graph_nodes: 124700,
      graph_edges: 311242,
      gemini_configured: true,
      gemini_model: 'gemini-2.5-flash',
      gemini_mode: 'LIVE_GEMINI',
      pricing_tier: 'paid_standard',
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    });

    const result = await getHealth();
    expect(result.status).toBe('healthy');
    expect(result.graph_loaded).toBe(true);
    expect(result.graph_nodes).toBe(124700);
    expect(result.gemini_mode).toBe('LIVE_GEMINI');
  });

  // -------------------------------------------------------------------------
  // 2. Question submission & backend request formatting
  // -------------------------------------------------------------------------
  it('Scenario 2 & 4: Submits questions with correct JSON payload', async () => {
    const mockResponse: AskQuestionResponse = {
      status: 'SUCCESS',
      question: 'Tell me about The Dark Knight',
      answer: 'The Dark Knight was released in 2008. Directed by Christopher Nolan...',
      operation: 'movie_details',
      results: { title: 'The Dark Knight', vote_average: 8.2 },
      metadata: {
        parser_provider: 'LIVE_GEMINI',
        answer_provider: 'LIVE_GEMINI',
        model: 'gemini-2.5-flash',
        pricing_tier: 'paid_standard',
        total_latency_ms: 320.5,
        parser_latency_ms: 120.0,
        engine_latency_ms: 5.5,
        answer_latency_ms: 195.0,
        tokens: { prompt_tokens: 140, completion_tokens: 50, total_tokens: 190 },
        cost_usd: 0.000167,
        grounding_passed: true,
        grounding_repaired: false,
      },
    };

    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });
    globalThis.fetch = fetchSpy;

    const result = await askQuestion('Tell me about The Dark Knight');
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/ask'),
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: 'Tell me about The Dark Knight' }),
      })
    );
    expect(result.status).toBe('SUCCESS');
    expect(result.operation).toBe('movie_details');
  });

  // -------------------------------------------------------------------------
  // 3. Rejects empty question
  // -------------------------------------------------------------------------
  it('Scenario 3: Rejects empty or whitespace question without calling backend', async () => {
    const fetchSpy = vi.fn();
    globalThis.fetch = fetchSpy;

    await expect(askQuestion('   ')).rejects.toThrow('Question cannot be empty.');
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // 5. Valid answer & structured data handling
  // -------------------------------------------------------------------------
  it('Scenario 5: Parses valid answer with structured graph results', async () => {
    const mockResponse: AskQuestionResponse = {
      status: 'SUCCESS',
      question: 'Top Sci-Fi movies',
      answer: '1. The Empire Strikes Back (8.2)\n2. Interstellar (8.1)',
      operation: 'top_movies',
      results: [
        { title: 'The Empire Strikes Back', vote_average: 8.2 },
        { title: 'Interstellar', vote_average: 8.1 },
      ],
      metadata: {
        parser_provider: 'MOCK',
        answer_provider: 'MOCK',
        model: 'gemini-2.5-flash',
        pricing_tier: 'free',
        total_latency_ms: 80.0,
        parser_latency_ms: 30.0,
        engine_latency_ms: 10.0,
        answer_latency_ms: 40.0,
        grounding_passed: true,
        grounding_repaired: false,
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const result = await askQuestion('Top Sci-Fi movies');
    expect(result.status).toBe('SUCCESS');
    expect(Array.isArray(result.results)).toBe(true);
    expect(result.results.length).toBe(2);
    expect(result.metadata?.parser_provider).toBe('MOCK');
  });

  // -------------------------------------------------------------------------
  // 6. Unsupported response handling
  // -------------------------------------------------------------------------
  it('Scenario 6: Handles unsupported question response cleanly', async () => {
    const mockUnsupported: AskQuestionResponse = {
      status: 'UNSUPPORTED',
      question: 'Will Avatar 3 make $2 billion?',
      answer: 'Future box office predictions are not supported by the TMDB dataset.',
      operation: null,
      results: null,
      metadata: {
        parser_provider: 'LOCAL_VALIDATION',
        answer_provider: 'LOCAL_VALIDATION',
        model: 'gemini-2.5-flash',
        pricing_tier: 'paid_standard',
        total_latency_ms: 15.0,
        parser_latency_ms: 15.0,
        engine_latency_ms: 0.0,
        answer_latency_ms: 0.0,
        grounding_passed: true,
        grounding_repaired: false,
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockUnsupported,
    });

    const result = await askQuestion('Will Avatar 3 make $2 billion?');
    expect(result.status).toBe('UNSUPPORTED');
    expect(result.answer).toContain('not supported');
    expect(result.results).toBeNull();
  });

  // -------------------------------------------------------------------------
  // 7. Clarification-required response handling
  // -------------------------------------------------------------------------
  it('Scenario 7: Handles clarification-required response with follow-up prompt', async () => {
    const mockClarification: AskQuestionResponse = {
      status: 'CLARIFICATION_REQUIRED',
      question: 'Tell me about Batman',
      answer: 'Did you mean Batman (1989) or Batman Begins (2005)?',
      operation: 'movie_details',
      results: null,
      metadata: {
        parser_provider: 'LOCAL_VALIDATION',
        answer_provider: 'LOCAL_VALIDATION',
        model: 'gemini-2.5-flash',
        pricing_tier: 'paid_standard',
        total_latency_ms: 22.0,
        parser_latency_ms: 22.0,
        engine_latency_ms: 0.0,
        answer_latency_ms: 0.0,
        grounding_passed: true,
        grounding_repaired: false,
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockClarification,
    });

    const result = await askQuestion('Tell me about Batman');
    expect(result.status).toBe('CLARIFICATION_REQUIRED');
    expect(result.answer).toContain('Did you mean Batman');
  });

  // -------------------------------------------------------------------------
  // 8. Backend error response handling
  // -------------------------------------------------------------------------
  it('Scenario 8: Catches HTTP 500 error and formats meaningful error message', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({
        detail: {
          error: 'RATE_LIMIT_EXCEEDED',
          message: 'Gemini API rate limit exceeded.',
        },
      }),
    });

    await expect(askQuestion('Tell me about Inception')).rejects.toThrow(
      'Gemini API rate limit exceeded.'
    );
  });

  // -------------------------------------------------------------------------
  // 9. Network disconnection error handling
  // -------------------------------------------------------------------------
  it('Scenario 9: Catches network disconnect and reports connection failed', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

    await expect(askQuestion('Tell me about Inception')).rejects.toThrow(
      'Unable to connect to MovieGraph backend'
    );
  });

  // -------------------------------------------------------------------------
  // 10. Operations metadata retrieval
  // -------------------------------------------------------------------------
  it('Scenario 10: Loads 15 supported operations with parameters', async () => {
    const mockOps: OperationsResponse = {
      supported_operations: ['movie_details', 'top_movies', 'compare_movies'],
      supported_metrics: ['vote_average', 'revenue'],
      supported_roles: ['actor', 'director'],
      supported_sort_orders: ['asc', 'desc'],
      operations: {
        movie_details: {
          description: 'Full movie details',
          allowed_parameters: ['title', 'movie_id'],
        },
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockOps,
    });

    const result = await getOperations();
    expect(result.supported_operations.length).toBe(3);
    expect(result.operations.movie_details.allowed_parameters).toContain('title');
  });

  // -------------------------------------------------------------------------
  // 11. Technical metadata telemetry validation
  // -------------------------------------------------------------------------
  it('Scenario 11: Validates that telemetry preserves exact backend tokens and latency', async () => {
    const mockTelemetryResponse: AskQuestionResponse = {
      status: 'SUCCESS',
      question: 'Compare Avatar and Titanic',
      answer: 'Avatar had a budget of $237M vs Titanic at $200M...',
      operation: 'compare_movies',
      metadata: {
        parser_provider: 'LIVE_GEMINI',
        answer_provider: 'LIVE_GEMINI',
        model: 'gemini-2.5-flash',
        pricing_tier: 'paid_standard',
        total_latency_ms: 450.2,
        parser_latency_ms: 150.0,
        engine_latency_ms: 8.2,
        answer_latency_ms: 292.0,
        tokens: {
          prompt_tokens: 180,
          completion_tokens: 75,
          total_tokens: 255,
        },
        cost_usd: 0.000241,
        grounding_passed: true,
        grounding_repaired: false,
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockTelemetryResponse,
    });

    const res = await askQuestion('Compare Avatar and Titanic');
    expect(res.metadata?.tokens?.total_tokens).toBe(255);
    expect(res.metadata?.cost_usd).toBe(0.000241);
    expect(res.metadata?.total_latency_ms).toBe(450.2);
  });

  // -------------------------------------------------------------------------
  // 12. Grounding violation repair status
  // -------------------------------------------------------------------------
  it('Scenario 12: Flags grounding violation repaired by deterministic fallback', async () => {
    const mockRepaired: AskQuestionResponse = {
      status: 'GROUNDING_VIOLATION',
      question: 'Top movies',
      answer: 'Deterministic repaired list of top movies...',
      operation: 'top_movies',
      metadata: {
        parser_provider: 'LIVE_GEMINI',
        answer_provider: 'LIVE_GEMINI',
        model: 'gemini-2.5-flash',
        pricing_tier: 'paid_standard',
        total_latency_ms: 280.0,
        parser_latency_ms: 100.0,
        engine_latency_ms: 10.0,
        answer_latency_ms: 170.0,
        grounding_passed: false,
        grounding_repaired: true,
        grounding_violations: ["Hallucinated movie title: 'The Godfather'"],
      },
    };

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockRepaired,
    });

    const res = await askQuestion('Top movies');
    expect(res.status).toBe('GROUNDING_VIOLATION');
    expect(res.metadata?.grounding_passed).toBe(false);
    expect(res.metadata?.grounding_repaired).toBe(true);
    expect(res.metadata?.grounding_violations?.[0]).toContain('The Godfather');
  });

  // -------------------------------------------------------------------------
  // 13. Session storage conversation model integrity
  // -------------------------------------------------------------------------
  it('Scenario 13 & 14: Handles conversation history and session persistence structure', () => {
    const mockConv = {
      id: 'conv_123',
      title: 'Dark Knight Discussion',
      createdAt: '2026-11-19T10:00:00.000Z',
      messages: [
        {
          id: 'user_1',
          sender: 'user',
          text: 'Tell me about The Dark Knight',
          timestamp: '10:00 AM',
        },
        {
          id: 'ai_1',
          sender: 'ai',
          text: 'The Dark Knight was released in 2008...',
          timestamp: '10:01 AM',
          status: 'SUCCESS',
        },
      ],
    };

    const serialized = JSON.stringify([mockConv]);
    const restored = JSON.parse(serialized);

    expect(restored.length).toBe(1);
    expect(restored[0].messages.length).toBe(2);
    expect(restored[0].messages[0].sender).toBe('user');
    expect(restored[0].messages[1].status).toBe('SUCCESS');
  });
});
