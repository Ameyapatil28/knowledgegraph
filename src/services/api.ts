/**
 * Centralized API Service for TMDB Knowledge Graph Backend.
 */

import {
  AskQuestionRequest,
  AskQuestionResponse,
  HealthResponse,
  OperationsResponse,
} from '../types/api';

// Configurable base URL with sensible fallback: default to empty string to leverage Vite's dev proxy
const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string) ?? '';

class ApiServiceError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number = 500, code?: string) {
    super(message);
    this.name = 'ApiServiceError';
    this.status = status;
    this.code = code;
  }
}

/**
 * Handle API responses and standardize error reporting
 */
async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = 'An unexpected error occurred.';
    let errorCode = 'SERVER_ERROR';

    try {
      const errorJson = await res.json();
      if (errorJson.detail) {
        if (typeof errorJson.detail === 'object') {
          errorDetail = errorJson.detail.message || errorDetail;
          errorCode = errorJson.detail.error || errorCode;
        } else {
          errorDetail = String(errorJson.detail);
        }
      } else if (errorJson.message) {
        errorDetail = errorJson.message;
        errorCode = errorJson.code || errorCode;
      }
    } catch {
      errorDetail = `Request failed with HTTP ${res.status}: ${res.statusText}`;
    }

    throw new ApiServiceError(errorDetail, res.status, errorCode);
  }

  return (await res.json()) as T;
}

/**
 * Send natural language question to the Knowledge Graph QA pipeline.
 */
export async function askQuestion(question: string): Promise<AskQuestionResponse> {
  const cleanQuestion = question.trim();
  if (!cleanQuestion) {
    throw new ApiServiceError('Question cannot be empty.', 400, 'EMPTY_QUESTION');
  }

  const payload: AskQuestionRequest = { question: cleanQuestion };

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    return await handleResponse<AskQuestionResponse>(res);
  } catch (err: any) {
    if (err instanceof ApiServiceError) {
      throw err;
    }
    console.error('[MovieGraph API Connection Error]:', err);
    // Network or server unreachable error
    throw new ApiServiceError(
      'Unable to connect to MovieGraph backend. Please ensure the server is running on http://127.0.0.1:8000.',
      0,
      'CONNECTION_FAILED'
    );
  }
}

/**
 * Check backend readiness, graph loading status, and Gemini provider mode.
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });

    return await handleResponse<HealthResponse>(res);
  } catch (err: any) {
    if (err instanceof ApiServiceError) {
      throw err;
    }
    console.error('[MovieGraph Health Check Error]:', err);
    throw new ApiServiceError(
      'Backend is currently offline or unreachable.',
      0,
      'HEALTH_CHECK_FAILED'
    );
  }
}

/**
 * Retrieve all 15 supported Knowledge Graph operations and metrics.
 */
export async function getOperations(): Promise<OperationsResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/operations`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });

    return await handleResponse<OperationsResponse>(res);
  } catch (err: any) {
    if (err instanceof ApiServiceError) {
      throw err;
    }
    throw new ApiServiceError(
      'Failed to load supported graph operations.',
      0,
      'OPERATIONS_FAILED'
    );
  }
}

export { ApiServiceError };
