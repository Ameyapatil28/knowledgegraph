/**
 * Custom React Hook for Managing MovieGraph AI Chat Sessions.
 */

import { useState, useCallback, useEffect } from 'react';
import { ChatMessage, Conversation, AskQuestionResponse } from '../types/api';
import { askQuestion, ApiServiceError } from '../services/api';

const SESSION_STORAGE_KEY = 'moviegraph_ai_conversations';
const ACTIVE_CONV_KEY = 'moviegraph_ai_active_id';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_STORAGE_KEY);
      if (saved) {
        return JSON.parse(saved);
      }
    } catch {
      // Fallback on storage read error
    }
    return [];
  });

  const [activeId, setActiveId] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem(ACTIVE_CONV_KEY);
    } catch {
      return null;
    }
  });

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [lastFailedQuestion, setLastFailedQuestion] = useState<string | null>(null);

  // Sync to session storage
  useEffect(() => {
    try {
      sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(conversations));
      if (activeId) {
        sessionStorage.setItem(ACTIVE_CONV_KEY, activeId);
      } else {
        sessionStorage.removeItem(ACTIVE_CONV_KEY);
      }
    } catch {
      // Ignore quota errors
    }
  }, [conversations, activeId]);

  // Active conversation object
  const activeConversation = conversations.find((c) => c.id === activeId) || null;
  const messages: ChatMessage[] = activeConversation ? activeConversation.messages : [];

  /**
   * Start a fresh new chat session
   */
  const startNewChat = useCallback(() => {
    const newId = `conv_${Date.now()}`;
    const newConv: Conversation = {
      id: newId,
      title: 'New Conversation',
      createdAt: new Date().toISOString(),
      messages: [],
    };
    setConversations((prev) => [newConv, ...prev]);
    setActiveId(newId);
    setLastFailedQuestion(null);
  }, []);

  /**
   * Switch to an existing conversation
   */
  const selectConversation = useCallback((id: string) => {
    setActiveId(id);
    setLastFailedQuestion(null);
  }, []);

  /**
   * Send a question to the backend and append responses
   */
  const sendMessage = useCallback(
    async (questionText: string) => {
      const clean = questionText.trim();
      if (!clean || isLoading) {
        return;
      }

      setLastFailedQuestion(null);
      let targetConvId = activeId;

      // Create new conversation if none exists
      if (!targetConvId || !conversations.some((c) => c.id === targetConvId)) {
        targetConvId = `conv_${Date.now()}`;
        const newConv: Conversation = {
          id: targetConvId,
          title: clean.slice(0, 35) + (clean.length > 35 ? '...' : ''),
          createdAt: new Date().toISOString(),
          messages: [],
        };
        setConversations((prev) => [newConv, ...prev]);
        setActiveId(targetConvId);
      }

      const userMsg: ChatMessage = {
        id: `user_${Date.now()}`,
        sender: 'user',
        text: clean,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      // Append user message immediately
      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === targetConvId) {
            const isFirst = c.messages.length === 0;
            return {
              ...c,
              title: isFirst ? clean.slice(0, 35) + (clean.length > 35 ? '...' : '') : c.title,
              messages: [...c.messages, userMsg],
            };
          }
          return c;
        })
      );

      setIsLoading(true);

      try {
        const resp: AskQuestionResponse = await askQuestion(clean);

        const aiMsg: ChatMessage = {
          id: `ai_${Date.now()}`,
          sender: 'ai',
          text: resp.answer,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: resp.status,
          operation: resp.operation,
          results: resp.results,
          metadata: resp.metadata,
        };

        setConversations((prev) =>
          prev.map((c) => {
            if (c.id === targetConvId) {
              return {
                ...c,
                messages: [...c.messages, aiMsg],
              };
            }
            return c;
          })
        );
      } catch (err: any) {
        const errorText =
          err instanceof ApiServiceError
            ? err.message
            : 'An unexpected connection error occurred while generating the answer.';

        setLastFailedQuestion(clean);

        const errorMsg: ChatMessage = {
          id: `ai_err_${Date.now()}`,
          sender: 'ai',
          text: `⚠️ **Error**: ${errorText}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          status: 'ERROR',
          error: errorText,
        };

        setConversations((prev) =>
          prev.map((c) => {
            if (c.id === targetConvId) {
              return {
                ...c,
                messages: [...c.messages, errorMsg],
              };
            }
            return c;
          })
        );
      } finally {
        setIsLoading(false);
      }
    },
    [activeId, conversations, isLoading]
  );

  /**
   * Retry the last question that failed
   */
  const retryLastQuestion = useCallback(() => {
    if (lastFailedQuestion) {
      sendMessage(lastFailedQuestion);
    }
  }, [lastFailedQuestion, sendMessage]);

  /**
   * Clear all messages in current conversation
   */
  const clearCurrentConversation = useCallback(() => {
    if (!activeId) return;
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id === activeId) {
          return { ...c, messages: [] };
        }
        return c;
      })
    );
  }, [activeId]);

  /**
   * Delete an entire conversation
   */
  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) {
        setActiveId(null);
      }
    },
    [activeId]
  );

  return {
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
  };
}
