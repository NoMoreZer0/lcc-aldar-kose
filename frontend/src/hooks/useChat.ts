import { useCallback, useEffect, useRef, useState } from 'react';
import { buildMockAssistantMessage } from '../mockResponses';
import type { ChatMessage, ChatSummary } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api/v1').replace(/\/+$/, '');
const shouldUseMock = import.meta.env.VITE_USE_MOCK !== 'false';
const ACTIVE_CHAT_STORAGE_KEY = 'ak-active-chat';

const buildUrl = (path: string) => `${API_BASE}${path}`;

const readStoredChatId = () => {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    return window.localStorage.getItem(ACTIVE_CHAT_STORAGE_KEY);
  } catch {
    return null;
  }
};

const storeActiveChatId = (chatId: string | null) => {
  if (typeof window === 'undefined') {
    return;
  }

  try {
    if (chatId) {
      window.localStorage.setItem(ACTIVE_CHAT_STORAGE_KEY, chatId);
    } else {
      window.localStorage.removeItem(ACTIVE_CHAT_STORAGE_KEY);
    }
  } catch {
    // ignore storage errors
  }
};

type AttachmentPayload = {
  type: 'image';
  url: string;
  alt: string;
};

type MessagePayload = {
  role: ChatMessage['role'];
  content: string;
  attachments?: AttachmentPayload[];
};

type ApiAttachment = AttachmentPayload & {
  id: string;
  created_at: string;
};

type ApiMessage = {
  id: string;
  role: ChatMessage['role'];
  content: string;
  sequence: number;
  created_at: string;
  attachments: ApiAttachment[];
};

type ApiChat = {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  messages: ApiMessage[];
};

type ApiChatSummary = {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
  last_message_at: string | null;
};

interface UseChatResult {
  chats: ChatSummary[];
  activeChatId: string | null;
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  sendMessage: (prompt: string) => Promise<void>;
  startNewChat: () => Promise<string>;
  selectChat: (chatId: string) => Promise<void>;
}

const parseJson = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    let detail: string | undefined;

    try {
      const data = await response.json();
      detail = typeof data === 'string' ? data : (data as { detail?: string }).detail;
    } catch {
      try {
        detail = await response.text();
      } catch {
        detail = undefined;
      }
    }

    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
};

const adaptMessage = (message: ApiMessage): ChatMessage => {
  const attachments =
    message.attachments && message.attachments.length > 0
      ? message.attachments.map(({ type, url, alt }) => ({ type, url, alt }))
      : undefined;

  return {
    id: message.id,
    role: message.role,
    content: message.content,
    createdAt: message.created_at,
    attachments
  };
};

const adaptSummary = (summary: ApiChatSummary): ChatSummary => ({
  id: summary.id,
  title: summary.title,
  createdAt: summary.created_at,
  updatedAt: summary.updated_at,
  lastMessageAt: summary.last_message_at,
  messageCount: summary.message_count
});

const adaptChatToSummary = (chat: ApiChat): ChatSummary => {
  const messageCount = chat.messages?.length ?? 0;
  const lastMessageAt = messageCount > 0 ? chat.messages[messageCount - 1].created_at : null;

  return {
    id: chat.id,
    title: chat.title,
    createdAt: chat.created_at,
    updatedAt: chat.updated_at,
    lastMessageAt,
    messageCount
  };
};

const sortChats = (items: ChatSummary[]) =>
  items.slice().sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());

const mapAttachmentsToPayload = (attachments?: ChatMessage['attachments']) =>
  attachments && attachments.length > 0
    ? attachments.map(({ type, url, alt }) => ({ type, url, alt }))
    : undefined;

export const useChat = (): UseChatResult => {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const chatCreationPromiseRef = useRef<Promise<string> | null>(null);

  const loadChatSummaries = useCallback(async (): Promise<ChatSummary[]> => {
    const response = await fetch(buildUrl('/chats'));
    const data = await parseJson<ApiChatSummary[]>(response);
    const summaries = sortChats(data.map(adaptSummary));
    setChats(summaries);
    return summaries;
  }, []);

  const listMessages = useCallback(async (targetChatId: string) => {
    const response = await fetch(buildUrl(`/chats/${targetChatId}/messages`));
    const data = await parseJson<ApiMessage[]>(response);
    setMessages(data.map(adaptMessage));
  }, []);

  const createChatRequest = useCallback(async (): Promise<ApiChat> => {
    const response = await fetch(buildUrl('/chats'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({})
    });

    return parseJson<ApiChat>(response);
  }, []);

  const persistMessage = useCallback(async (targetChatId: string, payload: MessagePayload) => {
    const response = await fetch(buildUrl(`/chats/${targetChatId}/messages`), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    return parseJson<ApiMessage>(response);
  }, []);

  const startNewChat = useCallback(async (): Promise<string> => {
    if (chatCreationPromiseRef.current) {
      return chatCreationPromiseRef.current;
    }

    setError(null);
    setMessages([]);
    setActiveChatId(null);
    setIsLoading(true);
    storeActiveChatId(null);

    const promise = (async () => {
      try {
        const chat = await createChatRequest();
        const summary = adaptChatToSummary(chat);
        setChats((prev) => sortChats([summary, ...prev.filter((item) => item.id !== summary.id)]));
        setActiveChatId(chat.id);
        storeActiveChatId(chat.id);

        const initialMessages = chat.messages?.map(adaptMessage) ?? [];
        setMessages(initialMessages);

        return chat.id;
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Unable to create a new chat. Please try again.';
        setError(message);
        throw err instanceof Error ? err : new Error(message);
      } finally {
        setIsLoading(false);
        chatCreationPromiseRef.current = null;
      }
    })();

    chatCreationPromiseRef.current = promise;

    return promise;
  }, [createChatRequest]);

  const ensureChatId = useCallback(async () => {
    if (activeChatId) {
      return activeChatId;
    }

    return startNewChat();
  }, [activeChatId, startNewChat]);

  const selectChat = useCallback(
    async (chatId: string) => {
      if (chatId === activeChatId) {
        return;
      }

      setError(null);
      setIsLoading(true);
      setMessages([]);

      try {
        await listMessages(chatId);
        setActiveChatId(chatId);
        storeActiveChatId(chatId);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Unable to load that conversation. Please try again.';
        setError(message);
        throw err instanceof Error ? err : new Error(message);
      } finally {
        setIsLoading(false);
      }
    },
    [activeChatId, listMessages]
  );

  const sendMessage = useCallback(
    async (prompt: string) => {
      const trimmedPrompt = prompt.trim();

      if (!trimmedPrompt) {
        return;
      }

      setIsLoading(true);
      setError(null);
      let userMessagePersisted = false;

      try {
        const targetChatId = await ensureChatId();

        const savedUser = await persistMessage(targetChatId, {
          role: 'user',
          content: trimmedPrompt
        });

        const userMessage = adaptMessage(savedUser);
        setMessages((prev) => [...prev, userMessage]);
        userMessagePersisted = true;

        if (shouldUseMock) {
          const mockAssistant = buildMockAssistantMessage(trimmedPrompt);
          const assistantPayload: MessagePayload = {
            role: 'assistant',
            content: mockAssistant.content
          };

          const attachmentPayloads = mapAttachmentsToPayload(mockAssistant.attachments);
          if (attachmentPayloads) {
            assistantPayload.attachments = attachmentPayloads;
          }

          const savedAssistant = await persistMessage(targetChatId, assistantPayload);
          const assistantMessage = adaptMessage(savedAssistant);
          setMessages((prev) => [...prev, assistantMessage]);
        }

        await loadChatSummaries();
        await listMessages(targetChatId);
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : 'Unexpected error while sending the message. Please try again.';
        setError(message);
        if (!userMessagePersisted) {
          throw err instanceof Error ? err : new Error(message);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [ensureChatId, listMessages, loadChatSummaries, persistMessage]
  );

  useEffect(() => {
    const initialise = async () => {
      setIsInitializing(true);
      setError(null);

      try {
        const chatList = await loadChatSummaries();

        if (chatList.length === 0) {
          setMessages([]);
          setActiveChatId(null);
          storeActiveChatId(null);
          return;
        }

        const storedId = readStoredChatId();
        const hasStoredChat = storedId && chatList.some((chat) => chat.id === storedId);

        const targetChatId = hasStoredChat ? (storedId as string) : chatList[0].id;
        setActiveChatId(targetChatId);
        storeActiveChatId(targetChatId);

        await listMessages(targetChatId);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Unable to load conversations. Please try again.';
        setError(message);
      } finally {
        setIsInitializing(false);
      }
    };

    void initialise();
  }, [listMessages, loadChatSummaries]);

  return {
    chats,
    activeChatId,
    messages,
    isLoading: isLoading || isInitializing,
    error,
    sendMessage,
    startNewChat,
    selectChat
  };
};
