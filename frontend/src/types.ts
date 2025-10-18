export type ChatRole = 'user' | 'assistant';

export interface MessageAttachment {
  type: 'image';
  url: string;
  alt: string;
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  attachments?: MessageAttachment[];
}

export interface ChatSummary {
  id: string;
  title: string | null;
  createdAt: string;
  updatedAt: string;
  lastMessageAt: string | null;
  messageCount: number;
}
