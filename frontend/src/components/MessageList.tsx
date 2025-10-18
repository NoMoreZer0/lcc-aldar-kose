import { MessageBubble } from './MessageBubble';
import type { ChatMessage } from '../types';

interface MessageListProps {
  messages: ChatMessage[];
  isLoading: boolean;
}

export const MessageList = ({ messages, isLoading }: MessageListProps) => (
  <section className="message-list" aria-live="polite" aria-busy={isLoading}>
    {messages.length === 0 && !isLoading ? (
      <div className="message-list__empty">
        <h2>Ask anything about Aldar Kose</h2>
        <p>Describe a topic or question and the assistant will help you explore it.</p>
      </div>
    ) : (
      messages.map((message) => <MessageBubble key={message.id} message={message} />)
    )}
    {isLoading && (
      <div className="message-bubble message-bubble--assistant message-bubble--loading">
        <div className="loading-indicator">
          <span />
          <span />
          <span />
        </div>
      </div>
    )}
  </section>
);
