import { CinematicCarousel } from './CinematicCarousel';
import type { ChatMessage, MessageAttachment } from '../types';

interface MessageBubbleProps {
  message: ChatMessage;
}

const roleLabel: Record<ChatMessage['role'], string> = {
  user: 'You',
  assistant: 'Assistant'
};

const renderAttachment = (attachment: MessageAttachment, index: number) => {
  return attachment.type === 'image' ? (
    <figure key={`${attachment.url}-${index}`} className="message-bubble__attachment">
      <img src={attachment.url} alt={attachment.alt} loading="lazy" />
      <figcaption>{attachment.alt}</figcaption>
    </figure>
  ) : null;
};

export const MessageBubble = ({ message }: MessageBubbleProps) => {
  const bubbleClassName = `message-bubble message-bubble--${message.role}`;
  const createdAt = new Date(message.createdAt);
  const formatUtcPlusFive = (date: Date) => {
    const utcPlusFive = new Date(date.getTime() + 5 * 60 * 60 * 1000);
    return `${utcPlusFive.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false })} UTC+5`;
  };
  const displayTime = Number.isNaN(createdAt.getTime())
    ? message.createdAt
    : formatUtcPlusFive(createdAt);

  return (
    <article className={bubbleClassName} aria-label={`${roleLabel[message.role]} message`}>
      <header className="message-bubble__meta">
        <span className="message-bubble__author">{roleLabel[message.role]}</span>
        <time dateTime={message.createdAt}>
          {displayTime}
        </time>
      </header>
      <div className="message-bubble__content">
        {message.content.split('\n').map((line, index) => (
          <p key={index}>{line}</p>
        ))}
      </div>
      {message.attachments && message.attachments.length > 0 ? (
        message.role === 'assistant' ? (
          <CinematicCarousel frames={message.attachments} title="Storyboard sequence" />
        ) : (
          <div className="message-bubble__attachments" aria-label="Attachments">
            {message.attachments.map(renderAttachment)}
          </div>
        )
      ) : null}
    </article>
  );
};
