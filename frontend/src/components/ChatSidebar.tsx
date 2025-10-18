import type { ChatSummary } from '../types';

interface ChatSidebarProps {
  chats: ChatSummary[];
  activeChatId: string | null;
  isBusy: boolean;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
}

const formatTimestamp = (iso: string | null) => {
  if (!iso) {
    return 'No messages yet';
  }

  try {
    return new Date(iso).toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch {
    return iso;
  }
};

const buildTitle = (chat: ChatSummary, index: number, total: number) => {
  const candidate = chat.title?.trim();
  if (candidate && candidate.length > 0) {
    return candidate;
  }

  return `Conversation ${total - index}`;
};

export const ChatSidebar = ({
  chats,
  activeChatId,
  isBusy,
  onNewChat,
  onSelectChat
}: ChatSidebarProps) => (
  <aside className="app-sidebar">
    <div className="app-sidebar__brand">
      <span className="app-sidebar__logo" aria-hidden="true">
        AK
      </span>
      <div>
        <strong>Aldar Kose</strong>
        <span>Conversational AI</span>
      </div>
    </div>

    <button
      type="button"
      className="app-sidebar__action"
      onClick={() => {
        onNewChat();
      }}
      disabled={isBusy}
    >
      New Chat
    </button>

    {chats.length === 0 ? (
      <p className="app-sidebar__empty">Start a conversation to see it appear here.</p>
    ) : (
      <nav className="app-sidebar__list" aria-label="Chat history">
        {chats.map((chat, index) => {
          const isActive = chat.id === activeChatId;
          const title = buildTitle(chat, index, chats.length);
          const messageLabel =
            chat.messageCount === 0
              ? 'No messages yet'
              : `${chat.messageCount} message${chat.messageCount === 1 ? '' : 's'}`;
          const subtitle = `${messageLabel} • ${formatTimestamp(chat.lastMessageAt ?? chat.updatedAt)}`;

          return (
            <button
              type="button"
              key={chat.id}
              className={`app-sidebar__chat${isActive ? ' app-sidebar__chat--active' : ''}`}
              aria-current={isActive ? 'page' : undefined}
              onClick={() => {
                if (!isActive) {
                  onSelectChat(chat.id);
                }
              }}
              disabled={isBusy}
            >
              <span className="app-sidebar__chat-title">{title}</span>
              <span className="app-sidebar__chat-subtitle">{subtitle}</span>
            </button>
          );
        })}
      </nav>
    )}
  </aside>
);
