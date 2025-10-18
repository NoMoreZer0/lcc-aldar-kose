import { ChatInput } from './components/ChatInput';
import { ChatSidebar } from './components/ChatSidebar';
import { MessageList } from './components/MessageList';
import { useChat } from './hooks/useChat';

const App = () => {
  const { chats, activeChatId, messages, isLoading, error, sendMessage, startNewChat, selectChat } =
    useChat();

  return (
    <div className="app-shell">
      <ChatSidebar
        chats={chats}
        activeChatId={activeChatId}
        isBusy={isLoading}
        onNewChat={() => {
          void startNewChat().catch(() => undefined);
        }}
        onSelectChat={(chatId) => {
          void selectChat(chatId).catch(() => undefined);
        }}
      />

      <main className="chat-panel">
        <header className="chat-panel__header">
          <div>
            <h1>Aldar Kose Assistant</h1>
            <p>Ask questions, brainstorm ideas, or explore knowledge about the Aldar Kose project.</p>
          </div>
          <button
            type="button"
            onClick={() => {
              void startNewChat().catch(() => undefined);
            }}
            disabled={isLoading}
          >
            Clear chat
          </button>
        </header>

        <MessageList messages={messages} isLoading={isLoading} />

        {error && (
          <div className="chat-panel__error" role="alert">
            {error}
          </div>
        )}

        <ChatInput onSend={sendMessage} isLoading={isLoading} />
      </main>
    </div>
  );
};

export default App;
