import { FormEvent, KeyboardEvent, useState } from 'react';

interface ChatInputProps {
  onSend: (prompt: string) => void | Promise<void>;
  isLoading: boolean;
}

export const ChatInput = ({ onSend, isLoading }: ChatInputProps) => {
  const [value, setValue] = useState('');

  const submitMessage = async () => {
    const trimmed = value.trim();

    if (!trimmed) {
      return;
    }

    try {
      await onSend(trimmed);
      setValue('');
    } catch {
      // preserve input so the user can retry
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submitMessage();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void submitMessage();
    }
  };

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Send a message..."
        disabled={isLoading}
        rows={1}
        aria-label="Message"
      />
      <button type="submit" disabled={isLoading || value.trim().length === 0}>
        Send
      </button>
    </form>
  );
};
