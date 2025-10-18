import type { ChatMessage, MessageAttachment } from './types';

interface MockRule {
  keywords: string[];
  response: string;
}

const mockRules: MockRule[] = [
  {
    keywords: ['hello', 'hi', 'privet'],
    response: `Hello! I'm the Aldar Kose assistant. Ask me about the project goals, data, or how the system works.`
  },
  {
    keywords: ['data', 'dataset', 'knowledge'],
    response: `The current dataset blends folklore-inspired narratives with structured learning content. We surface connections, summaries, and next-step suggestions to make exploration easier.`
  },
  {
    keywords: ['how', 'work', 'architecture', 'stack'],
    response: `The assistant pairs a React/Vite frontend with a Python backend that orchestrates large language models. The frontend captures prompts, streams answers, and keeps interaction lightweight for experimentation.`
  }
];

const defaultSuggestions = [
  'Summarise the latest findings about Aldar Kose.',
  'Draft an outline for a presentation.',
  'Ask for resources to learn more about the culture behind the tale.'
];

const randomInt = (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min;

const slugify = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 24);

const createImageAttachments = (prompt: string): MessageAttachment[] => {
  const count = randomInt(6, 10);
  const baseSeed = slugify(prompt) || 'aldar-kose';

  return Array.from({ length: count }, (_, index) => {
    const seed = `${baseSeed}-${index}-${randomInt(0, 9999)}`;

    return {
      type: 'image' as const,
      url: `https://picsum.photos/seed/${encodeURIComponent(seed)}/512/384`,
      alt: `Concept visual ${index + 1} inspired by "${prompt}"`
    };
  });
};

const buildPhotoStory = (prompt: string): ChatMessage => {
  const attachments = createImageAttachments(prompt);
  const storyline = [
    `Visual storyboard inspired by "${prompt}".`,
    'Each frame captures mood, palette, and narrative hints to help shape the tale.'
  ].join('\n');

  return {
    id: `assistant-mock-${Math.random().toString(36).slice(2)}`,
    role: 'assistant',
    content: storyline,
    createdAt: new Date().toISOString(),
    attachments
  };
};

const buildTextResponse = (prompt: string): ChatMessage => {
  const normalizedPrompt = prompt.toLowerCase();
  const match = mockRules.find((rule) => rule.keywords.some((keyword) => normalizedPrompt.includes(keyword)));

  const content = match
    ? match.response
    : `I'm still in prototype mode, but here's a thought starter based on your prompt:\n• ${defaultSuggestions.join(
        '\n• '
      )}`;

  return {
    id: `assistant-mock-${Math.random().toString(36).slice(2)}`,
    role: 'assistant',
    content,
    createdAt: new Date().toISOString()
  };
};

export const buildMockAssistantMessage = (prompt: string): ChatMessage => {
  const shouldSendPhotos = Math.random() < 0.5;

  return shouldSendPhotos ? buildPhotoStory(prompt) : buildTextResponse(prompt);
};
