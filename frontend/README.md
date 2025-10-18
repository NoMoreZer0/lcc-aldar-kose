# Aldar Kose Frontend

A React + Vite single-page application that provides a ChatGPT-inspired interface for conversing with the Aldar Kose assistant.

## Tech stack

- [React 18](https://react.dev/) with TypeScript for the UI
- [Vite](https://vitejs.dev/) for development server and build tooling
- Light CSS handcrafted styling (no component framework)

## Getting started

```bash
cd frontend
npm install
npm run dev
```

The development server proxies `/api` requests to `http://127.0.0.1:8000`. Adjust `vite.config.ts` or set `VITE_API_BASE` if your backend runs elsewhere.

> Mock assistant replies are enabled by default. Create a `.env` file with `VITE_USE_MOCK=false` once the backend is ready.
> While the backend is offline the assistant alternates between text answers and visual storyboards (6–10 curated placeholder images from Picsum).

### Environment variables

- `VITE_API_BASE` *(optional)* – override the default `/api/v1` base path if the API uses a different prefix.
- `VITE_USE_MOCK` *(default: true)* – disable to rely entirely on backend responses; docker-compose keeps it `true` until the API can generate assistant messages.

The UI boots by loading existing chats from the backend, remembers the active conversation in `localStorage`, and keeps the sidebar in sync whenever new messages are saved.

## Project structure

```
frontend/
├─ index.html            # Vite entry point
├─ src/
│  ├─ App.tsx            # Top-level layout and state wiring
│  ├─ main.tsx           # Application bootstrap
│  ├─ styles.css         # Global and component styling
│  ├─ mockResponses.ts   # Mocked assistant replies while API is offline
│  ├─ components/        # Reusable UI atoms
│  ├─ hooks/             # Custom React hooks (chat state machine)
│  └─ vite-env.d.ts      # Vite TypeScript environment definitions
└─ vite.config.ts        # Tooling configuration & API proxy
```

## Next steps

- Show previews/snippets for each chat in the sidebar.
- Introduce error boundaries and reconnect logic for streaming responses.
- Add optimistic updates for assistant responses when the real backend goes live.
