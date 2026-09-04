# L'Oracle Web UI

Modern, responsive web interface for the RAG Documentation Assistant.

## Features

- **Real-time Streaming**: WebSocket-based streaming for instant responses
- **Conversation Management**: Create new conversations, clear history
- **Source Citations**: View document sources with similarity scores
- **Responsive Design**: Works seamlessly on mobile and desktop
- **Modern UI**: Built with React, TypeScript, and Tailwind CSS

## Development

### Prerequisites

- Node.js 20+
- npm or yarn

### Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Configure API URL in `.env`:
```
VITE_API_URL=http://localhost:8080
```

4. Start development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

Built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Docker

### Development Mode

```bash
docker build -f Dockerfile.dev -t rag-web-ui:dev .
docker run -p 3000:3000 -v $(pwd):/app rag-web-ui:dev
```

### Production Mode

```bash
docker build -t rag-web-ui:prod .
docker run -p 80:80 rag-web-ui:prod
```

## Architecture

### Components

- **App.tsx**: Main application component with state management
- **MessageList**: Displays conversation messages with auto-scroll
- **MessageInput**: Text input with send button and keyboard shortcuts
- **SourceCard**: Displays document sources with similarity scores

### API Client

The `api/client.ts` module provides:
- REST API calls for non-streaming queries
- WebSocket support for streaming responses
- Error handling and retries
- Conversation history retrieval

### State Management

Uses React hooks for local state:
- `messages`: Array of conversation messages
- `conversationId`: Current conversation identifier
- `isLoading`: Loading state for API calls
- `error`: Error messages for display

## Configuration

### Environment Variables

- `VITE_API_URL`: Backend API URL (default: `http://localhost:8080`)

### Features

- **Streaming Toggle**: Switch between streaming and non-streaming modes
- **Auto-scroll**: Messages automatically scroll to bottom
- **Mobile Responsive**: Optimized layouts for all screen sizes
- **Error Handling**: User-friendly error messages with retry options

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

## License

See main project LICENSE file.
