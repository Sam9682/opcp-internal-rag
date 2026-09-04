import { useState, useCallback } from 'react';
import MessageList from './components/MessageList';
import MessageInput from './components/MessageInput';
import { Message } from './types';
import { apiClient } from './api/client';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useStreaming, setUseStreaming] = useState(true);

  const handleSendMessage = useCallback(async (content: string) => {
    // Add user message
    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      if (useStreaming) {
        // Streaming mode
        let streamedContent = '';
        const assistantMessage: Message = {
          role: 'assistant',
          content: '',
          timestamp: new Date(),
        };

        // Add placeholder for assistant message
        setMessages((prev) => [...prev, assistantMessage]);

        await apiClient.queryStream(
          {
            query: content,
            conversation_id: conversationId,
            top_k: 5,
          },
          // onChunk
          (chunk: string) => {
            streamedContent += chunk;
            setMessages((prev) => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = {
                ...assistantMessage,
                content: streamedContent,
              };
              return newMessages;
            });
          },
          // onComplete
          (response) => {
            setMessages((prev) => {
              const newMessages = [...prev];
              newMessages[newMessages.length - 1] = {
                role: 'assistant',
                content: response.answer,
                timestamp: new Date(),
                sources: response.sources,
              };
              return newMessages;
            });
            setConversationId(response.conversation_id);
            setIsLoading(false);
          },
          // onError
          (err) => {
            setError(err.message);
            setIsLoading(false);
            // Remove the placeholder message
            setMessages((prev) => prev.slice(0, -1));
          }
        );
      } else {
        // Non-streaming mode
        const response = await apiClient.query({
          query: content,
          conversation_id: conversationId,
          top_k: 5,
        });

        const assistantMessage: Message = {
          role: 'assistant',
          content: response.answer,
          timestamp: new Date(),
          sources: response.sources,
        };

        setMessages((prev) => [...prev, assistantMessage]);
        setConversationId(response.conversation_id);
        setIsLoading(false);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      setIsLoading(false);
    }
  }, [conversationId, useStreaming]);

  const handleNewConversation = () => {
    setMessages([]);
    setConversationId(undefined);
    setError(null);
  };

  const handleClearConversation = () => {
    if (window.confirm('Are you sure you want to clear this conversation?')) {
      setMessages([]);
      setConversationId(undefined);
      setError(null);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-3 sm:px-4 py-3 sm:py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 sm:space-x-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-lg sm:text-xl">O</span>
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl font-bold text-gray-800">L'Oracle</h1>
                <p className="text-xs sm:text-sm text-gray-500 hidden sm:block">RAG Documentation Assistant</p>
              </div>
            </div>
            <div className="flex items-center space-x-1 sm:space-x-2">
              <label className="hidden md:flex items-center space-x-2 text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={useStreaming}
                  onChange={(e) => setUseStreaming(e.target.checked)}
                  className="rounded"
                />
                <span>Streaming</span>
              </label>
              <button
                onClick={handleNewConversation}
                className="px-2 sm:px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-xs sm:text-sm font-medium"
              >
                <span className="hidden sm:inline">New Conversation</span>
                <span className="sm:hidden">New</span>
              </button>
              {messages.length > 0 && (
                <button
                  onClick={handleClearConversation}
                  className="px-2 sm:px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors text-xs sm:text-sm font-medium"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-3 sm:px-4 py-2 sm:py-3">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-2 flex-1 min-w-0">
              <svg className="w-4 h-4 sm:w-5 sm:h-5 text-red-600 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <span className="text-red-800 text-xs sm:text-sm truncate">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-600 hover:text-red-800 ml-2 flex-shrink-0"
            >
              <svg className="w-4 h-4 sm:w-5 sm:h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Messages */}
      <MessageList messages={messages} isLoading={isLoading} />

      {/* Input */}
      <MessageInput onSendMessage={handleSendMessage} disabled={isLoading} />
    </div>
  );
}

export default App;
