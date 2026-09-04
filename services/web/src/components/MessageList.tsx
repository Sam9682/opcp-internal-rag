import React, { useEffect, useRef } from 'react';
import { Message } from '../types';
import SourceCard from './SourceCard';

interface MessageListProps {
  messages: Message[];
  isLoading?: boolean;
}

const MessageList: React.FC<MessageListProps> = ({ messages, isLoading }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto p-2 sm:p-4 space-y-3 sm:space-y-4">
      {messages.length === 0 && !isLoading && (
        <div className="flex items-center justify-center h-full text-gray-500 px-4">
          <div className="text-center">
            <h2 className="text-xl sm:text-2xl font-semibold mb-2">Welcome to L'Oracle</h2>
            <p className="text-sm sm:text-base">Ask me anything about the documentation</p>
          </div>
        </div>
      )}

      {messages.map((message, index) => (
        <div
          key={index}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-full sm:max-w-3xl rounded-lg p-3 sm:p-4 ${
              message.role === 'user'
                ? 'bg-blue-600 text-white'
                : 'bg-white border border-gray-200'
            }`}
          >
            <div className="flex items-start space-x-2">
              <div className="flex-shrink-0">
                {message.role === 'user' ? (
                  <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-blue-700 flex items-center justify-center text-white font-semibold text-xs sm:text-base">
                    U
                  </div>
                ) : (
                  <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-purple-600 flex items-center justify-center text-white font-semibold text-xs sm:text-base">
                    O
                  </div>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="whitespace-pre-wrap break-words text-sm sm:text-base">{message.content}</div>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 sm:mt-4 space-y-2">
                    <div className="text-xs sm:text-sm font-semibold text-gray-700">Sources:</div>
                    {message.sources.map((source, idx) => (
                      <SourceCard key={idx} source={source} />
                    ))}
                  </div>
                )}
                <div className="text-xs mt-2 opacity-70">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        </div>
      ))}

      {isLoading && (
        <div className="flex justify-start">
          <div className="max-w-full sm:max-w-3xl rounded-lg p-3 sm:p-4 bg-white border border-gray-200">
            <div className="flex items-start space-x-2">
              <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-purple-600 flex items-center justify-center text-white font-semibold text-xs sm:text-base">
                O
              </div>
              <div className="flex space-x-1 items-center">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;
