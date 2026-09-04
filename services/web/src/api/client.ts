import axios, { AxiosInstance } from 'axios';
import { QueryRequest, QueryResponse, ConversationResponse } from '../types';

// Get API URL from environment or default to localhost
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

class APIClient {
  private client: AxiosInstance;
  private wsUrl: string;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Convert HTTP URL to WebSocket URL
    this.wsUrl = API_BASE_URL.replace(/^http/, 'ws');

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response) {
          // Server responded with error status
          console.error('API Error:', error.response.data);
          throw new Error(error.response.data.message || 'An error occurred');
        } else if (error.request) {
          // Request made but no response
          console.error('Network Error:', error.message);
          throw new Error('Network error. Please check your connection.');
        } else {
          // Something else happened
          console.error('Error:', error.message);
          throw error;
        }
      }
    );
  }

  /**
   * Send a query to the RAG API (non-streaming)
   */
  async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await this.client.post<QueryResponse>('/api/query', request);
    return response.data;
  }

  /**
   * Send a query with streaming response via WebSocket
   */
  async queryStream(
    request: QueryRequest,
    onChunk: (chunk: string) => void,
    onComplete: (response: QueryResponse) => void,
    onError: (error: Error) => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`${this.wsUrl}/ws/query`);
      let accumulatedResponse = '';
      let hasError = false;

      ws.onopen = () => {
        console.log('WebSocket connected');
        ws.send(JSON.stringify(request));
      };

      ws.onmessage = (event) => {
        try {
          const data = event.data;

          // Check for completion marker
          if (data === '[DONE]') {
            ws.close();
            return;
          }

          // Try to parse as JSON (final response with sources)
          try {
            const response = JSON.parse(data) as QueryResponse;
            onComplete(response);
            resolve();
            return;
          } catch {
            // Not JSON, treat as streaming chunk
            accumulatedResponse += data;
            onChunk(data);
          }
        } catch (error) {
          console.error('Error processing message:', error);
          hasError = true;
          const err = error instanceof Error ? error : new Error('Unknown error');
          onError(err);
          ws.close();
          reject(err);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        hasError = true;
        const error = new Error('WebSocket connection error');
        onError(error);
        reject(error);
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        if (!hasError && event.code !== 1000) {
          const error = new Error(`WebSocket closed unexpectedly: ${event.reason || 'Unknown reason'}`);
          onError(error);
          reject(error);
        }
      };
    });
  }

  /**
   * Get conversation history
   */
  async getConversation(conversationId: string): Promise<ConversationResponse> {
    const response = await this.client.get<ConversationResponse>(
      `/api/conversations/${conversationId}`
    );
    return response.data;
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      return response.status === 200;
    } catch {
      return false;
    }
  }
}

// Export singleton instance
export const apiClient = new APIClient();
