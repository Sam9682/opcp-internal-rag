/**
 * Type definitions for the RAG Web UI
 */

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Source[];
}

export interface Source {
  chunk_id: string;
  document_id: string;
  title: string;
  excerpt: string;
  file_path: string;
  similarity_score: number;
}

export interface QueryRequest {
  query: string;
  conversation_id?: string;
  top_k?: number;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  conversation_id: string;
}

export interface ConversationMessage {
  role: string;
  content: string;
  timestamp: string;
  sources?: Source[];
}

export interface ConversationResponse {
  conversation_id: string;
  messages: ConversationMessage[];
  total_messages: number;
}
