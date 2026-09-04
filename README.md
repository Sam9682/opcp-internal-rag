# Generic RAG Web Application

A sovereign, self-hosted Retrieval-Augmented Generation (RAG) web application for querying markdown documentation through an AI-powered chatbot interface called "L'Oracle".

## Overview

This application provides intelligent documentation search and question-answering capabilities using:

- **Vector Search**: PostgreSQL with pgvector extension for semantic similarity search
- **Embeddings**: BGE-M3 model for generating 1024-dimensional vectors
- **LLM**: Mistral-7B-Instruct for natural language response generation
- **Safety**: LLM Guard for input/output validation
- **Architecture**: Docker Compose microservices for easy deployment

## Features

- 📚 Automatic markdown documentation ingestion with file watching
- 🔍 Semantic search with vector embeddings (1024-dimensional BGE-M3)
- 💬 Conversational AI interface (L'Oracle) with context awareness
- 🛡️ Built-in safety checks and content filtering (LLM Guard)
- 🔒 Sovereign deployment - full data control, no external dependencies
- 📊 Conversation history and context management
- 🚀 GPU acceleration support (optional, 10-50x speedup)
- 🔐 JWT authentication and rate limiting
- 📈 Prometheus metrics and structured logging
- 🌐 WebSocket support for streaming responses

## Architecture

The application consists of the following microservices:

- **postgres**: PostgreSQL 16 with pgvector extension for vector storage
- **embedding-service**: Generates vector embeddings using BGE-M3 model
- **llm-service**: LLM inference for response generation (Mistral-7B)
- **ingestion-service**: Monitors and processes markdown files automatically
- **api-backend**: REST API gateway with authentication and rate limiting
- **web-ui**: React-based chat interface (L'Oracle)

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## System Requirements

### Minimum (CPU-only)
- **CPU**: 4 cores
- **RAM**: 16 GB
- **Storage**: 50 GB SSD
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows with WSL2
- **Docker**: 24.0+
- **Docker Compose**: 2.20+

### Recommended (GPU-accelerated)
- **CPU**: 8 cores
- **RAM**: 32 GB
- **Storage**: 100 GB SSD
- **GPU**: NVIDIA GPU with 16+ GB VRAM (RTX 4090, A100, etc.)
- **CUDA**: 12.0+
- **NVIDIA Container Toolkit**: Latest version

### Production (High Performance)
- **CPU**: 16+ cores
- **RAM**: 64 GB
- **Storage**: 500 GB SSD
- **GPU**: NVIDIA GPU with 24+ GB VRAM
- **Network**: Load balancer for API services
- **Database**: PostgreSQL with read replicas

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-generic-web-rag
```

### 2. Configure Environment

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

Edit `.env` with your preferred settings. Key variables:

```bash
# Database Configuration
POSTGRES_PASSWORD=your-secure-password-here
POSTGRES_PORT=5432

# Model Configuration
EMBEDDING_MODEL=BAAI/bge-m3              # Embedding model (1024-dim)
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2  # LLM for responses
DEVICE=cpu                                # Use 'cuda' for GPU acceleration

# RAG Configuration
TOP_K=5                                   # Number of context chunks
SIMILARITY_THRESHOLD=0.7                  # Minimum similarity score
CHUNK_SIZE=512                            # Tokens per chunk
CHUNK_OVERLAP=50                          # Overlap between chunks

# API Configuration
RATE_LIMIT_ANONYMOUS=100                  # Requests/hour for anonymous users
RATE_LIMIT_AUTHENTICATED=1000             # Requests/hour for authenticated users
MAX_PROMPT_TOKENS=4096                    # Maximum prompt size

# Security (Optional)
ENABLE_TLS=false                          # Enable TLS/SSL
JWT_SECRET=your-jwt-secret-here           # JWT signing secret
```

### 3. Add Documentation

Place your markdown files in the `docs/` directory:

```bash
# Copy your documentation
cp /path/to/your/docs/*.md docs/

# Or create sample documentation
echo "# Sample Documentation\n\nThis is a test document." > docs/sample.md
```

The ingestion service will automatically detect and process files within 10 seconds.

### 4. Start the Application

**CPU-only mode** (default):
```bash
docker-compose up -d
```

**GPU-accelerated mode** (requires NVIDIA GPU and Container Toolkit):
```bash
# Set DEVICE=cuda in .env first
docker-compose up -d
```

**View logs** during startup:
```bash
docker-compose logs -f
```

### 5. Verify Services

Check that all services are healthy:

```bash
# Using the health check script
./scripts/health-check.sh

# Or manually check each service
curl http://localhost:8080/health        # API Gateway
curl http://localhost:8001/health        # Embedding Service
curl http://localhost:8002/health        # LLM Service
```

### 6. Access the Application

- **Web UI (L'Oracle)**: http://localhost:3000
- **API Documentation**: http://localhost:8080/docs
- **API Endpoint**: http://localhost:8080/api/query
- **Metrics**: http://localhost:8080/metrics

### 7. Test a Query

Using the Web UI:
1. Open http://localhost:3000
2. Type a question about your documentation
3. L'Oracle will search and provide an answer with sources

Using the API:
```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this documentation about?",
    "top_k": 5
  }'
```

## Configuration Options

### Environment Variables

All configuration is managed through environment variables in the `.env` file.

#### Database Settings

```bash
POSTGRES_DB=rag_db                        # Database name
POSTGRES_USER=rag_user                    # Database user
POSTGRES_PASSWORD=changeme                # Database password (CHANGE THIS!)
POSTGRES_PORT=5432                        # PostgreSQL port
```

#### Model Settings

```bash
# Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3               # Model for text embeddings
EMBEDDING_DIMENSION=1024                  # Embedding vector dimension
EMBEDDING_BATCH_SIZE=32                   # Batch size for embedding generation

# LLM Model
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2  # Language model
LLM_MAX_LENGTH=4096                       # Maximum context length
TEMPERATURE=0.7                           # Response randomness (0.0-1.0)
MAX_TOKENS=512                            # Maximum response tokens

# Device Configuration
DEVICE=cpu                                # 'cpu' or 'cuda' for GPU
GPU_COUNT=1                               # Number of GPUs to use
```

#### RAG Pipeline Settings

```bash
# Document Processing
CHUNK_SIZE=512                            # Tokens per text chunk
CHUNK_OVERLAP=50                          # Overlap between chunks
DOCS_PATH=/docs                           # Documentation directory

# Query Processing
TOP_K=5                                   # Number of context chunks to retrieve
SIMILARITY_THRESHOLD=0.7                  # Minimum similarity score (0.0-1.0)
MAX_PROMPT_TOKENS=4096                    # Maximum prompt size
```

#### API & Security Settings

```bash
# Rate Limiting
RATE_LIMIT_ANONYMOUS=100                  # Requests/hour for anonymous users
RATE_LIMIT_AUTHENTICATED=1000             # Requests/hour for authenticated users

# Authentication (Optional)
ENABLE_AUTH=false                         # Enable JWT authentication
JWT_SECRET=your-secret-key                # JWT signing secret
JWT_EXPIRATION=86400                      # Token expiration (seconds)

# TLS/SSL (Optional)
ENABLE_TLS=false                          # Enable HTTPS
SSL_CERTFILE=/certs/cert.pem              # SSL certificate path
SSL_KEYFILE=/certs/key.pem                # SSL private key path
```

#### Monitoring & Logging

```bash
# Logging
LOG_LEVEL=INFO                            # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json                           # 'json' or 'text'

# Monitoring
ENABLE_METRICS=true                       # Enable Prometheus metrics
SENTRY_DSN=                               # Sentry error tracking (optional)
ENVIRONMENT=development                   # Environment name
```

#### Caching (Optional)

```bash
# Redis Cache
ENABLE_CACHE=false                        # Enable Redis caching
REDIS_HOST=redis                          # Redis hostname
REDIS_PORT=6379                           # Redis port
CACHE_TTL=3600                            # Cache TTL in seconds
```

### GPU Acceleration

To enable GPU acceleration for faster inference:

1. **Install NVIDIA drivers** (version 535+ recommended)
2. **Install NVIDIA Container Toolkit**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. **Verify GPU access**:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

4. **Update `.env`**:
   ```bash
   DEVICE=cuda
   GPU_COUNT=1
   ```

5. **Restart services**:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

**Performance improvement**: GPU acceleration provides 10-50x speedup for embedding generation and LLM inference.

### Advanced Configuration

#### Custom Docker Compose Override

Create `docker-compose.override.yml` for custom configurations:

```yaml
version: '3.8'

services:
  api-backend:
    environment:
      - CUSTOM_SETTING=value
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

#### Database Tuning

Edit `database/postgresql.conf` for performance tuning:

```conf
# Memory settings
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 64MB

# Connection settings
max_connections = 200

# Performance settings
random_page_cost = 1.1
effective_io_concurrency = 200
```

#### Vector Index Optimization

Adjust HNSW index parameters in `database/init.sql`:

```sql
-- For better search quality (slower build)
CREATE INDEX idx_text_chunks_embedding ON text_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 32, ef_construction = 128);

-- For faster build (lower quality)
CREATE INDEX idx_text_chunks_embedding ON text_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
```

## Usage

### Web Interface (L'Oracle)

1. **Open the application**: Navigate to http://localhost:3000
2. **Ask a question**: Type your query in the chat input
3. **View response**: L'Oracle will:
   - Search your documentation using semantic similarity
   - Retrieve the most relevant context chunks
   - Generate a natural language answer
   - Display source citations with similarity scores
4. **Continue conversation**: Ask follow-up questions with context awareness
5. **Start fresh**: Click "New Conversation" to clear history

### API Usage

#### Query Endpoint

**POST** `/api/query`

Process a query through the RAG pipeline:

```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I configure authentication?",
    "conversation_id": null,
    "top_k": 5
  }'
```

Response:
```json
{
  "answer": "To configure authentication...",
  "sources": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "title": "Authentication Guide",
      "excerpt": "Authentication can be configured...",
      "similarity_score": 0.89
    }
  ],
  "conversation_id": "uuid"
}
```

#### WebSocket Streaming

**WebSocket** `/ws/query`

Stream responses in real-time:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/query');

ws.onopen = () => {
  ws.send(JSON.stringify({
    query: "Explain the deployment process",
    conversation_id: null,
    top_k: 5
  }));
};

ws.onmessage = (event) => {
  if (event.data === '[DONE]') {
    console.log('Streaming complete');
  } else if (event.data.startsWith('[ERROR]')) {
    console.error('Error:', event.data);
  } else {
    // Display chunk
    console.log(event.data);
  }
};
```

#### Conversation History

**GET** `/api/conversations/{conversation_id}`

Retrieve conversation history:

```bash
curl http://localhost:8080/api/conversations/{conversation_id}
```

Response:
```json
{
  "conversation_id": "uuid",
  "messages": [
    {
      "role": "user",
      "content": "What is RAG?",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    {
      "role": "assistant",
      "content": "RAG stands for...",
      "timestamp": "2024-01-01T12:00:05Z",
      "sources": [...]
    }
  ],
  "total_messages": 2
}
```

#### Manual Ingestion

**POST** `/api/ingest`

Trigger manual document ingestion:

```bash
curl -X POST http://localhost:8080/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/docs"
  }'
```

### Adding Documentation

#### Automatic Ingestion

Simply add markdown files to the `docs/` directory:

```bash
# Add new file
echo "# New Documentation\n\nContent here." > docs/new-doc.md

# Update existing file
vim docs/existing-doc.md

# Delete file
rm docs/old-doc.md
```

The ingestion service automatically:
- Detects new files within 10 seconds
- Re-processes modified files
- Removes deleted files from the database

#### Manual Ingestion

Trigger ingestion via API:

```bash
curl -X POST http://localhost:8080/api/ingest
```

Or restart the ingestion service:

```bash
docker-compose restart ingestion-service
```

### Authentication (Optional)

If JWT authentication is enabled:

1. **Obtain a token** (implement your own auth endpoint)
2. **Include in requests**:
   ```bash
   curl -X POST http://localhost:8080/api/query \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}'
   ```

### Monitoring

#### Health Checks

```bash
# All services
./scripts/health-check.sh

# Individual services
curl http://localhost:8080/health        # API Gateway
curl http://localhost:8001/health        # Embedding Service
curl http://localhost:8002/health        # LLM Service
```

#### Metrics

Access Prometheus metrics:

```bash
curl http://localhost:8080/metrics
```

Key metrics:
- `rag_query_duration_seconds` - Query processing time
- `rag_query_total` - Total queries processed
- `rag_query_errors_total` - Query errors by type
- `rag_embedding_cache_hits_total` - Cache hit rate
- `rag_vector_search_duration_seconds` - Search latency

#### Logs

View service logs:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api-backend
docker-compose logs -f ingestion-service

# Last 100 lines
docker-compose logs --tail=100 api-backend
```

## Development

### Project Structure

```
.
├── docker-compose.yml          # Service orchestration
├── database/
│   └── init.sql               # Database schema and indexes
├── services/
│   ├── shared/                # Common utilities and models
│   ├── embedding/             # Embedding service
│   ├── llm/                   # LLM service
│   ├── ingestion/             # Document ingestion
│   ├── api/                   # API gateway
│   └── web/                   # Web UI
└── docs/                      # Documentation files
```

### Building Services

Build individual services:

```bash
docker-compose build embedding-service
docker-compose build api-backend
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api-backend
```

### Stopping Services

```bash
docker-compose down
```

To remove volumes (database data):

```bash
docker-compose down -v
```

## Database Schema

The application uses PostgreSQL with the following main tables:

- **documents**: Stores markdown files and metadata
- **text_chunks**: Stores text chunks with vector embeddings
- **conversations**: Manages conversation sessions
- **messages**: Stores user-assistant message exchanges
- **ingestion_jobs**: Tracks document processing jobs

## Security

- JWT authentication for API access
- Rate limiting (100 req/hour anonymous, 1000 req/hour authenticated)
- LLM Guard for prompt injection detection
- Input/output safety validation
- TLS encryption for network communication
- Conversation isolation per user

## Performance

Expected performance metrics:

- **Vector Search**: <50ms for 100K chunks
- **Embedding Generation**: 100-500 chunks/sec (GPU), 10-50 chunks/sec (CPU)
- **LLM Response**: 2-5 seconds for 512 tokens (GPU)
- **Concurrent Users**: 100+ with proper scaling

## Troubleshooting

### Services Won't Start

**Check logs**:
```bash
docker-compose logs
```

**Common issues**:

1. **Port conflicts**:
   ```bash
   # Check what's using the port
   sudo lsof -i :8080
   sudo lsof -i :5432
   
   # Change ports in .env
   POSTGRES_PORT=5433
   API_PORT=8081
   ```

2. **Insufficient memory**:
   ```bash
   # Check available memory
   free -h
   
   # Increase Docker memory limit (Docker Desktop)
   # Settings > Resources > Memory > 16GB+
   ```

3. **Docker daemon not running**:
   ```bash
   sudo systemctl start docker
   ```

### Ingestion Not Working

**Check ingestion service logs**:
```bash
docker-compose logs ingestion-service
```

**Common issues**:

1. **No markdown files in docs/**:
   ```bash
   ls -la docs/
   # Add some .md files
   ```

2. **Permission issues**:
   ```bash
   # Fix permissions
   chmod -R 755 docs/
   ```

3. **Database connection failed**:
   ```bash
   # Check PostgreSQL is running
   docker-compose ps postgres
   
   # Check database logs
   docker-compose logs postgres
   ```

### Slow Responses

**Enable GPU acceleration** (if available):
```bash
# Update .env
DEVICE=cuda

# Restart services
docker-compose down
docker-compose up -d
```

**Reduce context size**:
```bash
# Update .env
TOP_K=3                    # Reduce from 5 to 3
CHUNK_SIZE=256             # Reduce from 512
```

**Use quantized models**:
```bash
# Update .env
LLM_MODEL=TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
```

**Add more resources**:
```yaml
# docker-compose.override.yml
services:
  llm-service:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
```

### GPU Not Detected

**Verify NVIDIA drivers**:
```bash
nvidia-smi
```

**Check NVIDIA Container Toolkit**:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

**If not working**:
```bash
# Reinstall NVIDIA Container Toolkit
sudo apt-get purge nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Database Issues

**Connection refused**:
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check health
docker-compose exec postgres pg_isready -U rag_user
```

**Out of disk space**:
```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a --volumes
```

**Slow queries**:
```bash
# Check database size
docker-compose exec postgres psql -U rag_user -d rag_db -c "
  SELECT pg_size_pretty(pg_database_size('rag_db'));
"

# Vacuum and analyze
docker-compose exec postgres psql -U rag_user -d rag_db -c "
  VACUUM ANALYZE;
"
```

### Memory Errors

**OOM (Out of Memory) errors**:

1. **Reduce model size**:
   ```bash
   # Use smaller embedding model
   EMBEDDING_MODEL=BAAI/bge-base-en-v1.5  # 438 MB vs 2.27 GB
   
   # Use quantized LLM
   LLM_MODEL=TheBloke/Mistral-7B-Instruct-v0.2-GPTQ
   ```

2. **Increase swap space**:
   ```bash
   # Create 8GB swap file
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Limit container memory**:
   ```yaml
   # docker-compose.override.yml
   services:
     llm-service:
       deploy:
         resources:
           limits:
             memory: 8G
   ```

### API Errors

**429 Too Many Requests**:
```bash
# Increase rate limits in .env
RATE_LIMIT_ANONYMOUS=200
RATE_LIMIT_AUTHENTICATED=2000
```

**400 Bad Request - Security Error**:
- Query was rejected by LLM Guard
- Rephrase your query to avoid:
  - Prompt injection attempts
  - Toxic or harmful content
  - Personally identifiable information

**503 Service Unavailable**:
- Service is starting up (wait 1-2 minutes)
- Service crashed (check logs)
- Restart the service:
  ```bash
  docker-compose restart api-backend
  ```

### Getting Help

1. **Check logs** for all services:
   ```bash
   docker-compose logs > logs.txt
   ```

2. **Check system resources**:
   ```bash
   docker stats
   ```

3. **Verify configuration**:
   ```bash
   cat .env
   ```

4. **Run health checks**:
   ```bash
   ./scripts/health-check.sh
   ```

5. **Review documentation**:
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
   - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture details
   - [docs/API.md](docs/API.md) - API documentation

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

## Support

[Support Information]

## Additional Resources

### Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed system architecture and component design
- **[API.md](docs/API.md)** - Complete API reference with examples
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment guide
- **[SECURITY_IMPLEMENTATION.md](SECURITY_IMPLEMENTATION.md)** - Security features and best practices
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project

### Performance Optimization

For detailed performance tuning:
- See [DEPLOYMENT.md](DEPLOYMENT.md) for scaling strategies
- Review database optimization in `database/postgresql.conf`
- Check vector index tuning in `database/init.sql`

### Development Workflow

1. **Make changes** to code in `services/`
2. **Run tests**: `pytest services/`
3. **Rebuild service**: `docker-compose build <service>`
4. **Restart**: `docker-compose up -d <service>`
5. **Check logs**: `docker-compose logs -f <service>`
6. **Verify**: Test your changes

### Common Commands

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up -d --build

# Check service status
docker-compose ps

# Run health checks
./scripts/health-check.sh

# Backup database
./scripts/backup.sh

# Access database
docker-compose exec postgres psql -U rag_user -d rag_db
```

## Acknowledgments

This project uses the following open-source technologies:

- **[BGE-M3](https://huggingface.co/BAAI/bge-m3)** - BAAI's multilingual embedding model
- **[Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)** - Mistral AI's instruction-tuned language model
- **[pgvector](https://github.com/pgvector/pgvector)** - PostgreSQL extension for vector similarity search
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[React](https://react.dev/)** - JavaScript library for building user interfaces
- **[Docker](https://www.docker.com/)** - Container platform
- **[PostgreSQL](https://www.postgresql.org/)** - Open-source relational database

## Roadmap

Future enhancements:
- [ ] Multi-language support for UI
- [ ] Advanced search filters
- [ ] Document versioning
- [ ] User management system
- [ ] API key authentication
- [ ] Webhook notifications
- [ ] Export conversation history
- [ ] Custom model support
- [ ] Kubernetes deployment manifests
- [ ] Grafana dashboards

## FAQ

**Q: Can I use a different LLM model?**  
A: Yes, update `LLM_MODEL` in `.env` to any Hugging Face model compatible with the transformers library.

**Q: How do I use OpenAI API instead of local LLM?**  
A: Modify `services/shared/llm_service.py` to use OpenAI's API client instead of loading a local model.

**Q: Can I deploy this on cloud platforms?**  
A: Yes, see [DEPLOYMENT.md](DEPLOYMENT.md) for cloud deployment instructions.

**Q: How much does GPU acceleration help?**  
A: GPU provides 10-50x speedup for embeddings and LLM inference, reducing query time from 30s to 3-5s.

**Q: Is this production-ready?**  
A: Yes, with proper configuration. See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup.

**Q: Can I use this for non-English documentation?**  
A: Yes, BGE-M3 supports 100+ languages. For best results, use a multilingual LLM.

**Q: How do I backup my data?**  
A: Use `./scripts/backup.sh` to backup the PostgreSQL database. Store backups securely off-site.

**Q: What's the maximum document size?**  
A: No hard limit. Large documents are automatically chunked into 512-token segments.

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## Contact

- **Issues**: [GitHub Issues](repository-url/issues)
- **Discussions**: [GitHub Discussions](repository-url/discussions)
- **Email**: support@example.com
- **Website**: https://example.com
