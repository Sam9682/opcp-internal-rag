#!/bin/bash
# Generate self-signed certificates for development

set -e

echo "Generating TLS certificates for RAG application..."

# Create certs directory
mkdir -p certs

# Generate API Gateway certificate
echo "Generating API Gateway certificate..."
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout certs/api-key.pem \
  -out certs/api-cert.pem \
  -days 365 \
  -subj "/CN=localhost/O=RAG Application/C=US" \
  -addext "subjectAltName=DNS:localhost,DNS:api-backend,DNS:rag-api,IP:127.0.0.1"

# Generate PostgreSQL server certificate
echo "Generating PostgreSQL server certificate..."
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout certs/postgres-key.pem \
  -out certs/postgres-cert.pem \
  -days 365 \
  -subj "/CN=postgres/O=RAG Application/C=US" \
  -addext "subjectAltName=DNS:postgres,DNS:rag-postgres,DNS:localhost"

# Generate CA certificate for client verification (optional)
echo "Generating CA certificate..."
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout certs/ca-key.pem \
  -out certs/ca-cert.pem \
  -days 365 \
  -subj "/CN=RAG CA/O=RAG Application/C=US"

# Set proper permissions
chmod 600 certs/*-key.pem
chmod 644 certs/*-cert.pem

echo "Certificates generated successfully in ./certs/"
echo ""
echo "Files created:"
ls -lh certs/
echo ""
echo "⚠️  WARNING: These are self-signed certificates for DEVELOPMENT ONLY"
echo "For production, use certificates from a trusted Certificate Authority"
