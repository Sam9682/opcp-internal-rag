#!/bin/bash

# Setup script for initial deployment

set -e

echo "==================================="
echo "Generic RAG Web Application Setup"
echo "==================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
fi

echo "✓ Docker and Docker Compose are installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  Please edit .env and set secure passwords before starting services"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p docs
mkdir -p backups
echo "✓ Directories created"
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    echo "  You can enable GPU acceleration by setting DEVICE=cuda in .env"
    echo ""
else
    echo "ℹ No NVIDIA GPU detected, will use CPU mode"
    echo ""
fi

# Build services
echo "Building Docker images..."
docker-compose build
echo "✓ Docker images built"
echo ""

echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and set secure passwords"
echo "2. Add markdown files to docs/ directory"
echo "3. Start services: docker-compose up -d"
echo "4. Access web UI: http://localhost:3000"
echo ""
