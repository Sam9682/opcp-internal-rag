#!/bin/bash

# Validation script to check project setup

set -e

echo "==================================="
echo "Project Setup Validation"
echo "==================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

validate() {
    local check=$1
    local path=$2
    
    if [ -e "$path" ]; then
        echo -e "${GREEN}✓${NC} $check"
        return 0
    else
        echo -e "${RED}✗${NC} $check (missing: $path)"
        return 1
    fi
}

validate_dir() {
    local check=$1
    local path=$2
    
    if [ -d "$path" ]; then
        echo -e "${GREEN}✓${NC} $check"
        return 0
    else
        echo -e "${RED}✗${NC} $check (missing directory: $path)"
        return 1
    fi
}

echo "Checking project structure..."
echo ""

# Core files
validate "Docker Compose configuration" "docker-compose.yml"
validate "Environment template" ".env.example"
validate "README documentation" "README.md"
validate "Deployment guide" "DEPLOYMENT.md"
validate "Contributing guide" "CONTRIBUTING.md"
validate "License file" "LICENSE"
validate "Changelog" "CHANGELOG.md"
validate "Makefile" "Makefile"
validate "Gitignore" ".gitignore"

echo ""
echo "Checking database setup..."
validate "Database init script" "database/init.sql"

echo ""
echo "Checking shared package..."
validate "Shared __init__.py" "services/shared/__init__.py"
validate "Shared models" "services/shared/models.py"
validate "Shared config" "services/shared/config.py"
validate "Shared database" "services/shared/database.py"
validate "Shared requirements" "services/shared/requirements.txt"

echo ""
echo "Checking microservices..."
validate_dir "Embedding service" "services/embedding"
validate "Embedding Dockerfile" "services/embedding/Dockerfile"
validate "Embedding main" "services/embedding/main.py"
validate "Embedding requirements" "services/embedding/requirements.txt"

validate_dir "LLM service" "services/llm"
validate "LLM Dockerfile" "services/llm/Dockerfile"
validate "LLM main" "services/llm/main.py"
validate "LLM requirements" "services/llm/requirements.txt"

validate_dir "Ingestion service" "services/ingestion"
validate "Ingestion Dockerfile" "services/ingestion/Dockerfile"
validate "Ingestion main" "services/ingestion/main.py"
validate "Ingestion requirements" "services/ingestion/requirements.txt"

validate_dir "API service" "services/api"
validate "API Dockerfile" "services/api/Dockerfile"
validate "API main" "services/api/main.py"
validate "API requirements" "services/api/requirements.txt"

validate_dir "Web UI" "services/web"
validate "Web Dockerfile" "services/web/Dockerfile"
validate "Web package.json" "services/web/package.json"
validate "Web index.html" "services/web/index.html"
validate "Web vite config" "services/web/vite.config.ts"

echo ""
echo "Checking scripts..."
validate "Setup script" "scripts/setup.sh"
validate "Backup script" "scripts/backup.sh"
validate "Health check script" "scripts/health-check.sh"

echo ""
echo "Checking documentation..."
validate_dir "Docs directory" "docs"
validate "Sample documentation" "docs/sample-documentation.md"

echo ""
echo "==================================="
echo "Validation complete!"
echo "==================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠${NC}  .env file not found"
    echo "   Run: cp .env.example .env"
    echo ""
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠${NC}  Docker not found"
    echo "   Install Docker to run the application"
    echo ""
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠${NC}  Docker Compose not found"
    echo "   Install Docker Compose to run the application"
    echo ""
fi

echo "Project structure is valid!"
echo ""
