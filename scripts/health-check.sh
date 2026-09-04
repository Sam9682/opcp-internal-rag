#!/bin/bash

# Health check script for all services

set -e

echo "Checking service health..."
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_service() {
    local name=$1
    local url=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name: healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $name: unhealthy"
        return 1
    fi
}

# Check each service
check_service "PostgreSQL" "http://localhost:5432" || echo "  (Database may not have HTTP endpoint)"
check_service "API Backend" "http://localhost:8080/health"
check_service "Embedding Service" "http://localhost:8001/health"
check_service "LLM Service" "http://localhost:8002/health"
check_service "Web UI" "http://localhost:3000"

echo ""
echo "Health check complete"
