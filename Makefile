.PHONY: help up down build logs clean restart health

help:
	@echo "Generic RAG Web Application - Make Commands"
	@echo ""
	@echo "  make up          - Start all services"
	@echo "  make down        - Stop all services"
	@echo "  make build       - Build all services"
	@echo "  make logs        - View logs from all services"
	@echo "  make clean       - Stop services and remove volumes"
	@echo "  make restart     - Restart all services"
	@echo "  make health      - Check health of all services"
	@echo ""

up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	@echo "All services stopped and volumes removed"

restart:
	docker-compose restart

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8080/health || echo "API: Not responding"
	@curl -s http://localhost:8001/health || echo "Embedding: Not responding"
	@curl -s http://localhost:8002/health || echo "LLM: Not responding"
