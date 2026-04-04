.PHONY: help dev up down logs clean api-dev frontend-dev stack-dev build build-api build-frontend health-api health-frontend shell-backend shell-frontend eval-retrieval

help: ## Show available commands
	@echo "RAGagument"
	@echo ""
	@echo "Development:"
	@echo "  dev             Start the Docker dev stack in the foreground"
	@echo "  up              Start the Docker dev stack in the background"
	@echo "  down            Stop the Docker dev stack"
	@echo "  logs            Follow Docker dev logs"
	@echo "  clean           Stop the Docker dev stack and remove volumes"
	@echo "  api-dev         Start the FastAPI backend locally"
	@echo "  frontend-dev    Start the Next.js frontend locally"
	@echo "  stack-dev       Print the local two-terminal workflow"
	@echo "  eval-retrieval  Run the sample retrieval benchmark"
	@echo ""
	@echo "Build:"
	@echo "  build           Build both dev images"
	@echo "  build-api       Build the backend API image"
	@echo "  build-frontend  Build the frontend image"
	@echo ""
	@echo "Health:"
	@echo "  health-api      Check the local FastAPI health endpoint"
	@echo "  health-frontend Check the local frontend root"

dev: ## Start the Docker dev stack in the foreground
	docker compose -f docker-compose.dev.yml up --build

up: ## Start the Docker dev stack in the background
	docker compose -f docker-compose.dev.yml up --build -d

down: ## Stop the Docker dev stack
	docker compose -f docker-compose.dev.yml down

logs: ## Follow Docker dev logs
	docker compose -f docker-compose.dev.yml logs -f

clean: ## Stop the Docker dev stack and remove named volumes
	docker compose -f docker-compose.dev.yml down -v --remove-orphans

api-dev: ## Start the FastAPI backend locally
	uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

frontend-dev: ## Start the Next.js frontend locally
	cd frontend && npm run dev

stack-dev: ## Print the local two-terminal workflow
	@echo "Terminal 1: make api-dev"
	@echo "Terminal 2: make frontend-dev"

build: build-api build-frontend ## Build both dev images

build-api: ## Build the backend API image
	docker build -f Dockerfile.api -t ragagument-api:dev .

build-frontend: ## Build the frontend image
	docker build -f frontend/Dockerfile -t ragagument-frontend:dev frontend

health-api: ## Check the local FastAPI health endpoint
	curl -f http://localhost:8000/api/health || echo "API health check failed"

health-frontend: ## Check the local frontend root
	curl -f http://localhost:3000 || echo "Frontend health check failed"

shell-backend: ## Open a shell in the backend container
	docker compose -f docker-compose.dev.yml exec backend /bin/sh

shell-frontend: ## Open a shell in the frontend container
	docker compose -f docker-compose.dev.yml exec frontend /bin/sh

eval-retrieval: ## Run the sample retrieval benchmark
	python scripts/run_retrieval_eval.py --dataset evals/sample_retrieval_eval.json
