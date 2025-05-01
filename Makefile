.PHONY: build up down logs test test-coverage clean

# Docker commands
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

# Testing commands
test:
	docker compose run --rm app pytest -xvs tests/

test-coverage:
	docker compose run --rm app pytest --cov=. tests/

# Cleaning commands
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 