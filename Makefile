.PHONY: build run test test-local test-cov test-unit shell clean test-coverage

build:
	docker build -t slackbot-v2 .

run:
	docker compose up

# Docker-based tests
test:
	docker compose run --rm bot pytest -p no:cacheprovider tests/

test-cov:
	docker compose run --rm bot python -m pytest

# Comprehensive test coverage report 
test-coverage:
	docker compose run --rm bot bash -c "coverage run -m pytest tests/ && coverage report --include='app/*' > tests/coverage/report.txt && cat tests/coverage/report.txt"

# Local tests (run outside of Docker)
test-local:
	python -m pytest

test-cov-local:
	python -m pytest --cov=app --cov-report=term

test-unit:
	docker compose run --rm bot pytest -p no:cacheprovider tests/unit/

shell:
	docker compose run --rm bot bash

clean:
	docker compose down -v
	docker rmi slackbot-v2 