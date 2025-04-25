.PHONY: build run test shell clean

build:
	docker build -t slackbot-v2 .

run:
	docker compose up

test:
	docker compose run --rm bot pytest -p no:cacheprovider tests/

shell:
	docker compose run --rm bot bash

clean:
	docker compose down -v
	docker rmi slackbot-v2 