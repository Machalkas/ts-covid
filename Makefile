all: build run

build:
	docker build -t app .

run:
	docker-compose up