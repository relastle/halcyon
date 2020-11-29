.PHONY: lint
lint:
	flake8 ./halcyon
	mypy ./halcyon

.PHONY: integration_test
integration_test:
	coverage run --omit='./integration_tests/**/*,./setup.py' --source=. -m pytest -vvs --diff-type=split ./integration_tests
	coverage xml -i
	coverage report -m

.PHONY: docker
docker:
	docker build -t relastle/halcyon -f ./Dockerfiles/Dockerfile .

.PHONY: docker-test
docker-test:
	docker build -t relastle/halcyon-test -f ./Dockerfiles/Dockerfile.test .

.PHONY: clean
clean:
	rm -rf ./**/__pycache__
	rm -rf ./**/.mypy_cache
