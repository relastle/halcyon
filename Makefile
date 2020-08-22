.PHONY: lint
lint:
	flake8 ./halcyon
	mypy ./halcyon

.PHONY: integration_test
integration_test:
	pytest -vvs ./integration_tests

.PHONY: clean
clean:
	rm -rf ./**/__pycache__
	rm -rf ./**/.mypy_cache
