.PHONY: lint
lint:
	flake8 ./halcyon
	mypy ./halcyon

.PHONY: clean
clean:
	rm -rf ./**/__pycache__
	rm -rf ./**/.mypy_cache
