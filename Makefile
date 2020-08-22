.PHONY: docker
docker:
	rm -f ./Dockerfile
	ln -s ./Dockerfiles/Dockerfile ./Dockerfile
	docker build -t relastle/halcyon:0.2.0 .
	docker push relastle/halcyon:0.2.0

.PHONY: integration_test
integration_test:
	./bin/halcyon --config ./net_model/human_WGS/config.json -i halcyon/tests/test_fast5s -o ./output.fasta
