default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

install_full:
	pip install .[full]

lint:
	isort --check --diff . && black --check --diff .

upload_test:
	flit publish --repository testpypi

upload:
	flit publish

test:
	if [ -d ./tests ]; then python -m unittest discover -v -s ./tests; fi;

clean:
	rm -rf build dist *.egg-info doc_srcs
