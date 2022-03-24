ifndef OPENAPI_SPEC_URI
OPENAPI_SPEC_URI=https://api.furiosa.ai/openapi.yaml
endif

default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

uninstall:
	pip uninstall -y furiosa-runtime; \
	pip uninstall -y furiosa-quantizer; \
	pip uninstall -y furiosa-litmus; \
	pip uninstall -y furiosa-tools; \
	pip uninstall -y furiosa-server; \
	pip uninstall -y furiosa-registry; \
	pip uninstall -y furiosa-sdk

install_full:
	pip install --use-feature=in-tree-build .[full]

build:
	python -m build

lint:
	isort --check --diff . && black --check --diff .

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

test:
	if [ -d ./tests ]; then python -m unittest discover -v -s ./tests; fi;

clean:
	rm -rf build dist *.egg-info doc_srcs

update_version:
ifndef SDK_VERSION
	$(error "SDK_VERSION is not set")
endif
	find ../ -name setup.py -exec sed -i "s/version = \"[^\"]*\"/version = \"${SDK_VERSION}\"/" '{}' \;
