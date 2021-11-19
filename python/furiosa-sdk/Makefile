ifndef OPENAPI_SPEC_URI
OPENAPI_SPEC_URI=https://api.furiosa.ai/openapi.yaml
endif

default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

uninstall:
	pip uninstall -y --quiet furiosa-runtime; \
	pip uninstall -y --quiet furiosa-quantizer; \
	pip uninstall -y --quiet furiosa-model-validator; \
	pip uninstall -y --quiet furiosa-tools; \
	pip uninstall -y --quiet furiosa-server; \
	pip uninstall -y --quiet furiosa-registry; \
	pip uninstall -y --quiet furiosa-sdk

install_full:
	pip install --quiet --use-feature=in-tree-build .[full]

build:
	python -m build

lint:
	pylint furiosa

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

test:
	python -m unittest discover -v -s ./tests

doc:
	mkdir -p build/docs/src/furiosa
	cp -a ../furiosa-runtime/furiosa/runtime build/docs/src/furiosa/runtime
	cp -a ../furiosa-quantizer/furiosa/quantizer build/docs/src/furiosa/quantizer
	cp -a furiosa build/docs/src
	pdoc3 --html --force --output-dir build/docs/html build/docs/src/furiosa

clean:
	rm -rf build dist *.egg-info doc_srcs

update_version:
ifndef SDK_VERSION
	$(error "SDK_VERSION is not set")
endif
	find ../ -name setup.py -exec sed -i "s/version = '[^']*'/version = '${SDK_VERSION}'/" '{}' \;
