ifndef OPENAPI_SPEC_URI
OPENAPI_SPEC_URI=https://api.furiosa.ai/openapi.yaml
endif

generate-client:
	@OPENAPI_GENERATOR_VERSION=3.0.2 \
	java -jar share/openapi-generator-cli-5.0.1.jar generate \
	-i ${OPENAPI_SPEC_URI} -g python -c share/default.json \
  -o ./;

compile:
	MACOSX_DEPLOYMENT_TARGET=10.15 python ./setup.py develop

build:
	MACOSX_DEPLOYMENT_TARGET=10.15 python ./setup.py build

lint:
	pylint furiosa

wheel: compile
	MACOSX_DEPLOYMENT_TARGET=10.15 python ./setup.py bdist_wheel

test:
	python -m unittest discover -v -s ./tests

doc:
	pdoc3 --html --force --output-dir html_output furiosa	

clean:
	rm -rf build nux/_api/*.so dist/*

default: build
