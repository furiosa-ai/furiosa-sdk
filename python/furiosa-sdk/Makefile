ifndef OPENAPI_SPEC_URI
OPENAPI_SPEC_URI=https://api.furiosa.ai/openapi.yaml
endif

default: compile

generate-openapi:
	@OPENAPI_GENERATOR_VERSION=3.0.2 \
	java -jar share/openapi-generator-cli-5.0.1.jar generate \
	--global-property apiTests=false,apiDocs=false,modelTests=false,modelDocs=false \
	-i ${OPENAPI_SPEC_URI} -g python -c share/default.json \
  -o ./;

compile:
	MACOSX_DEPLOYMENT_TARGET=10.15 python ./setup.py develop

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
	mkdir -p build/docs/src
	cp -a ../furiosa-sdk-runtime/furiosa build/docs/src
	cp -a furiosa build/docs/src
	pdoc3 --html --force --output-dir build/docs/html build/docs/src/furiosa

clean:
	rm -rf build dist *.egg-info doc_srcs
