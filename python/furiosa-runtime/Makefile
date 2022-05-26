default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

lint:
	isort --check --diff . && black --check --diff .

upload_test:
	flit publish --repository testpypi

upload:
	flit publish

test:
	FURIOSA_TEST_IS_RUNNING=1 python -m unittest discover -v -s ./tests

clean:
	rm -rf build dist/* *.egg-info
