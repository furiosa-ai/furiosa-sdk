default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

lint:
	isort --check --diff . && black --check --diff .

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

test:
	python -m unittest

clean:
	rm -rf build dist/* *.egg-info
