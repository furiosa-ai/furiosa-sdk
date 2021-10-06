default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

build:
	python -m build

lint:
	pylint furiosa/tools

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

test:
	python ./setup.py test

clean:
	rm -rf build dist/* *.egg-info
