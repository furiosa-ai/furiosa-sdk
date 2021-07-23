default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python ./setup.py develop

build:
	python -m build

lint:
	pylint furiosa_sdk_cli

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

test:
	python ./setup.py test

doc:
	pdoc3 --html --force --output-dir html_output furiosa	

clean:
	rm -rf build dist/* *.egg-info
