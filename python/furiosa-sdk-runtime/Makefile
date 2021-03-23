compile:
	MACOSX_DEPLOYMENT_TARGET=10.15 python ./setup.py develop

build:
	MACOSX_DEPLOYMENT_TARGET=10.15 python ./setup.py build

lint:
	pylint furiosa

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

test:
	python -m unittest discover -v -s ./tests

doc:
	pdoc3 --html --force --output-dir html_output furiosa	

clean:
	rm -rf build dist/* *.egg-info

default: compile
