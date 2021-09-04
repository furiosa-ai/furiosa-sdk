default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

build:
	python -m build

lint:
	pylint furiosa/runtime

upload_test:
	twine upload --repository testpypi dist/*

upload:
	twine upload dist/*

test:
	catchsegv python -q -X faulthandler -m unittest discover -v -s ./tests

doc:
	pdoc3 --html --force --output-dir html_output furiosa	

clean:
	rm -rf build dist/* *.egg-info
