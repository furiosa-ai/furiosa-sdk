default: install 

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

lint:
	isort --check --diff . && black --check --diff . && pylint --verbose furiosa/ tests/

upload_test:
	flit publish --repository testpypi

upload:
	flit publish

test:
	python -m unittest discover -v -s ./tests

doc:
	pdoc3 --html --force --output-dir html_output furiosa	

clean:
	rm -rf build dist/* *.egg-info
