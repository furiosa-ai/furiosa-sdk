default: install

install:
	MACOSX_DEPLOYMENT_TARGET=10.15 python -m pip install --editable .

build:
	python -m build

lint:
	isort --check --diff . && black --check --diff .

upload_test:
	flit publish --repository testpypi

upload:
	flit publish

test:
	pytest --capture=no

doc:
	pdoc3 --html --force --output-dir html_output furiosa

clean:
	rm -rf build dist/* *.egg-info
