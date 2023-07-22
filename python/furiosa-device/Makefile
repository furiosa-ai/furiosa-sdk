default: install

install:
	python -m pip install --editable .

lint:
	isort --check --diff . && black --check --diff .

upload:
	flit publish

test:
	pytest --capture=no --cov=furiosa --cov-report=xml:coverage.xml

clean:
	rm -rf build dist/* *.egg-info