default: install

install:
	python -m pip install --editable .

lint:
	isort --check --diff . && black --check --diff .

upload:
	flit publish

test:
	pytest --capture=no

clean:
	rm -rf build dist/* *.egg-info