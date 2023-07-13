default: install

install:
	python -m pip install --editable .

lint:
	isort --check --diff . && black --check --diff . && ruff check .

upload:
	flit publish

test:
	pytest --capture=no --cov=furiosa --cov-report=xml:coverage.xml

doc:
	pdoc3 --html --force --output-dir html_output furiosa	

clean:
	rm -rf build dist/* *.egg-info
