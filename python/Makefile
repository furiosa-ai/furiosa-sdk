SHELL := /bin/bash

uninstall:
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for MODULE in "$${MODULES[@]}"; do \
		pip uninstall -y furiosa-$${MODULE}; \
	done

install:
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for (( idx=$${#MODULES[@]}-1 ; idx>=0 ; idx-- )) ; do \
    cd furiosa-$${MODULES[idx]}; pip install -e . ; cd .. ; \
	done

lint:
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for MODULE in "$${MODULES[@]}"; do \
    $(MAKE) -C furiosa-$${MODULE} lint ; \
	done

apply-lint:
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for MODULE in "$${MODULES[@]}"; do \
    cd furiosa-$${MODULE}; isort .; black . ; cd .. ; \
	done

test:
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for MODULE in "$${MODULES[@]}"; do \
    $(MAKE) -C furiosa-$${MODULE} test ; \
	done

build:
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for MODULE in "$${MODULES[@]}"; do \
    $(MAKE) -C furiosa-$${MODULE} build ; \
	done

clean:
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for MODULE in "$${MODULES[@]}"; do \
    $(MAKE) -C furiosa-$${MODULE} clean ; \
	done

set-version:
ifndef SDK_VERSION
	$(error "SDK_VERSION is not set")
endif
	MODULES=("litmus" "quantizer" "serving" "server" "models" "registry" "sdk" "tools" "cli" "runtime" "common"); \
	for MODULE in "$${MODULES[@]}"; do \
    sed -i "s/version = \"[^\"]*\"/version = \"${SDK_VERSION}\"/" furiosa-$${MODULE}/setup.py ; \
	done