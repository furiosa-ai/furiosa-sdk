SHELL := /bin/bash

MODULES := litmus quantizer serving server models registry sdk tools cli runtime common

uninstall:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		pip uninstall -y furiosa-$${MODULE}; \
	done

install:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for (( idx=$${#MODULES[@]}-1 ; idx>=0 ; idx-- )) ; do \
		cd furiosa-$${MODULES[idx]}; pip install -e . ; cd .. ; \
	done

lint:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		$(MAKE) -C furiosa-$${MODULE} lint ; \
	done

apply-lint:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		cd furiosa-$${MODULE}; isort .; black . ; cd .. ; \
	done

test:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		$(MAKE) -C furiosa-$${MODULE} test ; \
	done

build:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		$(MAKE) -C furiosa-$${MODULE} build ; \
	done

clean:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		$(MAKE) -C furiosa-$${MODULE} clean ; \
	done

set-version:
ifndef SDK_VERSION
	$(error "SDK_VERSION is not set")
endif
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		sed -i='' "s/version = \"[^\"]*\"/version = \"${SDK_VERSION}\"/" furiosa-$${MODULE}/pyproject.toml ; \
	done
