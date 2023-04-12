SHELL := /bin/bash
MODULES := litmus serving server sdk quantizer optimizer tools cli runtime registry common

uninstall:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	for MODULE in "$${MODULES[@]}"; do \
		pip uninstall -y furiosa-$${MODULE}; \
	done

install:
	IFS=' ' read -a MODULES <<< '$(MODULES)'; \
	INSTALL_CMD="pip install --pre"; \
	for MODULE in "$${MODULES[@]}"; do \
		INSTALL_CMD+=" -e furiosa-$${MODULE}"; \
	done; \
	$${INSTALL_CMD}

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
		sed -i.bak "s/version = \"[^\"]*\"/version = \"${SDK_VERSION}\"/" furiosa-$${MODULE}/pyproject.toml ; rm -f furiosa-$${MODULE}/pyproject.toml.bak ; \
	done; \
	sed -i.bak "s/version = \"[^\"]*\"/version = \"${SDK_VERSION}\"/" ../docs/ko/source/conf.py ; rm -f ../docs/ko/source/conf.py.bak ; \
	sed -i.bak "s/release = \"[^\"]*\"/relase = \"${SDK_VERSION}\"/" ../docs/ko/source/conf.py ; rm -f ../docs/ko/source/conf.py.bak ; \
	sed -i.bak "s/version = \"[^\"]*\"/version = \"${SDK_VERSION}\"/" ../docs/en/source/conf.py ; rm -f ../docs/en/source/conf.py.bak ; \
	sed -i.bak "s/release = \"[^\"]*\"/release = \"${SDK_VERSION}\"/" ../docs/en/source/conf.py ; rm -f ../docs/en/source/conf.py.bak ;
