.PHONY: docs clean

check-docs-env:
ifndef DOCS_PATH
	$(error "DOCS_PATH is not set")
endif
ifndef DOCS_VERSION
	$(error "DOCS_VERSION is not set")
endif

docs-ko:
	make -C docs/ko html

docs-en:
	make -C docs/en html

deploy-ko: check-docs-env
	mkdir -p ${DOCS_PATH}/${DOCS_VERSION}/ko;
	cp -a docs/ko/build/html/* ${DOCS_PATH}/${DOCS_VERSION}/ko

deploy-en: check-docs-env
	mkdir -p ${DOCS_PATH}/${DOCS_VERSION}/en;
	cp -a docs/en/build/html/* ${DOCS_PATH}/${DOCS_VERSION}/en

docs-clean:
	make -C docs/ko clean

