.PHONY: ko en clean

check-docs-env:
ifndef DOCS_PATH
	$(error "DOCS_PATH is not set")
endif
ifndef DOCS_VERSION
	$(error "DOCS_VERSION is not set")
endif

ko:
	make -C ko html

python-apidoc:
	cd en/source/api && ./python_apidoc.sh

en: python-apidoc
	make -C en html

html: ko en

deploy-ko: check-docs-env
	mkdir -p ${DOCS_PATH}/${DOCS_VERSION}/ko;
	cp -a ko/build/html/* ${DOCS_PATH}/${DOCS_VERSION}/ko

deploy-en: check-docs-env
	mkdir -p ${DOCS_PATH}/${DOCS_VERSION}/en;
	cp -a en/build/html/* ${DOCS_PATH}/${DOCS_VERSION}/en

deploy: deploy-ko deploy-en

clean:
	make -C ko clean
	make -C en clean

