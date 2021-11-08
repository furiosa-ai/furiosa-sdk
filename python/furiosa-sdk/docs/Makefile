# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
ALLSPHINXOPTS = "$(BUILDDIR)"
FURIOSADOCS   = furiosa

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doc:
	mkdir -p $(BUILDDIR)/src/$(FURIOSADOCS)
	cp -a ../../furiosa-runtime/furiosa/runtime $(BUILDDIR)/src/$(FURIOSADOCS)/runtime
	cp -a ../../furiosa-sdk-quantizer/furiosa/quantizer $(BUILDDIR)/src/$(FURIOSADOCS)/quantizer
	rsync -av --progress ../furiosa $(BUILDDIR)/src --exclude runtime
	sphinx-apidoc -o . $(BUILDDIR)/src $(BUILDDIR)/src/$(FURIOSADOCS)/runtime/errors/*
	make html

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(FURIOSADOCS)
	rm furiosa.*.rst
	rm modules.rst
