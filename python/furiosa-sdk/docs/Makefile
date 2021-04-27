# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
ALLSPHINXOPTS = "$(BUILDDIR)"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

doc:
	mkdir -p "$(BUILDDIR)"/furiosa
	cp -a ../../furiosa-sdk-runtime/furiosa_sdk_runtime $(BUILDDIR)/furiosa/runtime
	cp -a ../../furiosa-sdk-quantizer/furiosa_sdk_quantizer $(BUILDDIR)/furiosa/quantizer
	cp -a ../furiosa $(BUILDDIR)/
	@$(SPHINXBUILD) -c ./ -b html . $(BUILDDIR)/html 

clean:
	rm -rf $(BUILDDIR)