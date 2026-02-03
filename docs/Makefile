# Minimal makefile for Sphinx documentation

SPHINXBUILD   ?= uv run python -m sphinx
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean

help:
	@echo "  html       Build HTML documentation"
	@echo "  clean      Remove build directory"

html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
