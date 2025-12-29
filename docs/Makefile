# Minimal makefile for Sphinx documentation

SPHINXBUILD   ?= uv run python -m sphinx
JUPYTEXT      ?= uv run jupytext
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help html clean notebooks

help:
	@echo "  html       Build HTML documentation"
	@echo "  clean      Remove build directory"
	@echo "  notebooks  Convert .py examples to .ipynb"

notebooks:
	$(JUPYTEXT) --to notebook examples/*.py

html: notebooks
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
