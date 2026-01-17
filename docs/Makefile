# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Tell Makefile that these targets are not files.
.PHONY: help Makefile apidoc

# Generate API documentation from Python source code
# -f: force overwrite
# -e: put each module file in its own page (no single big file)
# -M: put module documentation before submodule documentation
# -o: output directory (source/reference)
apidoc:
	sphinx-apidoc -f -e -M -o source/reference ../src/pytrebuchet

# Build HTML documentation
# Overrides the default target to run apidoc first
html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
