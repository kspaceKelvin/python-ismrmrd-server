.SUFFIXES:
# phony runners, but actual sentinel files so only rerun when needed
.PHONY: test format
test: .test
format: .format

SHELL := /usr/bin/bash
PYTHON ?= python3
PYTHON_FILES=$(wildcard *.py tests/*py tests/helpers/*.py)

.ONESHELL: # source in same shell as pytest
.test: $(PYTHON_FILES) single-shot.sh | .venv/
	source .venv/bin/activate
	$(PYTHON) -m pytest tests | tee $@

# don't format all. Would be a big git revision
.format: $(wildcard test/*py tests/helpers/*.py) #$(PYTHON_FILES)
	isort $? | tee $@
	black $? | tee -a $@

# if we don't have .venv, make it and install this package
# use 'uv' if we have it
.ONESHELL:
.venv/:
	if command -v uv; then
		uv venv .venv;
		uv pip install -e .[dev];
	else
	  $(PYTHON) -m venv .venv/;
	  $(PYTHON) -m pip install -e  .[dev];
	fi
