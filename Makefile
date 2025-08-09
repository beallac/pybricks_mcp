# Simple project automation

# Default: show help when running plain `make`
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV_DIR := .venv
PIP := $(VENV_DIR)/bin/pip

.PHONY: help setup clean

# List available targets (auto-generated from lines ending with ## ...)
help: ## List available make targets and what they do
	@echo "Available targets:" 
	@awk 'BEGIN { FS = ":.*## " } /^[a-zA-Z0-9_.-]+:.*## / { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST) 

# Create a local virtual environment and install dependencies
setup: $(VENV_DIR)/bin/activate ## Create a local virtual environment and install dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "\n✔ Environment ready. Activate it with:"
	@echo "   source .venv/bin/activate\n"

# Rule that ensures the venv exists
$(VENV_DIR)/bin/activate: ## Ensure the virtual environment exists
	$(PYTHON) -m venv $(VENV_DIR)

# Remove the virtual environment
clean: ## Remove the virtual environment
	rm -rf $(VENV_DIR)
