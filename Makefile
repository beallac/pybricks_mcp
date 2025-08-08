# Simple project automation

PYTHON ?= python3
VENV_DIR := .venv
PIP := $(VENV_DIR)/bin/pip

.PHONY: setup clean

# Create a local virtual environment and install dependencies
setup: $(VENV_DIR)/bin/activate
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	@echo "\n✔ Environment ready. Activate it with:"
	@echo "   source .venv/bin/activate\n"

# Rule that ensures the venv exists
$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)

# Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)
