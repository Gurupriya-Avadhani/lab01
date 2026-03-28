# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Create venv and install requirements
setup:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Initializing DVC..."
	$(VENV)/bin/dvc init --no-scm -f
	@echo "Setup complete. Use 'source .venv/bin/activate' to enter the environment."

# Run the boilerplate
run:
	@export PYTHONPATH=$${PYTHONPATH}:$(shell pwd) && $(PYTHON) src/demo/app.py

# A shortcut to see logs
logs:
	cat logs/pipeline.log

# Clean up environment
clean:
	rm -rf $(VENV)
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} +