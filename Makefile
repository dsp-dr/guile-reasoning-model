# Reasoning Model Project Makefile

# Guile variables
GUILE = guile
GUILD = guild
GUILE_WARNINGS = -Wunbound-variable -Warity-mismatch -Wformat

# Python variables
PYTHON = python3
PIP = pip3
VENV = venv
PDF_TOOL = pdf-splitter.py
TEST_PDF = tmp/Build_a_Reasoning_Model_(From_Scratch)_v1_MEAP.pdf

# Source files
SOURCES = $(shell find src -name "*.scm" 2>/dev/null)
COMPILED = $(SOURCES:%.scm=%.go)

.PHONY: all compile test clean setup venv experiments

all: compile

# Guile setup and compilation
setup:
	@echo "Setting up project structure..."
	@mkdir -p src/{tokenizer,model,generation,inference,training,evaluation}
	@mkdir -p examples tests docs benchmarks data experiments
	@echo "Setup complete!"

compile: $(COMPILED)

%.go: %.scm
	$(GUILD) compile $(GUILE_WARNINGS) -o $@ $<

test:
	$(GUILE) run-tests.scm

run-example:
	$(GUILE) -L src examples/basic-generation.scm

# Python environment
venv:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install PyPDF2 click pyyaml tabulate requests

# PDF Processing
pdf-analyze:
	. $(VENV)/bin/activate && $(PYTHON) $(PDF_TOOL) analyze "$(TEST_PDF)"

pdf-toc:
	. $(VENV)/bin/activate && $(PYTHON) $(PDF_TOOL) extract-toc "$(TEST_PDF)"

pdf-split:
	. $(VENV)/bin/activate && $(PYTHON) $(PDF_TOOL) split "$(TEST_PDF)" --output-dir output/

# Experiments
exp01:
	. $(VENV)/bin/activate && cd experiments && $(PYTHON) 01_pdf_splitter_validation.py

exp02:
	. $(VENV)/bin/activate && cd experiments && $(PYTHON) 02_tokenization_exploration.py

exp03:
	. $(VENV)/bin/activate && cd experiments && $(PYTHON) 03_chain_of_thought.py

exp04:
	. $(VENV)/bin/activate && cd experiments && $(PYTHON) 04_ollama_reasoning.py

exp05:
	. $(VENV)/bin/activate && cd experiments && $(PYTHON) 05_evaluation_metrics.py

experiments: venv exp01 exp02 exp03 exp04 exp05

# Clean
clean:
	find . -name "*.go" -delete
	find . -name "*~" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf $(VENV)
	rm -rf output/
	rm -rf experiments/output_*
	rm -rf experiments/*.json

clean-python:
	rm -rf $(VENV)
	rm -rf output/
	rm -rf experiments/output_*
	rm -rf experiments/*.json

# Help
help:
	@echo "Reasoning Model Project"
	@echo "======================="
	@echo ""
	@echo "Guile targets:"
	@echo "  make setup       - Create project structure"
	@echo "  make compile     - Compile Scheme files"
	@echo "  make test        - Run tests"
	@echo "  make run-example - Run example"
	@echo ""
	@echo "Python/PDF targets:"
	@echo "  make venv        - Setup Python environment"
	@echo "  make pdf-analyze - Analyze PDF structure"
	@echo "  make pdf-toc     - Extract table of contents"
	@echo "  make pdf-split   - Split PDF by chapters"
	@echo ""
	@echo "Experiments:"
	@echo "  make exp01       - PDF splitter validation"
	@echo "  make exp02       - Tokenization exploration"
	@echo "  make exp03       - Chain-of-thought prompting"
	@echo "  make exp04       - Ollama integration"
	@echo "  make exp05       - Evaluation metrics"
	@echo "  make experiments - Run all experiments"
	@echo ""
	@echo "  make clean       - Remove all generated files"
	@echo "  make clean-python - Remove Python generated files only"
