GUILE = guile
GUILD = guild
GUILE_WARNINGS = -Wunbound-variable -Warity-mismatch -Wformat

SOURCES = $(shell find src -name "*.scm")
COMPILED = $(SOURCES:%.scm=%.go)

.PHONY: all compile test clean setup

all: compile

setup:
	@echo "Setting up project structure..."
	@mkdir -p src/{tokenizer,model,generation,inference,training,evaluation}
	@mkdir -p examples tests docs benchmarks data
	@echo "Setup complete!"

compile: $(COMPILED)

%.go: %.scm
	$(GUILD) compile $(GUILE_WARNINGS) -o $@ $

test:
	$(GUILE) run-tests.scm

clean:
	find . -name "*.go" -delete
	find . -name "*~" -delete

run-example:
	$(GUILE) -L src examples/basic-generation.scm
