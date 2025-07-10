.PHONY: install clean test dev train format lint

ENV_NAME = SV-EBM
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo "")
CONDA_ACTIVATE := $(shell if [ -f "$(CONDA_BASE)/etc/profile.d/conda.sh" ]; then echo "$(CONDA_BASE)/etc/profile.d/conda.sh"; elif [ -f "$(CONDA_BASE)/Scripts/activate" ]; then echo "$(CONDA_BASE)/Scripts/activate"; else echo ""; fi)

help:
	@echo "Available targets:"
	@echo "  install  - Set up conda environment and install dependencies"
	@echo "  clean    - Remove conda environment"
	@echo "  test     - Run tests"
	@echo "  dev      - Start development session"
	@echo "  train    - Start training session"
	@echo "  format   - Format code"
	@echo "  lint     - Run linting"
	@echo "  help     - Show this help"

install:
	@chmod +x scripts/init.sh
	@./scripts/init.sh

clean:
	@echo "Removing conda environment..."
	@conda env remove -n $(ENV_NAME) -y 2>/dev/null || echo "Environment not found"

define conda_run
	@if [ -n "$(CONDA_ACTIVATE)" ]; then \
		. "$(CONDA_ACTIVATE)" && conda activate $(ENV_NAME) && $(1); \
	else \
		echo "Warning: Could not find conda activation script. Trying direct activation..."; \
		conda activate $(ENV_NAME) && $(1); \
	fi
endef

test:
	$(call conda_run,python -m pytest tests/ -v)

dev:
	@tmux kill-session -t svebm_dev 2>/dev/null || true
	@tmux new-session -d -s svebm_dev -n main
	@tmux send-keys -t svebm_dev:main "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME); else conda activate $(ENV_NAME); fi" Enter
	@tmux new-window -t svebm_dev -n logs
	@tmux send-keys -t svebm_dev:logs "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && tail -f logs/*.log; else conda activate $(ENV_NAME) && tail -f logs/*.log; fi" Enter
	@echo "Dev session ready: tmux attach-session -t svebm_dev"

train:
	@tmux kill-session -t svebm_train 2>/dev/null || true
	@tmux new-session -d -s svebm_train -n training
	@tmux send-keys -t svebm_train:training "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && python main.py; else conda activate $(ENV_NAME) && python main.py; fi" Enter
	@echo "Training session ready: tmux attach-session -t svebm_train"

format:
	$(call conda_run,black src/ tests/)

lint:
	$(call conda_run,flake8 src/ tests/)
