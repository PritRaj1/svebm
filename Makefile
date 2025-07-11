.PHONY: install clean test dev train format lint logs clear-logs

ENV_NAME = SV-EBM
CONDA_BASE := $(shell conda info --base 2>/dev/null || echo "")
CONDA_ACTIVATE := $(shell if [ -f "$(CONDA_BASE)/etc/profile.d/conda.sh" ]; then echo "$(CONDA_BASE)/etc/profile.d/conda.sh"; elif [ -f "$(CONDA_BASE)/Scripts/activate" ]; then echo "$(CONDA_BASE)/Scripts/activate"; else echo ""; fi)

help:
	@echo "Available targets:"
	@echo "  install  - Set up conda environment and install dependencies"
	@echo "  clean    - Remove conda environment"
	@echo "  test     - Run tests in tmux session with logging"
	@echo "  logs     - View latest test log"
	@echo "  clear-logs - Remove all log files"
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
	@mkdir -p logs
	@tmux kill-session -t svebm_test 2>/dev/null || true
	@tmux new-session -d -s svebm_test -n testing
	@tmux send-keys -t svebm_test:testing "if [ -f '$(CONDA_ACTIVATE)' ]; then . '$(CONDA_ACTIVATE)' && conda activate $(ENV_NAME) && python -m pytest tests/ -v 2>&1 | tee logs/pytest_$(shell date +%Y%m%d_%H%M%S).log; else conda activate $(ENV_NAME) && python -m pytest tests/ -v 2>&1 | tee logs/pytest_$(shell date +%Y%m%d_%H%M%S).log; fi" Enter
	@echo "Test session started in tmux. Attach with: tmux attach-session -t svebm_test"
	@echo "Log file: logs/pytest_$(shell date +%Y%m%d_%H%M%S).log"

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

logs:
	@if [ -d "logs" ] && [ -n "$$(ls -A logs 2>/dev/null)" ]; then \
		echo "Latest test log:"; \
		ls -t logs/pytest_*.log 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "No test logs found"; \
	else \
		echo "No logs directory or no log files found"; \
	fi

clear-logs:
	@if [ -d "logs" ]; then \
		echo "Removing all log files..."; \
		rm -rf logs/*; \
		echo "Logs cleared."; \
	else \
		echo "No logs directory found."; \
	fi
