.PHONY: help install uninstall test dev train format lint logs clear-logs

ENV_NAME = SV-EBM

help:
	@echo "Available targets:"
	@echo "  install  - Set up conda environment and install dependencies"
	@echo "  uninstall - Remove conda environment"
	@echo "  test     - Run tests in tmux session with logging"
	@echo "  logs     - View latest test log"
	@echo "  clear-logs - Remove all log files"
	@echo "  dev      - Start development session"
	@echo "  train    - Run sequential training jobs from jobs.txt"
	@echo "  format   - Format code"
	@echo "  lint     - Run linting"
	@echo "  help     - Show this help"

install:
	@chmod +x scripts/init.sh
	@./scripts/init.sh

uninstall:
	@echo "Removing conda environment..."
	@conda env remove -n $(ENV_NAME) -y 2>/dev/null || echo "Environment not found"

test:
	@mkdir -p logs
	@tmux kill-session -t svebm_test 2>/dev/null || true
	@tmux new-session -d -s svebm_test -n testing
	@tmux send-keys -t svebm_test:testing "conda activate $(ENV_NAME) && python -m pytest tests/ -v 2>&1 | tee logs/pytest_$(shell date +%Y%m%d_%H%M%S).log" Enter
	@echo "Test session started in tmux. Attach with: tmux attach-session -t svebm_test"

dev:
	@tmux kill-session -t svebm_dev 2>/dev/null || true
	@tmux new-session -d -s svebm_dev -n main
	@tmux send-keys -t svebm_dev:main "conda activate $(ENV_NAME)" Enter
	@tmux new-window -t svebm_dev -n logs
	@tmux send-keys -t svebm_dev:logs "conda activate $(ENV_NAME) && tail -f logs/*.log" Enter
	@echo "Dev session ready: tmux attach-session -t svebm_dev"

train:
	@chmod +x scripts/train.sh
	@mkdir -p logs
	@tmux kill-session -t svebm_sequential 2>/dev/null || true
	@tmux new-session -d -s svebm_sequential -n runner
	@tmux send-keys -t svebm_sequential:runner "./scripts/train.sh $(if $(JOBS_FILE),$(JOBS_FILE),jobs.txt) 2>&1 | tee logs/sequential_training_$(shell date +%Y%m%d_%H%M%S).log" Enter
	@echo "Training jobs started in tmux session 'svebm_sequential'"
	@echo "Attach with: tmux attach-session -t svebm_sequential"
	@echo "Sequential training log: logs/sequential_training_$(shell date +%Y%m%d_%H%M%S).log"
	@echo "Or monitor with: tmux list-sessions"

format:
	@conda run -n $(ENV_NAME) black src/ tests/

lint:
	@conda run -n $(ENV_NAME) flake8 src/ tests/

logs:
	@if [ -d "logs" ] && [ -n "$$(ls -A logs 2>/dev/null)" ]; then \
		echo "Latest log:"; \
		ls -t logs/*.log 2>/dev/null | head -1 | xargs cat 2>/dev/null || echo "No logs found"; \
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
