.PHONY: install clean help

help:
	@echo "Available targets:"
	@echo "  install  - Run the init script to set up conda environment and install requirements"
	@echo "  clean    - Remove the conda environment"
	@echo "  help     - Show this help message"

install:
	@echo "Running installation script..."
	@chmod +x scripts/init.sh
	@./scripts/init.sh

clean:
	@echo "Removing conda environment 'SV-EBM'..."
	@conda env remove -n SV-EBM -y 2>/dev/null || echo "Environment 'SV-EBM' not found or already removed"
	@echo "Cleanup completed"
