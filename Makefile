.PHONY: crash-test
crash-test:
	python -m crash_test.run_all

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  crash-test    Run the complete crash test harness"
