GREEN   := \033[0;32m
RED     := \033[0;31m
BLUE    := \033[0;34m
YELLOW  := \033[0;33m
RESET   := \033[0m

DATASET = images

PKGS = matplotlib

.SILENT:

all: check
	@echo "$(BLUE)  Packages Ready!$(RESET)"
	@echo "$(YELLOW) Usage:\x1B[33m make {a}$(RESET)"

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg > /dev/null 2>&1; \
		fi; \
	done

f:
	-python -m flake8 .

d: f
	python Distribution.py $(DATASET)/Apple
	python Distribution.py $(DATASET)/Grape

a: f
	python Augmentation.py $(DATASET)/Apple/Healthy/image\ (1).JPG

t: f
	python Transformation.py -src $(DATASET)/Grape/Healthy

clean:


fclean: clean


gpush: fclean
	git add .
	git commit -m "Augmented"
	git push

re: fclean all