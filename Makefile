GREEN   := \033[0;32m
RED     := \033[0;31m
BLUE    := \033[0;34m
YELLOW  := \033[0;33m
RESET   := \033[0m

DATASET = images

PKGS = matplotlib

export PYTHONPATH := ./src

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
	-python -m flake8 src/Part_1 src/Part_2 src/Part_3

d: f
	python src/Part_1/Distribution.py $(DATASET)/Apple
	python src/Part_1/Distribution.py $(DATASET)/Grape

a: f
	python src/Part_2/Augmentation.py $(DATASET)/Apple/Healthy/image\ (1).JPG $(DATASET)/Apple/Healthy/image\ (2).JPG $(DATASET)/Apple/Healthy/image\ (3).JPG

s: f
	python src/Part_3/Transformation.py -src $(DATASET) -dst ./masked -tsf mask

t: f
	python src/Part_3/Transformation.py -src $(DATASET)/Grape/Spot

train:
	python src/Part_4/train.py masked

clean:


fclean: clean


gpush: fclean
	git add .
	git commit -m "Cleaner"
	git push

re: fclean all
