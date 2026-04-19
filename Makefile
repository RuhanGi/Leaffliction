GREEN   := \033[0;32m
RED     := \033[0;31m
BLUE    := \033[0;34m
YELLOW  := \033[0;33m
RESET   := \033[0m

DATASET = images

PKGS = matplotlib tensorflow cv2

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
	-python -m flake8 src

d: f
	python src/Part_1/Distribution.py $(DATASET)/train
	python src/Part_1/Distribution.py $(DATASET)/val

a: f
	python src/Part_2/Augmentation.py $(DATASET)/train/Apple_Healthy/image\ (1).JPG $(DATASET)/train/Apple_Healthy/image\ (3).JPG $(DATASET)/train/Apple_Healthy/image\ (4).JPG

s: f
	python src/Part_3/Transformation.py -src $(DATASET) -dst ./masked -tsf mask

t: f
	python src/Part_3/Transformation.py -src $(DATASET)/train/Grape_Spot

train:
	python src/Part_4/train.py masked

val:
	python src/Part_4/predict.py -src masked/val

# do wildcard in Linux
v:
	python src/Part_4/predict.py "$(DATASET)/val/Apple_Scab/image (2).JPG" "$(DATASET)/val/Apple_Scab/image (14).JPG" "$(DATASET)/val/Grape_Spot/image (2).JPG"

clean:

fclean: clean

gpush: fclean
	git add .
	git commit -m "final checks"
	git push

re: fclean all
