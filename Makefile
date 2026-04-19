GREEN   := \033[0;32m
RED     := \033[0;31m
BLUE    := \033[0;34m
YELLOW  := \033[0;33m
RESET   := \033[0m

DATASET = images

PKGS = matplotlib tensorflow cv2

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
	python src/Distribution.py $(DATASET)/train
	python src/Distribution.py $(DATASET)/val

d1: f
	python src/Distribution.py og_images

a: f
	python src/Augmentation.py $(DATASET)/train/Apple_Healthy/image\ (1).JPG $(DATASET)/train/Apple_Healthy/image\ (3).JPG $(DATASET)/train/Apple_Healthy/image\ (4).JPG

s: f
	python src/Transformation.py -src $(DATASET) -dst ./masked -tsf mask

t: f
	python src/Transformation.py -src $(DATASET)/train/Grape_Spot

train:
	python src/train.py masked

val:
	python src/predict.py -src masked/val

# do wildcard in Linux
v:
	python src/predict.py "$(DATASET)/val/Apple_Scab/image (2).JPG" "$(DATASET)/val/Apple_Scab/image (14).JPG" "$(DATASET)/val/Grape_Spot/image (2).JPG"

eval1:
	python src/predict.py Unit_test1/Apple_Black_rot1.JPG Unit_test1/Apple_healthy1.JPG Unit_test1/Apple_healthy2.JPG Unit_test1/Apple_rust.JPG Unit_test1/Apple_scab.JPG

eval2:
	python src/predict.py Unit_test2/Grape_Black_rot1.JPG Unit_test2/Grape_Black_rot2.JPG Unit_test2/Grape_Esca.JPG Unit_test2/Grape_healthy.JPG Unit_test2/Grape_spot.JPG

clean:

fclean: clean

gpush: fclean
	git add .
	git commit -m "final"
	git push

re: fclean all
