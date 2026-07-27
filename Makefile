# ─────────────────────────────────────────────────────────────────────────────
# Makefile — antisaccade modeling environment
#
# make build              build the image (run once, or when requirements change)
# make run SCRIPT=train.py            run a script
# make run SCRIPT=convert_papers.py   run a different script
# make gpu SCRIPT=train.py            same but with GPU passthrough
# make shell                          interactive bash session
# ─────────────────────────────────────────────────────────────────────────────

IMAGE   = antisaccade
CACHE   = $(PWD)/.cache
WORKDIR = /workspace
SCRIPT  ?= train.py

# Volume flags shared across targets
VOLS = -v $(PWD):/workspace -v $(CACHE):/workspace/.cache

.PHONY: build run gpu shell

build:
	docker build -t $(IMAGE) .

run: build
	docker run --rm $(VOLS) $(IMAGE) $(SCRIPT)

gpu: build
	docker run --rm --gpus all $(VOLS) $(IMAGE) $(SCRIPT)

shell: build
	docker run --rm -it $(VOLS) --entrypoint bash $(IMAGE)
