# ---------------------------------------------------------------------------
# Three-method molecule generation - container and training workflow.
#
#   make init            configure .env, create the four host dirs, build the image
#   make in              interactive shell inside the container
#   make exec ARGS="..." run a single command in the container
#   make train MODEL=edm torchrun on all visible GPUs (MODEL = geobfn|edm|llm)
#   make train-all       train the three methods one after the other
#   make eval MODEL=edm CKPT=<checkpoint>
#   make all             train + evaluate all three methods
#   make smoke           tiny end-to-end check of all three (CPU, no QM9)
#   make cv MODEL=edm    cross-validation to pick the hyper-parameters
#   make samples MODEL=edm  10 random samples + the markdown analysis
#   make exp             every experiment, end to end (run_experiments.sh)
#   make test            unit checks (math, tokeniser, cross-references)
#   make preprocess      build dataset/qm9/{train,valid,test}.{npz,smi}
#   make down            stop and remove the container
#
# On Windows, run make from WSL or Git Bash (the Makefile calls docker compose).
# Everything below is also available as ./docker.py <command> if make is missing.
# ---------------------------------------------------------------------------

SHELL   := /bin/bash
PY      ?= python3
DOCKER  ?= $(PY) docker.py

MODEL   ?= geobfn
GPUS    ?=
CKPT    ?=
ARGS    ?=

.DEFAULT_GOAL := help
.PHONY: help init build in exec up down status logs train train-all eval cv samples exp all smoke test preprocess doctor clean

help:  ## show this help
	@echo "targets:"
	@echo "  init                 configure .env + create dirs + build image"
	@echo "  in                   shell inside the container"
	@echo "  train MODEL=edm      train one method (GPUS=n to pin the GPU count)"
	@echo "  train-all            train geobfn, edm and llm in sequence"
	@echo "  eval MODEL=edm CKPT=path/to/best.pt"
	@echo "  all                  train + evaluate all three methods"
	@echo "  smoke                end-to-end check on synthetic data"
	@echo "  test                 unit checks"
	@echo "  preprocess           build the dataset artefacts"
	@echo "  build/up/down/status/logs/doctor"
	@echo ""
	@echo "extra Hydra overrides: make train MODEL=edm ARGS='data.batch_size=32'"

init:  ## configure .env, create the four host directories, build the image
	$(DOCKER) init

build:  ## rebuild the image without touching .env
	$(DOCKER) build

in:  ## interactive shell in the container
	$(DOCKER) in

exec:  ## one command in the container: make exec ARGS="python train.py --help"
	$(DOCKER) exec $(ARGS)

up:  ## start the container in the background
	$(DOCKER) up

down:  ## stop and remove the container
	$(DOCKER) down

status:  ## container status + configured paths
	$(DOCKER) status

logs:  ## follow the container logs
	$(DOCKER) logs

doctor:  ## check docker / GPU / configured paths
	$(DOCKER) doctor

train:  ## train one method: make train MODEL=geobfn [GPUS=8] [ARGS='key=value']
	$(DOCKER) train --model $(MODEL) $(if $(GPUS),--gpus $(GPUS),) $(if $(ARGS),-- $(ARGS),)

train-all:  ## train the three methods in sequence
	for m in geobfn edm llm; do \
		$(DOCKER) train --model $$m $(if $(GPUS),--gpus $(GPUS),) $(if $(ARGS),-- $(ARGS),) || exit 1; \
	done

eval:  ## evaluate a checkpoint: make eval MODEL=edm CKPT=/workspace/logs/.../best.pt
	@test -n "$(CKPT)" || (echo "usage: make eval MODEL=edm CKPT=<checkpoint>" && exit 1)
	$(DOCKER) eval --model $(MODEL) --ckpt $(CKPT) $(if $(ARGS),-- $(ARGS),)

cv:  ## cross-validation: make cv MODEL=geobfn [ARGS='train.lr=3e-4']
	$(DOCKER) exec bash scripts/cv/cv_$(MODEL).sh $(if $(ARGS),$(ARGS),)

samples:  ## 10 random samples + analysis: make samples MODEL=geobfn
	$(DOCKER) exec bash scripts/samples/sample_$(MODEL).sh $(if $(ARGS),$(ARGS),)

exp:  ## every experiment end to end: make exp [ARGS="--steps 2000"]
	$(DOCKER) exec bash run_experiments.sh $(if $(ARGS),$(ARGS),)

all:  ## train + evaluate all three methods
	$(DOCKER) all $(if $(GPUS),--gpus $(GPUS),)

smoke:  ## 1-minute end-to-end check (synthetic data, tiny model)
	$(DOCKER) smoke

test:  ## unit checks that need neither GPU nor data
	$(DOCKER) test

preprocess:  ## build dataset/qm9 artefacts from raw QM9
	$(DOCKER) preprocess $(if $(ARGS),-- $(ARGS),)

clean:  ## remove the local logs/ and container caches (keeps data/)
	rm -rf logs .container
