# ============================================================
# NOTE all the modules installed are compiled binaries in C
# Where this modules can be found in the .venv/bin folder
# for example: if we want to run the vine_worker we can use
# $ ./.venv/bin/vine_worker localhost 9123
# ============================================================

DIR ?= taskvine-ddp-training
PROGRAM ?= main.py
OUTPUT ?= package.tar.gz
	
run:
	conda run -p ${PWD}/.venv python $(PROGRAM)
	
run-worker:
	${PWD}/.venv/bin/vine_worker localhost 9123

create-venv:
	@conda create -p ${PWD}/.venv python=3.13
	@conda install -p ${PWD}/.venv -c conda-forge ndcctools graphviz
	
activate:
	source /home/lax/miniconda3/etc/profile.d/conda.sh && conda activate ${PWD}/.venv
	
build-poncho: # make build-poncho DIR=taskvine-ddp-training PROGRAM=resnet.py OUTPUT=package.tar.gz
	source ${HOME}/miniconda3/etc/profile.d/conda.sh && conda activate ${PWD}/.venv && ${PWD}/.venv/bin/poncho_package_analyze ${PWD}/$(DIR)/$(PROGRAM) ${PWD}/$(DIR)/package.json
	source ${HOME}/miniconda3/etc/profile.d/conda.sh && conda activate ${PWD}/.venv && ${PWD}/.venv/bin/poncho_package_create ${PWD}/$(DIR)/package.json ${PWD}/$(DIR)/$(OUTPUT)

clean:
	@rm -rf *.o *.so *.pyc

.PHONY: clean venv
