# ============================================================
# NOTE all the modules installed are compiled binaries in C
# Where this modules can be found in the .venv/bin folder
# for example: if we want to run the vine_worker we can use
# $ ./.venv/bin/vine_worker localhost 9123
# ============================================================

PROGRAM ?= main.py

run:
	conda run -p ${PWD}/.venv python $(PROGRAM)
	
run-worker:
	${PWD}/.venv/bin/vine_worker localhost 9123

create-venv:
	@sudo conda create -p ${PWD}/.venv python=3.13
	@sudo conda install -p ${PWD}/.venv -c conda-forge ndcctools

clean:
	@rm -rf *.o *.so *.pyc

.PHONY: clean venv
