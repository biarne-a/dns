# -*- mode: makefile -*-
install_mac_m2:
	#conda create --name=dns-m2 python=3.9
	#conda activate mos-m2
	conda install -c apple tensorflow-deps
	pip install tensorflow-macos==2.14 tensorflow-metal
	conda install -c conda-forge --file requirements.in


install:
	conda create --name=dns python=3.9
	conda activate mos
	conda install -c conda-forge tensorflow
	conda install -c conda-forge --file requirements.in


clean:
	./scripts/clean.sh
