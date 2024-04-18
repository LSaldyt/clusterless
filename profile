#!/bin/bash
poetry run python -m cProfile -o profile.out main.py $@ 
poetry run python -m tuna profile.out
