# Optional convenience wrappers (Unix / Colab). On Windows without `make`, run the
# underlying commands directly — either `python -m docstring_tuner.<module>` or the
# installed console scripts (dt-data, dt-train, dt-infer, dt-eval, dt-demo).

.PHONY: setup setup-gpu data train infer eval demo test typecheck fmt lint

setup:
	pip install -r requirements-cpu.txt
	pip install -e . --no-deps

setup-gpu:
	pip install -r requirements-gpu.txt
	pip install -e . --no-deps

data:
	python -m docstring_tuner.data

train:
	python -m docstring_tuner.train

infer:
	python -m docstring_tuner.generate

eval:
	python -m docstring_tuner.evaluate

demo:
	python -m docstring_tuner.demo

test:
	pytest

typecheck:
	mypy

fmt:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .
