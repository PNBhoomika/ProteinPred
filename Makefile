.PHONY: test app
test:
	python -m pytest -q

app:
	python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
