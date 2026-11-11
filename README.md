# Algorithmic Game Theory Homework

Assignments for Modern Algorithmic Game Theory (NOPT021) at Charles University. Read the weekly prompts in [tasks](tasks/) and implement your homework in [solutions](solutions/).

## Set up Python

From the repository root, create and activate a virtual environment with Python 3.14, then install the course dependencies.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

In a new terminal, run `source .venv/bin/activate` again before testing.

## Run the tests

From the repository root, run the Week 1 tests.

```sh
python -m pytest tests/test_week01_classic.py tests/test_week01_random.py
```

Replace `01` with the week you want to test. To run all included tests, use `python -m pytest tests`. The repository includes automated tests for Weeks 1–6. Tests for unfinished functions will fail until you implement them.
