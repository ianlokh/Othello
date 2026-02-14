# CLAUDE.md — Othello (Pygame)

## Project Overview

Othello board game built with Python and Pygame. Includes an RL training pipeline (TensorFlow) and a Pygame-based interactive UI.

## Build / Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Pygame UI
python othello_main_pygame.py

# Run the terminal UI
python othello_main.py

# Train the RL agent (optional memory profiling with -m memory_profiler)
python othello/othello_train.py

# Evaluate the agent
python othello_eval.py
```

Python version: **3.11** (see requirements.txt)

## Linting

```bash
# Check style
flake8 --max-line-length 120 .

# Auto-format
black --line-length 120 .

# Sort imports
isort .
```

## Code Style

- Follow **PEP 8** throughout. Max line length is **120** characters.
- Use a **clean game loop**: keep input handling, state update, and rendering in clearly separated phases within the main loop.
- Organise drawable objects with **sprite groups** (`pygame.sprite.Group`) rather than manually iterating loose lists.
- Keep game state out of global variables — pass it through classes or a dedicated state object.
- Use descriptive names for constants (e.g., `CELL_SIZE`, `BOARD_SIZE`) and keep them at the top of the file.
