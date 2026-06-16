"""
Thin wrapper kept for backwards compatibility with existing scripts.

Use `python main.py train --cv` or `python cross_validation.py` directly instead.
"""

from cross_validation import main

if __name__ == "__main__":
    main()
