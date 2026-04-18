"""Root conftest.py — pytest configuration root marker.

``pyproject.toml`` sets ``pythonpath = ["src"]`` so that package imports such
as ``from scraping.extract_shot_data import ...`` work without any manual
``sys.path`` manipulation here.
"""
