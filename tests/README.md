# Test Suite Documentation

This directory contains unit tests verifying data integrity, model shapes, and configuration validity.

---

## Running the Tests
```bash
# Run all tests
python -m unittest discover tests/

# Run individual test modules
python -m unittest tests/test_model_shapes.py
python -m unittest tests/test_data_loading.py
python -m unittest tests/test_preprocessing.py
```
