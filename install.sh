python3.9 -m venv python_venvs/NestedAE
source python_venvs/NestedAE/bin/activate
pip install -U build
python -m build --wheel
pip install .
pytest