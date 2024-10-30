import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the project root to sys.path
sys.path.insert(0, project_root)
