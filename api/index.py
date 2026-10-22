import sys
from pathlib import Path

# Add root directory to sys.path so modules can be resolved on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.api.main import app
