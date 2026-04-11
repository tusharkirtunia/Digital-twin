import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

target_metadata = None       # ← find this line, replace with:
from app.models.user import Base
target_metadata = Base.metadata
