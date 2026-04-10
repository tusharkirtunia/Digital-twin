# target_metadata = None        ← find this line, replace with:
from app.models.user import Base
target_metadata = Base.metadata

