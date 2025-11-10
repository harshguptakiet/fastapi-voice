Dependencies
- Python 3.10+
- MySQL running locally or accessible via network
- Install Python deps:

```bash
pip install -r requirements.txt
```

Setup
- Copy .env.example to .env and update DATABASE_URL for your MySQL instance
- (Optional) Create venv: python3 -m venv .venv && source .venv/bin/activate
- Install deps: pip install -r requirements.txt


How to create venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dev server start

```bash
uvicorn app.main:app --reload
```

Prod server start

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```


Migrations for production (Alembic)

Auto table creation runs only in dev. For production, use Alembic migrations:

1) Install Alembic (once)
```bash
pip install alembic
```

2) Initialize Alembic in the project (creates alembic/ folder)
```bash
alembic init alembic
```

3) Configure Alembic
- In alembic.ini, set sqlalchemy.url to your DATABASE_URL or read from .env in env.py
- In alembic/env.py, set target_metadata to app.core.database.Base.metadata

Example env.py edits:
```python
from app.core.database import Base
target_metadata = Base.metadata
```

4) Generate a migration from current models
```bash
alembic revision --autogenerate -m "create tables"
```

5) Apply migrations
```bash
alembic upgrade head
```

Run these steps in your CI/CD or before starting the prod app to ensure schema is up to date.

