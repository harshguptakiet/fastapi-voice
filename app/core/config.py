import os

from dotenv import load_dotenv


# Load environment variables from .env if present
load_dotenv()


# Minimal app configuration
APP_NAME = os.getenv("APP_NAME", "Bot Backend")
ENV = os.getenv("ENV", "dev")

# Use PyMySQL driver for easy install; update credentials as needed
# Example: mysql+pymysql://user:password@localhost:3306/botdb
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Invalid database connection string")
