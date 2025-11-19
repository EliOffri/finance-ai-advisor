import psycopg
import os
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return psycopg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        dbname=os.getenv("DB_NAME", "finance_ai"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", None),
    )
