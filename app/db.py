import psycopg2
from config.settings import DB_CONFIG

def get_connection():
    return psycopg2.connect(**DB_CONFIG)
# This function establishes a connection to the PostgreSQL database using the provided configuration settings.

def fetch_job_descriptions(limit=10000):
    with get_connection() as conn:
        with conn.cursor() as curs:
            curs.execute(""" SELECT id, title, description FROM dataset
                         WHERE description IS NOT NULL """, (limit,))
            
            return curs.fetchall()

