import os
import psycopg2
from datetime import datetime

DB_PARAMS = {
    "host": "127.0.0.1",
    "database": "ml_metadata",
    "user": "postgres",
    "password": "MYpostsql^123",
    "port": "5433",
}

DATA_DIR = os.path.join("data", "raw")


def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except Exception as e:
        print(f"Connection Failed: {e}")
        return None


def ingest_data():
    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()

    print("Cleaning old recordsd")
    cursor.execute("TRUNCATE TABLE satellite_data RESTART IDENTITY;")
    conn.commit()

    print(f"Scanning Directory: {DATA_DIR} ...")

    count = 0

    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
                folder_name = os.path.basename(root)
                label = folder_name

                full_path = os.path.join(root, filename)

                insert_query = """
                    INSERT INTO satellite_data
                    (filename, ground_truth_label, source_satellite, capture_date)
                    VALUES (%s, %s, 'Sentinel-2', %s) 
                """

                today = datetime.now().date()

                cursor.execute(insert_query, (full_path, label, today))
                count += 1

                if count % 1000 == 0:
                    print(f"Processed {count} images...")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"\n Success! Ingestion {count} images into PostgreSQL.")


if __name__ == "__main__":
    ingest_data()
