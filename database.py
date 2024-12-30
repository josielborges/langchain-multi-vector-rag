import os
from dotenv import load_dotenv

load_dotenv()
host = os.getenv("PG_VECTOR_HOST")
user = os.getenv("PG_VECTOR_USER")
password = os.getenv("PG_VECTOR_PASSWORD")
COLLECTION_NAME = os.getenv("PG_DATABASE")

CONNECTION_STRING = f"postgresql+psycopg://{user}:{password}@{host}:5438/{COLLECTION_NAME}"