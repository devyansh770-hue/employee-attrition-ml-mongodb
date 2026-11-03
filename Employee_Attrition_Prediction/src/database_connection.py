"""
database_connection.py
----------------------
Provides a reusable MongoDB connection via PyMongo.
Credentials and connection parameters are loaded from the .env file.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load environment variables from .env in project root
load_dotenv(os.path.join(BASE_DIR, ".env"))

DEFAULT_MONGO_URI = "mongodb://localhost:27017/"
DEFAULT_DB_NAME   = "employee_attrition_db"


def get_env_var(key: str, default: str = None) -> str:
    """Return environment variable or raise if missing and no default."""
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(
            f"Missing required environment variable: '{key}'. "
            "Please check your .env file."
        )
    return value


def get_mongo_client(uri: str = None, timeout_ms: int = 3000) -> MongoClient:
    """
    Create and return a PyMongo MongoClient instance.

    Parameters
    ----------
    uri : str, optional
        MongoDB connection URI. Defaults to MONGO_URI from .env.
    timeout_ms : int
        Server selection timeout in milliseconds.

    Returns
    -------
    MongoClient
    """
    if uri is None:
        uri = get_env_var("MONGO_URI", DEFAULT_MONGO_URI)
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


def get_mongo_db(db_name: str = None):
    """
    Return the MongoDB database instance.

    Parameters
    ----------
    db_name : str, optional
        Database name. Defaults to MONGO_DB_NAME from .env.

    Returns
    -------
    pymongo.database.Database
    """
    if db_name is None:
        db_name = get_env_var("MONGO_DB_NAME", DEFAULT_DB_NAME)
    client = get_mongo_client()
    return client[db_name]


def get_collection(collection_name: str = "employees"):
    """Return a reference to a specific MongoDB collection."""
    db = get_mongo_db()
    return db[collection_name]


def test_connection() -> bool:
    """
    Verify that MongoDB server is reachable and active.

    Returns
    -------
    bool : True on success, False on failure.
    """
    try:
        client = get_mongo_client(timeout_ms=2500)
        # The ismaster command is cheap and does not require auth.
        client.admin.command("ping")
        print("✅ MongoDB connection successful.")
        return True
    except (ServerSelectionTimeoutError, PyMongoError, EnvironmentError, Exception) as exc:
        print(f"❌ MongoDB connection failed: {exc}")
        return False


if __name__ == "__main__":
    test_connection()
