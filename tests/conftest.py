import os
import pytest
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

@pytest.fixture(scope="session")
def llm():
    """Create a real Ollama LLM instance for testing."""
    return OllamaLLM(model="deepseek-r1:1.5b", temperature=0)

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Load environment variables
    load_dotenv()
    
    # Set test-specific environment variables
    os.environ.update({
        'DB_HOST': 'localhost',
        'DB_PORT': '5432',
        'DB_NAME': 'da_test',
        'DB_USER': 'postgres',
        'DB_PASSWORD': 'postgres'
    })
    
    # Ensure the test database exists and is properly configured
    try:
        import psycopg2
        conn_params = {
            'host': 'localhost',
            'port': '5432',
            'database': 'postgres',  # Connect to default database first
            'user': 'postgres',
            'password': 'postgres'
        }
        
        with psycopg2.connect(**conn_params) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Create test database if it doesn't exist
                cur.execute("SELECT 1 FROM pg_database WHERE datname = 'da_test'")
                if not cur.fetchone():
                    cur.execute("CREATE DATABASE da_test")
                
                # Connect to test database and set up permissions
                conn_params['database'] = 'da_test'
                with psycopg2.connect(**conn_params) as test_conn:
                    with test_conn.cursor() as test_cur:
                        test_cur.execute("GRANT ALL PRIVILEGES ON DATABASE da_test TO postgres")
                        test_cur.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres")
                        test_cur.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres")
    except Exception as e:
        pytest.fail(f"Failed to set up test database: {str(e)}")

@pytest.fixture(scope="session")
def test_db_params():
    """Return test database parameters."""
    return {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT')),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    } 