import pytest
import os
from dotenv import load_dotenv


load_dotenv(dotenv_path=os.path.join(os.getcwd(), "../src/.env")) 
VECTORSTORE_PATH = os.getenv("VECTOR_DB_PATH")
print(VECTORSTORE_PATH)

@pytest.fixture(scope="module", autouse=True)
def setup_vectorstore():
    """Fixture to create and clean up a test vectorstore directory."""
    if not os.path.exists(VECTORSTORE_PATH):
        os.makedirs(VECTORSTORE_PATH)
    
    # Create mock files for testing
    with open(os.path.join(VECTORSTORE_PATH, "test_vectorstore_1"), "w") as f:
        f.write("Mock vectorstore 1")
    with open(os.path.join(VECTORSTORE_PATH, "test_vectorstore_2"), "w") as f:
        f.write("Mock vectorstore 2")
    
    yield  # Test execution happens here

    # Cleanup after tests
    os.remove(os.path.join(VECTORSTORE_PATH, "test_vectorstore_1"))
    os.remove(os.path.join(VECTORSTORE_PATH, "test_vectorstore_2"))

def test_vectorstore_files_exist():
    """Test that mock files were created by the fixture."""
    files = os.listdir(VECTORSTORE_PATH)
    assert "test_vectorstore_1" in files
    assert "test_vectorstore_2" in files