# Python Backend for AI-Powered Insights Platform

This Python backend is built using FastAPI and supports a Retrieval-Augmented Generation (RAG) system for analyzing and comparing PDF-based reports. The backend also integrates with Google Cloud services for vector storage, OCR processing, and secure environment variable management via Google Cloud Secret Manager.

## Table of Contents
1. Features
2. Technology Stack
3. Setup Instructions
    3a. Prerequisites
    3b. Installation
    3c. Running the Server
4. API Endpoints
5. Cloud Integration
6. Testing
7. Deployment
8. Future Improvements


## 1. FEATURES

PDF Upload and Text Extraction: Supports both direct text-based PDFs and scanned PDFs using OCR (Tesseract or Google Cloud Vision).
Vector Search: Uses FAISS for document similarity search and retrieval.
Summarization and Q&A: Summarizes documents and answers queries using OpenAI's GPT models.
Comparison of Reports: Extracts common insights and unique content between two uploaded reports.
Google Cloud Integration:
Google Cloud Storage for storing PDFs and vector stores.
Google Cloud Secret Manager for secure management of sensitive keys.
JWT Authentication for secure access to the backend.
Scalable Architecture: Deployable on Google Cloud Run.

## 2. TECHNOLOGY STACK

Python 3.12
FastAPI – For building the web API.
FAISS – For vector search.
PyPDF2 / PyPDFLoader – For processing PDF files.
Pytesseract / Google Cloud Vision – For OCR processing.
OpenAI GPT – For summarization and Q&A tasks.
Google Cloud Services:
Cloud Storage
Secret Manager
Cloud Run

## 3. SETUP INSTRUMENTS

### 3a. PREREQUISITES

Python 3.12+ installed.

Docker installed (optional for deployment).

Google Cloud SDK installed and configured.

Environment Variables
You need to set the following environment variables. These can be managed using Google Cloud Secret Manager and .env (for non-sensitive variables):

Variable Name	Description
OPENAI_KEY	OpenAI API Key
SQLITE_KEY	Key for SQLite encryption (if applicable)
SECRET_KEY	JWT secret key for authentication
ALGORITHM	Algorithm for JWT (e.g., HS256)

### 3b. INSTALLATION

```
git clone https://github.com/ashsin19/rag-survey-backend.git

cd rag-survey-backend

python -m venv venv

Windows:
venv\Scripts\activate 

Linux:
source venv/bin/activate

pip install --no-cache-dir -r requirements.txt

pip install pytest pytest-asyncio httpx google-cloud-storage google-cloud-vision google-cloud-secret-manager nltk wordcloud matplotlib

python -m nltk.downloader wordnet

cd src
```

### 3c. RUNNING THE SERVER

```
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4. API ENDPOINTS

### Authentication
Method	Endpoint	    Description
POST	/token	        Obtain a JWT access token.

### Reports
Method	Endpoint	    Description
POST	/upload/	    Upload a PDF report.
GET	    /reports/	    List all uploaded reports.
DELETE	/reports/{id}	Delete a report and its vector store.
POST    /stats/         Get dynamic stats of reports processed

### Query and Compare
Method	Endpoint	    Description
POST	/query/	        Query a report and get insights.
POST	/compare/	    Compare two reports and get results.

## 5. CLOUD INTEGRATIONS

### Google Cloud Services

Cloud Storage: Stores PDFs and vector stores (index.faiss and metadata).

Secret Manager: Securely manages sensitive keys (OPENAI_KEY, SECRET_KEY, etc.).

Cloud Run: Deploys the backend as a scalable, containerized service.

Cloud Build: Stream build changes and triggers a new build when production branch in git repo is updated

## 6. TESTING

Tests are written using pytest. The tests are part of cloud build and will be executed before the build is fully deployed if successful

## 7. DEPLOYMENTS

Cloud Build Trigger actively monitors the git repo. The workflow for git repo is MAIN Branch -> Merge Changes to Production Branch with approvals -> Change event streamed to cloud build trigger -> Deployment initialized with cloudbuild.yaml

## 8. FUTURE IMPROVEMENTS

a. Add Sentiment Analysis for uploaded documents. For this feature, we can use BERT-based transformer model. Once the result is shared to the frontend, we can use react-chartjs to visualize the sentiment trends

b. Implement User Role Management for enhanced security.

c. Implement user-based report view. For this feature, we will need to tag the reports to username blobs in cloud storage. In such case, the blob structure will be vectorstore -> username -> filename -> index.faiss and similarly the report blob will follow the structure.