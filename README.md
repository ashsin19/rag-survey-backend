Python Backend for AI-Powered Insights Platform
This Python backend is built using FastAPI and supports a Retrieval-Augmented Generation (RAG) system for analyzing and comparing PDF-based reports. The backend also integrates with Google Cloud services for vector storage, OCR processing, and secure environment variable management via Google Cloud Secret Manager.

Table of Contents
1. Features
2. Technology Stack
3. Setup Instructions
4. Prerequisites
5. Environment Variables
6. Installation
7. Running the Server
8. API Endpoints
9. Cloud Integration
10. Testing
11. Deployment
12. Future Improvements

FEATURES

PDF Upload and Text Extraction: Supports both direct text-based PDFs and scanned PDFs using OCR (Tesseract or Google Cloud Vision).
Vector Search: Uses FAISS for document similarity search and retrieval.
Summarization and Q&A: Summarizes documents and answers queries using OpenAI's GPT models.
Comparison of Reports: Extracts common insights and unique content between two uploaded reports.
Google Cloud Integration:
Google Cloud Storage for storing PDFs and vector stores.
Google Cloud Secret Manager for secure management of sensitive keys.
JWT Authentication for secure access to the backend.
Scalable Architecture: Deployable on Google Cloud Run.

TECHNOLOGY STACK

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

SETUP INSTRUMENTS

Prerequisites
Python 3.12+ installed.
Docker installed (optional for deployment).
Google Cloud SDK installed and configured.
Environment Variables
You need to set the following environment variables. These can be managed using Google Cloud Secret Manager or .env (for local development):

Variable Name	Description
OPENAI_KEY	OpenAI API Key
SQLITE_KEY	Key for SQLite encryption (if applicable)
SECRET_KEY	JWT secret key for authentication
ALGORITHM	Algorithm for JWT (e.g., HS256)


