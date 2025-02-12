import os
import openai
from utils import *
import pandas as pd
# server.py

from fastapi import FastAPI, Request, UploadFile
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# Create FastAPI instance
app = FastAPI()
actions = execute_api()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://my-react-app-369543119888.us-central1.run.app","localhost:8080"],  # Allow your React app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.middleware("http")
async def redirect_to_https(request: Request, call_next):
    """Middleware to redirect HTTP requests to HTTPS."""
    if request.url.scheme == "http":
        url = request.url.replace(scheme="https")
        return RedirectResponse(url=str(url))
    return await call_next(request)

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = actions.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = actions.create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload/")
async def upload_report(file: UploadFile = File(...), current_user: str = Depends(actions.verify_token)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_path = os.path.join(actions.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Ensure the file exists before processing
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File save error: File not found")
        loader = PyPDFLoader(file_path)
        try:
            documents = loader.load()
            for doc in documents:
                doc.metadata["filename"] = file.filename
        except Exception:
            documents = []

        extracted_text = "\n".join([doc.page_content for doc in documents])

        # If PyPDFLoader fails or returns empty text, use OCR
        if not extracted_text.strip():
            extracted_text = actions.extract_text_from_images(file_path)

        # if not extracted_text.strip():
        #     raise HTTPException(status_code=500, detail="Failed to extract text from PDF. The document might be encrypted or unreadable.")

        # Process extracted text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(extracted_text)

        if not chunks:
            raise HTTPException(status_code=500, detail="Error processing document: No valid text extracted.")

        vector_store = FAISS.from_texts(chunks, OpenAIEmbeddings(openai_api_key=actions.OPENAI_KEY), metadatas=[{"filename": file.filename}] * len(chunks))
        vector_store.save_local(f"{actions.VECTOR_DB_PATH}/{file.filename}")
        actions.vector_stores[file.filename] = vector_store
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query/")
async def query_report(request: QueryRequest, current_user: str = Depends(actions.verify_token)):
    if not actions.vector_stores:
        raise HTTPException(status_code=400, detail="No reports have been processed yet.")
    
    try:
        all_results = []
        query_prompt = actions.construct_query_prompt(request.query)
        for store in actions.vector_stores.values():
            results = store.similarity_search(query_prompt, k=10)
            ranked_results = actions.get_document_rerank(3,query_prompt,results)
            # all_results.extend([doc.page_content for doc in results])
            all_results.extend(ranked_results)
        
        return {"results": all_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying reports: {str(e)}")

@app.post("/compare/")
async def compare_reports(request: QueryRequest, current_user: str = Depends(actions.verify_token)):
    if len(actions.vector_stores) < 2:
        raise HTTPException(status_code=400, detail="Need to have at least 2 reports to compare.")
    res=actions.generate_compare_reports(request.query,actions.vector_stores)
    return res

@app.get("/reports/")
async def list_reports():
    """List all uploaded reports."""
    try:
        reports = os.listdir(actions.UPLOAD_DIR)
        return {"reports": reports}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.delete("/reports/{filename}")
async def delete_report(filename: str):
    """Delete a report and its associated vectors."""
    file_path = os.path.join(actions.UPLOAD_DIR, filename)
    vector_store_path = os.path.join(actions.VECTOR_DB_PATH, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        # Delete the file
        os.remove(file_path)

        # Delete the associated vector store
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)  # Remove the entire directory for the vector store

        return {"message": f"Report '{filename}' and associated vectors deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting report: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI API"}