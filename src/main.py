import os
from langchain.llms import OpenAI
from utils import *
import pandas as pd
# server.py
from langchain.schema import Document
from langchain.chains import RetrievalQA, load_summarize_chain
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

@app.on_event("startup")
async def startup_event():
    filenames = actions.list_filenames_in_gcs()
    if not filenames:
        print("No vector stores found in GCS. Skip loading.")
        return
    for filename in filenames:
        try:
            local_vectorstore_path = actions.download_vectorstore_from_gcs(filename)
            vector_store = FAISS.load_local(
                local_vectorstore_path, 
                OpenAIEmbeddings(openai_api_key=actions.OPENAI_KEY),
                allow_dangerous_deserialization=True  # In final production version, this will require an additional step to compute and match SHA256 hash values. 
                                                      #To do that we need to securely store the hash values
                                                      # We can also configure GCS to prevent unauthorized access
            )
            actions.vector_stores[filename] = vector_store
            print(f"Vector Store Loaded for {filename}")        
        except Exception as e:
            print(f"Failed to load vector store for {filename}: {e}")


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
    actions.upload_pdf_to_gcs(file_path,file.filename,actions.UPLOAD_DIR)
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
        actions.upload_vectorstore_to_gcs(actions.VECTOR_DB_PATH,actions.VECTOR_DB_PATH)
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
        llm = OpenAI(openai_api_key=actions.OPENAI_KEY, temperature=0.3)
        for store_name, store in actions.vector_stores.items():
            results = store.similarity_search(query_prompt, k=4)
            ranked_results = actions.get_document_rerank(3,request.query,results)
            all_results.extend(ranked_results)
            print(f" Ranked results: {ranked_results}")
            top_documents = [doc for doc in ranked_results]
            if top_documents:
                documents = [Document(page_content=doc) for doc in top_documents]
                summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = summarize_chain.run(documents)
                print(f"Summary for {store_name}: {summary}")
            retriever = store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa_chain.run(request.query)
        return {"summary": summary, "answer": answer, "documents": [doc for doc in all_results]}
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
        bucket = actions.storage_client.bucket(actions.BUCKET_NAME)
        blobs = bucket.list_blobs(prefix=actions.UPLOAD_DIR)
        reports = set()
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) > 1:  # Ensure it is not the root folder
                report_name = parts[-1]
                if report_name:  # Avoid empty strings
                    reports.add(report_name)
        
        return {"reports": list(reports)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.delete("/reports/{filename}")
async def delete_report(filename: str):
    """Delete a report and its associated vectors."""
    
    bucket = actions.storage_client.bucket(actions.BUCKET_NAME)
    
    try:
        # Delete the report from uploaded_reports
        report_blob = bucket.blob(f"{actions.UPLOAD_DIR}/{filename}")
        if not report_blob.exists():
            raise HTTPException(status_code=404, detail=f"Report '{filename}' not found in GCS.")
        
        report_blob.delete()
        print(f"Deleted report: {actions.UPLOAD_DIR}/{filename}")

        # Delete all associated vector store files for the report
        blobs_to_delete = bucket.list_blobs(prefix=f"{actions.VECTOR_DB_PATH}/{filename}")
        vector_store_deleted = False
        for blob in blobs_to_delete:
            blob.delete()
            vector_store_deleted = True
            print(f"Deleted vector store file: {blob.name}")
        
        if not vector_store_deleted:
            print(f"No vector store found for '{filename}'.")
        removed_vector_store = actions.vector_stores.pop(filename, None)
        if removed_vector_store:
            print(f"Removed '{filename}' from vector_stores.")
        else:
            print(f"'{filename}' not found in vector_stores.")
        return {"message": f"Report '{filename}' and associated vectors deleted successfully from GCS."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting report: {str(e)}")

@app.get("/list-vectorstores")
def list_vectorstores():
    vector_store_path = os.path.join(actions.VECTOR_DB_PATH)
    if not os.path.exists(vector_store_path):
        return {"message": "Vectorstore directory not found."}

    # List all subdirectories in the vectorstore directory
    vectorstores = [
        directory for directory in os.listdir(vector_store_path)
        if os.path.isdir(os.path.join(vector_store_path, directory))
    ]
    return {"vectorstores": vectorstores}

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI API"}