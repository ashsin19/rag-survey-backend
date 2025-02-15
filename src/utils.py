import json
import tempfile
import os
import requests
import base64
from dotenv import load_dotenv
import platform
import re
from pydantic import BaseModel
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from jose import JWTError, jwt
from starlette.responses import RedirectResponse
from datetime import datetime, timedelta
import bcrypt
import shutil
import stat
import os
import cv2
import logging
import pytesseract
from pdf2image import convert_from_path
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import torch
import gc
import numpy as np
import concurrent.futures
from google.cloud import storage
from google.cloud import vision
from pydantic import BaseModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import re
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class Token(BaseModel):
    access_token: str
    token_type: str

# User model for token payload
class TokenData(BaseModel):
    username: str | None = None

class QueryRequest(BaseModel):
    query: str

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class execute_api:
    def __init__(self):
        self.load_dotenv()
        self.OPENAI_KEY = os.getenv("OPENAI_KEY")
        self.table_name=os.getenv("TBL_NAME")
        self.API_KEY=os.getenv("SQLITE_KEY")
        self.OWNER_NAME = os.getenv("DB_OWNER")
        self.BASE_URL = os.getenv("DB_BASEURL")
        self.SECRET_KEY = os.getenv("SECRET_KEY")
        self.ALGORITHM = os.getenv("ALGORITHM")
        self.LOGIN_DB = os.getenv("LOGIN_DB")
        self.UPLOAD_DIR = os.getenv("UPLOAD_DIR")
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
        self.BUCKET_NAME = os.getenv("BUCKET_NAME")
        self.storage_client = storage.Client()
        self.vector_stores = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for", "about",
    "above", "after", "against", "between", "from", "in", "into", "on", "with",
    "we", "us", "it", "he", "she", "they", "is", "are", "was", "were", "been",
    "by", "to", "of", "at", "before", "after", "over", "under", "as", "if", "because", 
    "once", "while", "just", "now", "then", "too", "very", "can", "could", 
    "shall", "should", "will", "would", "may", "might", "must", "do", "does", "did"
}
        if platform.system() == "Linux" and shutil.which("pdftotext"):
            # Use system-installed pdftotext for Docker environment
            self.POPPLER_PATH = "/usr/bin"
        else:
            # Use local Poppler path for development environment
            self.POPPLER_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "poppler", "Library", "bin"))
        if platform.system() == "Linux" and shutil.which("tesseract"):
            # Docker environment: Use system-installed Tesseract binary
            self.TESSERACT_PATH = "/usr/bin/tesseract"
        else:
            # Local environment (Windows): Use the local Tesseract executable path
            self.TESSERACT_PATH = os.path.abspath(os.path.join(os.getcwd(), "..", "Tesseract-OCR", "tesseract.exe"))
        pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_PATH
        logging.basicConfig(level=logging.DEBUG)

    def create_folders_in_src(self,folders):
        """Create folders in 'src' directory with Linux-specific permission handling."""
        src_path = os.path.abspath("src")  # Get the absolute path to the 'src' directory

        for folder in folders:
            folder_path = os.path.join(src_path, folder)
            
            if not os.path.exists(folder_path):
                try:
                    os.makedirs(folder_path)
                    print(f"Created folder: {folder_path}")
                    
                    # Linux-specific: Set directory permissions to 755
                    if os.name == "posix":  
                        os.chmod(folder_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                        print(f"Set Linux permissions for: {folder_path}")
                except PermissionError:
                    print(f"Permission denied: {folder_path}")
                except OSError as e:
                    print(f"Error creating folder {folder_path}: {e}")
            else:
                print(f"Folder already exists: {folder_path}")


    def verify_password(self,plain_password: str, hashed_password: str):
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

    def load_dotenv(self):
        load_dotenv()

    def authenticate_user(self,username: str, password: str):
        user = self.get_user_from_db(username)
        if user is None:
            return False
        if not self.verify_password(password, user['hashed_password'].encode('utf-8')):
            return False
        return user

    def create_access_token(self,data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    def upload_vectorstore_to_gcs(self,vectorstore_path, gcs_prefix="vectorstore"):
        """Upload all files in a FAISS vectorstore directory to Google Cloud Storage."""
        try: 
            
            bucket = self.storage_client.bucket(self.BUCKET_NAME)
            for root, _, files in os.walk(vectorstore_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, vectorstore_path)  # Preserve folder structure
                    destination_blob_name = f"{gcs_prefix}/{relative_path}"
                    
                    blob = bucket.blob(destination_blob_name)
                    blob.upload_from_filename(local_file_path)
                    
                    print(f"Uploaded {local_file_path} to {destination_blob_name}.")
        except Exception as e:
            print(f"Error uploading file: {str(e)}")


    def upload_pdf_to_gcs(self,local_file_path, destination_blob_name,foldername):
        """Upload a PDF to Google Cloud Storage."""
        try:
            bucket = self.storage_client.bucket(self.BUCKET_NAME)
            
            # Create the full path in the bucket
            destination_blob_name = f"{foldername}/{destination_blob_name}"
            blob = bucket.blob(destination_blob_name)
            
            # Log the local file path and destination
            print(f"Uploading {local_file_path} to {destination_blob_name}...")

            # Upload the file
            blob.upload_from_filename(local_file_path)
            
            # Verify upload success
            if blob.exists():
                print(f"File successfully uploaded to {destination_blob_name}.")
            else:
                print(f"File upload failed for {destination_blob_name}.")
    
        except Exception as e:
            print(f"Error uploading file: {str(e)}")

    def list_filenames_in_gcs(self):
        """List unique filenames (folders) under the vectorstore root in GCS."""
        bucket = self.storage_client.bucket(self.BUCKET_NAME)
        
        blobs = bucket.list_blobs(prefix=self.VECTOR_DB_PATH)
        filenames = set()
        
        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) > 1 and parts[1].strip():  # Ensure it's not the root folder itself
                filenames.add(parts[1])  # Add the second part of the path (filename folder)
        
        return list(filenames)
    
    def download_vectorstore_from_gcs(self,filename):
        """Download vector store files from GCS and save them locally."""
        bucket = self.storage_client.bucket(self.BUCKET_NAME)

        local_path = f"/tmp/{self.VECTOR_DB_PATH}/{filename}"
        os.makedirs(local_path, exist_ok=True)

        blobs = bucket.list_blobs(prefix=f"{self.VECTOR_DB_PATH}/{filename}/")
        for blob in blobs:
            local_file_path = os.path.join(local_path, blob.name.split("/")[-1])
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {blob.name} to {local_file_path}")
        return local_path

    def get_user_from_db(self,username: str):
        cursor=self.execute_query(self.LOGIN_DB,f"SELECT username, password FROM users WHERE username = '{username}'")
        for item in cursor:
            result = True
            if result:
                return {"username": item[0]['Value'], "hashed_password": item[1]['Value']}
        return None   
    
    def process_image(self,img):
        """Function to extract text from a single image using pytesseract."""
        return pytesseract.image_to_string(img, config="--oem 1 --psm 6")

    def extract_text_from_images(self,pdf_path):
        """Extracts text using OCR from scanned PDF images."""
        client = vision.ImageAnnotatorClient()
        extracted_text = []
        # Use a temporary directory for image files
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(
                pdf_path,
                poppler_path=self.POPPLER_PATH,
                output_folder=temp_dir,  # Save images in the temporary directory
                fmt='jpeg',
                dpi=150,
                strict=False
            )
            for image in images:
                # Convert PIL Image to bytes
                image_byte_array = io.BytesIO()
                image.save(image_byte_array, format='JPEG')
                image_content = image_byte_array.getvalue()

                image = vision.Image(content=image_content)
                response = client.text_detection(image=image)

                if response.error.message:
                    raise Exception(f"Vision API error: {response.error.message}")
                extracted_text.append(response.full_text_annotation.text)
            
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     results = list(executor.map(self.process_image, images))
        
            # extracted_text = "\n".join(results)
        # Join all extracted text into a single string
        return "\n".join(extracted_text)
    
    def tokenize_words(self,text):
        """Clean text, lemmatize, and tokenize into words while filtering stop words."""
        words = re.findall(r'\b\w+\b', text.lower())  # Extract words using regex
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return lemmatized_words

    def execute_query(self,DB_NAME,sql_query):
        """Execute a SQL query against DBHub.io."""
        url = f"{self.BASE_URL}/query"
        query_bytes = sql_query.encode("ascii") 
        query_base64 = base64.b64encode(query_bytes) 
        # Prepare the payload
        params = {
            'apikey': self.API_KEY,
            'dbowner': self.OWNER_NAME,
            'dbname': DB_NAME,
            'sql': query_base64
        }
        # Make the POST request
        response = requests.post(url, params)

        # Check if the response is successful
        if response.status_code == 200:
            return response.json()  # Return the JSON response if successful
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def get_report_comparison(self, query, rpt1, rpt2):
        res1 = rpt1.similarity_search(query, k=7)
        res2 = rpt2.similarity_search(query, k=7)

        content1 = set()
        content2 = set()
        for doc in res1:
            content1.update(self.split_text_to_sentences(doc.page_content))
        for doc in res2:
            content2.update(self.split_text_to_sentences(doc.page_content))
        # common_sentences = self.extract_common_sentences_cosine(list(content1), list(content2))
        common_words = self.extract_common_words(list(content1), list(content2))
        unique_to_report1 = content1 - content2
        unique_to_report2 = content2 - content1
        return {
            "common_insights": list(common_words),
            "unique_in_report_1": list(unique_to_report1),
            "unique_in_report_2": list(unique_to_report2)
        }
    
    def split_text_to_sentences(self,text):
        pattern_match = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
        sentences = re.split(pattern_match,text)
        processed_sentences = [sentence.replace("\n", " ").strip() for sentence in sentences if sentence.strip()]
        return processed_sentences
        
    def remove_unicode_characters(self,text):
        """Remove Unicode characters and replace them with plain text."""
        if isinstance(text, str):
            return re.sub(r'[^\x00-\x7F]+', ' ', text)  # Replace non-ASCII characters with a space
        elif isinstance(text, list):
            return [self.remove_unicode_characters(t) for t in text]  # Recursively clean lists
        return text

    def extract_common_sentences_cosine(self, s1, s2):
        threshold = 0.8
        similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embed1 = similarity_model.encode(s1, convert_to_tensor=True)
        embed2 = similarity_model.encode(s2, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(embed1,embed2)
        common_sentences = []
        for i in range(len(s1)):
            for j in range(len(s2)):
                if scores[i][j] > threshold:
                    common_sentences.append(s1[i])
        return list(set(common_sentences))
    
    def extract_common_words(self,content1, content2):
        """Extract common words between two sets of content with their frequencies."""
        words1 = Counter(self.tokenize_words(" ".join(content1)))
        words2 = Counter(self.tokenize_words(" ".join(content2)))

        common_words_set = {word for word in words1 if word in words2}
        common_words_list = list(common_words_set)
        return common_words_list


    def get_document_rerank(self,n: int, query: str, docs):
        """Re-rank documents using Langchain's OpenAI integration."""
        try:
            llm = OpenAI(openai_api_key=self.OPENAI_KEY, temperature=0.3)
            
            # Construct the prompt using Langchain's PromptTemplate
            prompt_template = PromptTemplate(
                input_variables=["query", "documents"],
                template="""
                You are a helpful assistant that ranks documents based on relevance to the query.
                
                Query: {query}
                
                Documents:
                {documents}
                
                Please return the documents in descending order of relevance with their original content.
                """
            )
            print(f"Prompt Template {prompt_template}")
            print(f"Type of variable docs: {type(docs)}")
            print(f"Documents: {docs}")
            for fdoc in docs:
                try:
                    print(fdoc.page_content)
                except:
                    raise HTTPException(status_code=500, detail=f"Error comparing reports: {str(e)}")
            cleaned_docs = [ doc.page_content.replace("\n", " ").replace("=", "").replace("'", "").replace('"', "") for doc in docs ]
            print(f"Cleaned Docs of length {len(cleaned_docs)}: {cleaned_docs}")
            # Prepare the list of documents as a single string
            documents_text = "\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(cleaned_docs)])

            # Run the LLM with the constructed prompt
            chain = LLMChain(llm=llm, prompt=prompt_template)
            response = chain.run({"query": query, "documents": documents_text})
            
            return self.parse_reranked_documents(response, n)
        except Exception as e:
            print(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error comparing reports: {str(e)}")

    def parse_reranked_documents(self,response, n):
        """Parse the response to extract the top N documents."""
        # Split the response by documents and clean up the text
        ranked_docs = response.split("Document ")[1:]
        ranked_docs = [doc.split(": ", 1)[1].strip() for doc in ranked_docs if ": " in doc]
        
        return ranked_docs[:n]

    def ensure_list(self,value):
        """Ensure the value is always returned as a list."""
        if isinstance(value, list):
            return value
        return [value]

    def generate_compare_reports(self, query,vec_stores:dict):
        rept_keys = list(vec_stores.keys())
        results = []
        try:
            for x in range(len(rept_keys)):
                for y in range(x+1,len(rept_keys)):
                    comparison = self.get_report_comparison(query, vec_stores[rept_keys[x]],vec_stores[rept_keys[y]])
                    formatted_result = {
                    "report_1": rept_keys[x],
                    "report_2": rept_keys[y],
                    "comparison": {
                        "common_insights": [],
                        "unique_in_report_1": self.ensure_list(self.remove_unicode_characters(comparison["unique_in_report_1"])),
                        "unique_in_report_2": self.ensure_list(self.remove_unicode_characters(comparison["unique_in_report_2"]))
                        }
                    }
                
                    results.append(formatted_result)
            response={"comparisons": results}
            if not results:
                response = {"comparisons": "No comparisons found."}
            return JSONResponse(content=response, media_type="application/json")

        except Exception as e:
            print(f"Exception: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error comparing reports: {str(e)}")

    def construct_query_prompt(self,user_query):
        """Refine user query to be more context-aware and improve LLM response quality."""
        prompt_template = f"""
        You are an AI assistant helping a user retrieve insights from a research report.
        Given the following question: "{user_query}",
        improve it to be more specific, structured, and informative.
        If needed, add context for better retrieval.

        Here are the rules that we want to follow in generating the insights

        1. The insights should be concise and framed as a structued result for better understanding
        2. Extract the key topics in the user query
        3. If required, then the question can be segregated into logical components 

        You should return the improved queries only. Avoid any unnecessary detail
        """
        return prompt_template
    
    def generate_wordcloud(self,text):
        """Generate a word cloud from the given text and return it as a base64 string."""
        wordcloud = WordCloud(width=800, height=400, background_color="black").generate(text)
        buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        return img_base64
      

    def verify_token(self,token: str = Depends(oauth2_scheme)):
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid authentication token")
            return username
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
