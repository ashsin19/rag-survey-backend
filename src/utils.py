import json
import os
import requests
import base64
from dotenv import load_dotenv
import openai
import re
from pydantic import BaseModel
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from starlette.responses import RedirectResponse
from datetime import datetime, timedelta
import bcrypt

class Token(BaseModel):
    access_token: str
    token_type: str

# User model for token payload
class TokenData(BaseModel):
    username: str | None = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class execute_api:
    def __init__(self):
        self.load_dotenv()
        self.table_name=os.getenv("TBL_NAME")
        self.API_KEY=os.getenv("SQLITE_KEY")
        self.OWNER_NAME = os.getenv("DB_OWNER")
        self.BASE_URL = os.getenv("DB_BASEURL")
        self.SECRET_KEY = os.getenv("SECRET_KEY")
        self.ALGORITHM = os.getenv("ALGORITHM")
        self.LOGIN_DB = os.getenv("LOGIN_DB")

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
    
    def get_user_from_db(self,username: str):
        cursor=self.execute_query(self.LOGIN_DB,f"SELECT username, password FROM users WHERE username = '{username}'")
        for item in cursor:
            result = True
            if result:
                return {"username": item[0]['Value'], "hashed_password": item[1]['Value']}
        return None   

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