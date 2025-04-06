from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from fastapi_jwt_auth import AuthJWT
import supabase
import bcrypt
from config import Settings

router = APIRouter()
settings = Settings()

# Initialize Supabase Client
supabase_client = supabase.create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

# User schema
class UserLogin(BaseModel):
    email: str
    password: str

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

@router.post("/login")
def login(user: UserLogin, Authorize: AuthJWT = Depends()):
    try:
        # Fetch user from Supabase
        response = supabase_client.from_('users').select('*').eq('email', user.email).single()
        if not response or not verify_password(user.password, response['password']):
            raise HTTPException(status_code=400, detail="Invalid credentials")
        
        # Generate JWT token
        access_token = Authorize.create_access_token(subject=user.email)
        return {"access_token": access_token}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/signup")
def signup(user: UserLogin):
    try:
        # Hash password
        hashed_password = hash_password(user.password)
        
        # Register user in Supabase
        response = supabase_client.from_('users').insert({
            'email': user.email,
            'password': hashed_password,
        }).execute()
        
        if not response:
            raise HTTPException(status_code=400, detail="User already exists")
        
        return {"message": "User registered successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))