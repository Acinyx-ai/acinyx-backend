# ================= IMPORTS =================

from fastapi import FastAPI, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Request

from pydantic import BaseModel, EmailStr

from sqlalchemy import Column, Integer, String, create_engine, or_
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from passlib.context import CryptContext

from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError, ExpiredSignatureError

import os
import base64
import uuid
import logging
import requests  # MOVED TO TOP (FIXED)

from openai import OpenAI

import uvicorn


# ================= LOGGING =================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acinyx")


# ================= CONFIG =================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")

BASE_URL = os.getenv("BASE_URL", "").rstrip("/")

PORT = int(os.getenv("PORT", 8000))

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_SECRET")

JWT_ALGORITHM = "HS256"

JWT_EXPIRE_DAYS = 30


if not DATABASE_URL:
    raise Exception("DATABASE_URL missing")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY missing")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )


client = OpenAI(api_key=OPENAI_API_KEY)


# ================= PLAN LIMITS =================

PLAN_LIMITS = {
    "free": {"chat": 20, "image": 3, "poster": 3, "humanize": 20},
    "basic": {"chat": -1, "image": 50, "poster": 50, "humanize": 100},
    "pro": {"chat": -1, "image": 200, "poster": 200, "humanize": -1},
    "mega": {"chat": -1, "image": -1, "poster": -1, "humanize": -1}
}


# ================= DATABASE =================

# Handle SQLite vs PostgreSQL
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    plan = Column(String, default="free")
    chat_used = Column(Integer, default=0)
    image_used = Column(Integer, default=0)
    poster_used = Column(Integer, default=0)
    humanize_used = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)


# ================= OUTPUT =================

OUTPUT_DIR = "outputs"
POSTER_DIR = "posters"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(POSTER_DIR, exist_ok=True)


# ================= APP =================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)

app.mount(
    "/posters",
    StaticFiles(directory=POSTER_DIR),
    name="posters"
)


# ================= HEALTH =================

@app.get("/health")
@app.head("/health")  # ADDED HEAD support
def health():
    return {"status": "ok"}


# ================= ROOT =================

@app.get("/")
@app.head("/")  # ADDED HEAD support (FIXES 405 ERROR)
def root():
    return {"status": "running"}


# ================= SECURITY =================

pwd = CryptContext(schemes=["bcrypt"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(p):
    return pwd.hash(p)


def verify_password(p, h):
    return pwd.verify(p, h)


def create_token(data):
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    data.update({"exp": expire})
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user


def check_usage_limit(user: User, feature: str):
    """Check if user has reached their usage limit for a feature"""
    plan_limits = PLAN_LIMITS.get(user.plan, PLAN_LIMITS["free"])
    limit = plan_limits.get(feature, 0)
    
    if limit == -1:  # Unlimited
        return True
    
    used = getattr(user, f"{feature}_used", 0)
    if used >= limit:
        return False
    return True


def increment_usage(user: User, feature: str, db: Session):
    """Increment usage counter for a feature"""
    setattr(user, f"{feature}_used", getattr(user, f"{feature}_used", 0) + 1)
    db.commit()


# ================= SIGNUP =================

class Signup(BaseModel):
    username: str
    email: EmailStr
    password: str


@app.post("/signup")
def signup(data: Signup, db: Session = Depends(get_db)):
    try:
        existing_user = db.query(User).filter(
            or_(User.username == data.username, User.email == data.email)
        ).first()

        if existing_user:
            raise HTTPException(400, "Username or email already exists")

        user = User(
            username=data.username,
            email=data.email,
            password_hash=hash_password(data.password)
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        return {"ok": True, "message": "User created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(500, "Internal server error")


# ================= LOGIN =================

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.username == form.username).first()
        
        if not user:
            raise HTTPException(401, "Invalid username or password")
        
        if not verify_password(form.password, user.password_hash):
            raise HTTPException(401, "Invalid username or password")

        token = create_token({"sub": user.username})
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "username": user.username,
            "plan": user.plan
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(500, "Internal server error")


# ================= USER INFO =================

@app.get("/user/info")
def user_info(user: User = Depends(get_current_user)):
    return {
        "username": user.username,
        "email": user.email,
        "plan": user.plan,
        "usage": {
            "chat": user.chat_used,
            "image": user.image_used,
            "poster": user.poster_used,
            "humanize": user.humanize_used
        },
        "limits": PLAN_LIMITS.get(user.plan, PLAN_LIMITS["free"])
    }


# ================= CHAT =================

@app.post("/ai/chat")
async def chat(message: str = Form(...),
               user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    try:
        # Check usage limit
        if not check_usage_limit(user, "chat"):
            raise HTTPException(429, "Chat limit reached. Please upgrade your plan.")
        
        # Call OpenAI
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            max_tokens=500
        )
        
        reply = res.choices[0].message.content
        
        # Increment usage
        increment_usage(user, "chat", db)
        
        return {"reply": reply, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")


# ================= IMAGE =================

@app.post("/ai/image")
async def image(prompt: str = Form(...),
                user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    try:
        # Check usage limit
        if not check_usage_limit(user, "image"):
            raise HTTPException(429, "Image generation limit reached. Please upgrade your plan.")
        
        # Generate image
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        # Download image from URL
        img_url = response.data[0].url
        img_response = requests.get(img_url)
        
        if img_response.status_code != 200:
            raise HTTPException(500, "Failed to download generated image")
        
        # Save image
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        
        with open(path, "wb") as f:
            f.write(img_response.content)
        
        # Increment usage
        increment_usage(user, "image", db)
        
        # Return full URL or path
        image_url = f"/outputs/{filename}"
        if BASE_URL:
            image_url = f"{BASE_URL}{image_url}"
            
        return {"image": image_url, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")


# ================= POSTER =================

@app.post("/ai/poster")
async def poster(prompt: str = Form(...),
                 style: str = Form("realistic"),
                 user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    try:
        # Check usage limit
        if not check_usage_limit(user, "poster"):
            raise HTTPException(429, "Poster generation limit reached. Please upgrade your plan.")
        
        # Enhance prompt for poster generation
        enhanced_prompt = f"Create a professional poster: {prompt}. Style: {style}, high quality, marketing poster design"
        
        # Generate image
        response = client.images.generate(
            model="dall-e-2",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        # Download image
        img_url = response.data[0].url
        img_response = requests.get(img_url)
        
        if img_response.status_code != 200:
            raise HTTPException(500, "Failed to download generated poster")
        
        # Save poster
        filename = f"poster_{uuid.uuid4()}.png"
        path = os.path.join(POSTER_DIR, filename)
        
        with open(path, "wb") as f:
            f.write(img_response.content)
        
        # Increment usage
        increment_usage(user, "poster", db)
        
        # Return full URL or path
        poster_url = f"/posters/{filename}"
        if BASE_URL:
            poster_url = f"{BASE_URL}{poster_url}"
            
        return {"poster": poster_url, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Poster generation error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")


# ================= HUMANIZE =================

@app.post("/ai/humanize")
async def humanize(text: str = Form(...),
                   user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    try:
        # Check usage limit
        if not check_usage_limit(user, "humanize"):
            raise HTTPException(429, "Humanize limit reached. Please upgrade your plan.")
        
        # Humanize text using GPT
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a text humanizer. Make the given text more natural, conversational, and human-like while preserving the original meaning."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000
        )
        
        humanized_text = res.choices[0].message.content
        
        # Increment usage
        increment_usage(user, "humanize", db)
        
        return {"humanized": humanized_text, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Humanize error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")


# ================= FILE SERVER =================

@app.get("/files/{file_type}/{filename}")
async def get_file(file_type: str, filename: str):
    """Serve generated files"""
    if file_type not in ["outputs", "posters"]:
        raise HTTPException(404, "File type not found")
    
    file_path = os.path.join(file_type, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    
    return FileResponse(file_path)


# ================= ERROR HANDLERS =================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "success": False}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "success": False}
    )


# ================= RUN =================

if __name__ == "__main__":
    uvicorn.run(
        "backendapp:app",
        host="0.0.0.0",
        port=PORT,
        reload=False
    )