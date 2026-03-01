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
import requests

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
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

client = OpenAI(api_key=OPENAI_API_KEY)


# ================= PLAN LIMITS =================

PLAN_LIMITS = {
    "free": {"chat": 20, "image": 3, "poster": 3, "humanize": 20},
    "basic": {"chat": -1, "image": 50, "poster": 50, "humanize": 100},
    "pro": {"chat": -1, "image": 200, "poster": 200, "humanize": -1},
    "mega": {"chat": -1, "image": -1, "poster": -1, "humanize": -1}
}


# ================= DATABASE =================

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300
)

# Test database connection
try:
    with engine.connect() as conn:
        logger.info("✅ Database connection successful")
except Exception as e:
    logger.error(f"❌ Database connection failed: {e}")
    raise

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

app = FastAPI(title="Acinyx.AI API", version="1.0.0")


# ================= MIDDLEWARE =================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for debugging"""
    logger.info(f"📥 {request.method} {request.url.path}")
    start_time = datetime.utcnow()
    
    try:
        response = await call_next(request)
        process_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.0f}ms")
        return response
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "success": False}
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= STATIC FILES =================

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/posters", StaticFiles(directory=POSTER_DIR), name="posters")


# ================= ROOT & HEALTH =================

@app.get("/")
@app.head("/")
@app.options("/")
async def root():
    """API root endpoint"""
    return {
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/signup",
            "/token",
            "/user/info",
            "/ai/chat",
            "/ai/image",
            "/ai/poster",
            "/ai/humanize",
            "/files/{file_type}/{filename}"
        ]
    }


@app.get("/health")
@app.head("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }


# ================= CATCH-ALL ROUTE =================

@app.api_route("/{path_name:path}", methods=["GET", "HEAD", "OPTIONS"])
async def catch_all(path_name: str, request: Request):
    """Catch-all route for undefined paths"""
    logger.info(f"⚠️ Undefined path: {path_name} with method {request.method}")
    
    if request.method == "HEAD":
        return JSONResponse(content={}, status_code=200)
    if request.method == "OPTIONS":
        return JSONResponse(content={}, status_code=200)
    
    raise HTTPException(status_code=404, detail=f"Endpoint '/{path_name}' not found")


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
    current = getattr(user, f"{feature}_used", 0)
    setattr(user, f"{feature}_used", current + 1)
    db.commit()


# ================= AUTH ENDPOINTS =================

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

        logger.info(f"✅ New user signed up: {data.username}")
        return {"ok": True, "message": "User created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(500, "Internal server error")


@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.username == form.username).first()
        
        if not user or not verify_password(form.password, user.password_hash):
            raise HTTPException(401, "Invalid username or password")

        token = create_token({"sub": user.username})
        
        logger.info(f"✅ User logged in: {user.username}")
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


# ================= AI ENDPOINTS =================

@app.post("/ai/chat")
async def chat(message: str = Form(...),
               user: User = Depends(get_current_user),
               db: Session = Depends(get_db)):
    try:
        if not check_usage_limit(user, "chat"):
            raise HTTPException(429, "Chat limit reached. Please upgrade your plan.")
        
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": message}],
            max_tokens=500
        )
        
        reply = res.choices[0].message.content
        increment_usage(user, "chat", db)
        
        logger.info(f"✅ Chat completed for user: {user.username}")
        return {"reply": reply, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(500, f"Server error: {str(e)}")


# ================= FIXED IMAGE ENDPOINT =================
@app.post("/ai/image")
async def image(prompt: str = Form(...),
                user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    try:
        if not check_usage_limit(user, "image"):
            raise HTTPException(429, "Image generation limit reached. Please upgrade your plan.")
        
        # FIXED: Removed 'quality' parameter - DALL-E 2 doesn't support it
        # Try DALL-E 3 first (better quality), fall back to DALL-E 2
        try:
            response = client.images.generate(
                model="dall-e-3",  # Try DALL-E 3 first
                prompt=prompt,
                size="1024x1024",
                n=1
            )
        except Exception as e:
            logger.warning(f"DALL-E 3 failed, falling back to DALL-E 2: {e}")
            response = client.images.generate(
                model="dall-e-2",  # Fall back to DALL-E 2
                prompt=prompt,
                size="1024x1024",
                n=1
            )
        
        img_url = response.data[0].url
        img_response = requests.get(img_url, timeout=30)
        
        if img_response.status_code != 200:
            raise HTTPException(500, "Failed to download generated image")
        
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        
        with open(path, "wb") as f:
            f.write(img_response.content)
        
        increment_usage(user, "image", db)
        
        image_url = f"/outputs/{filename}"
        if BASE_URL:
            image_url = f"{BASE_URL}{image_url}"
        
        logger.info(f"✅ Image generated for user: {user.username}")
        return {"image": image_url, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image generation error: {str(e)}")
        error_message = str(e)
        if "quota" in error_message.lower() or "billing" in error_message.lower():
            raise HTTPException(429, "OpenAI API quota exceeded. Please check your billing details.")
        raise HTTPException(500, f"Server error: {error_message}")


# ================= FIXED POSTER ENDPOINT =================
@app.post("/ai/poster")
async def poster(prompt: str = Form(...),
                 style: str = Form("realistic"),
                 user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    try:
        if not check_usage_limit(user, "poster"):
            raise HTTPException(429, "Poster generation limit reached. Please upgrade your plan.")
        
        enhanced_prompt = f"Create a professional poster: {prompt}. Style: {style}, high quality, marketing poster design"
        
        # FIXED: Removed 'quality' parameter
        # Try DALL-E 3 first, fall back to DALL-E 2
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=enhanced_prompt,
                size="1024x1024",
                n=1
            )
        except Exception as e:
            logger.warning(f"DALL-E 3 failed for poster, falling back to DALL-E 2: {e}")
            response = client.images.generate(
                model="dall-e-2",
                prompt=enhanced_prompt,
                size="1024x1024",
                n=1
            )
        
        img_url = response.data[0].url
        img_response = requests.get(img_url, timeout=30)
        
        if img_response.status_code != 200:
            raise HTTPException(500, "Failed to download generated poster")
        
        filename = f"poster_{uuid.uuid4()}.png"
        path = os.path.join(POSTER_DIR, filename)
        
        with open(path, "wb") as f:
            f.write(img_response.content)
        
        increment_usage(user, "poster", db)
        
        poster_url = f"/posters/{filename}"
        if BASE_URL:
            poster_url = f"{BASE_URL}{poster_url}"
        
        logger.info(f"✅ Poster generated for user: {user.username}")
        return {"poster": poster_url, "success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Poster generation error: {str(e)}")
        error_message = str(e)
        if "quota" in error_message.lower() or "billing" in error_message.lower():
            raise HTTPException(429, "OpenAI API quota exceeded. Please check your billing details.")
        raise HTTPException(500, f"Server error: {error_message}")


@app.post("/ai/humanize")
async def humanize(text: str = Form(...),
                   user: User = Depends(get_current_user),
                   db: Session = Depends(get_db)):
    try:
        if not check_usage_limit(user, "humanize"):
            raise HTTPException(429, "Humanize limit reached. Please upgrade your plan.")
        
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a text humanizer. Make the given text more natural and conversational while preserving the original meaning."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000
        )
        
        humanized_text = res.choices[0].message.content
        increment_usage(user, "humanize", db)
        
        logger.info(f"✅ Text humanized for user: {user.username}")
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


# ================= STARTUP EVENT =================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Acinyx.AI API starting up...")
    logger.info(f"📡 Database URL: {DATABASE_URL.split('@')[0] if '@' in DATABASE_URL else 'configured'}")
    logger.info(f"🔑 OpenAI API: {'configured' if OPENAI_API_KEY else 'missing'}")
    logger.info(f"🌐 Base URL: {BASE_URL or 'not set'}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"📁 Poster directory: {POSTER_DIR}")
    
    # Test OpenAI API connection
    try:
        # Simple test to check if API key is valid
        client.models.list()
        logger.info("✅ OpenAI API connection successful")
    except Exception as e:
        logger.error(f"❌ OpenAI API connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Acinyx.AI API shutting down...")


# ================= RUN =================

if __name__ == "__main__":
    uvicorn.run(
        "backendapp:app",
        host="0.0.0.0",
        port=PORT,
        reload=False
    )