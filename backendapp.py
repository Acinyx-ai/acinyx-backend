from fastapi import (
    FastAPI, HTTPException, Depends,
    UploadFile, File, Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from passlib.context import CryptContext

from datetime import datetime, timedelta
from jose import jwt, JWTError

import os
import time
import base64

from openai import OpenAI
from PIL import Image, ImageDraw

import uvicorn


# =================================================
# ENV
# =================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24

# IMPORTANT: do NOT pass proxies / custom http client
client = OpenAI()


# =================================================
# DATABASE
# =================================================
DATABASE_URL = "sqlite:///./acinyx.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    plan = Column(String, default="free")
    chat_used = Column(Integer, default=0)
    poster_used = Column(Integer, default=0)


# ✅ IMPORTANT
# Only create tables – DO NOT DROP
Base.metadata.create_all(bind=engine)


# =================================================
# APP
# =================================================
app = FastAPI(title="Acinyx.AI Backend", version="3.9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =================================================
# SECURITY
# =================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def hash_password(p: str):
    return pwd_context.hash(p)


def verify_password(p: str, h: str):
    return pwd_context.verify(p, h)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =================================================
# JWT helpers
# =================================================
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user


# =================================================
# PLANS
# =================================================
PLANS = {
    "free": {
        "chat": 5,
        "poster": 2,
        "watermark": True
    },
    "basic": {
        "chat": 100,
        "poster": 20,
        "watermark": False
    },
    "pro": {
        "chat": 500,
        "poster": 100,
        "watermark": False
    },
    "mega": {
        "chat": 2000,
        "poster": 300,
        "watermark": False
    },
}


# =================================================
# FILE SYSTEM
# =================================================
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# =================================================
# AUTH
# =================================================
class SignupBody(BaseModel):
    username: str
    email: str
    password: str


@app.post("/signup")
def signup(data: SignupBody, db: Session = Depends(get_db)):

    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(400, "User exists")

    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(400, "Email already registered")

    user = User(
        username=data.username,
        email=data.email,
        password_hash=hash_password(data.password)
    )

    db.add(user)
    db.commit()

    return {"message": "Account created"}


@app.post("/token")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    user = db.query(User).filter(User.username == form.username).first()

    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    access_token = create_access_token({"sub": user.username})

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "plan": user.plan
    }


# =================================================
# AI CHAT
# =================================================
@app.post("/ai/chat")
async def ai_chat(
    message: str = Form(None),
    image: UploadFile = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    if user.chat_used >= PLANS[user.plan]["chat"]:
        raise HTTPException(403, "Chat limit reached")

    if not message and not image:
        raise HTTPException(422, "Message or image required")

    messages = [
        {"role": "system", "content": "You are Acinyx.AI. Analyze images when provided."}
    ]

    if image:
        encoded = base64.b64encode(await image.read()).decode()
        mime = image.content_type or "image/png"

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": message or "Analyze this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"}
                }
            ]
        })
    else:
        messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.6
    )

    user.chat_used += 1
    db.commit()

    return {"reply": response.choices[0].message.content}


# =================================================
# AI POSTER
# =================================================
SIZE_MAP = {
    "portrait": "1024x1536",
    "square": "1024x1024",
    "landscape": "1536x1024",
    "instagram": "1080x1920",
}


@app.post("/ai/poster/ai-generate")
async def ai_poster(
    title: str = Form(""),
    description: str = Form(""),
    style: str = Form("cinematic"),
    size: str = Form("portrait"),
    image: UploadFile = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    if user.poster_used >= PLANS[user.plan]["poster"]:
        raise HTTPException(403, "Poster limit reached")

    image_size = SIZE_MAP.get(size, "1024x1536")

    prompt = f"""
Create a {style} poster.
Cinematic lighting, professional quality.
Subject: {title}
Description: {description}
"""

    img = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=image_size
    )

    image_bytes = base64.b64decode(img.data[0].b64_json)

    filename = f"poster_{int(time.time())}.png"
    path = f"outputs/{filename}"

    with open(path, "wb") as f:
        f.write(image_bytes)

    if PLANS[user.plan]["watermark"]:
        im = Image.open(path).convert("RGBA")
        draw = ImageDraw.Draw(im)
        draw.text((20, im.height - 40), "Acinyx.AI", fill=(255, 255, 255, 160))
        im.save(path)

    user.poster_used += 1
    db.commit()

    return {"poster_url": f"/outputs/{filename}"}


# =================================================
# RUN
# =================================================
if __name__ == "__main__":
    uvicorn.run(
        "backendapp:app",
        host="0.0.0.0",
        port=8000
    )
