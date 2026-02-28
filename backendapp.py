# ================= IMPORTS =================

from fastapi import FastAPI, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from pydantic import BaseModel, EmailStr

from sqlalchemy import Column, Integer, String, create_engine, or_
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from passlib.context import CryptContext

from datetime import datetime, timedelta

from jose import jwt, JWTError

import os
import base64
import uuid
import logging

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

engine = create_engine(
    DATABASE_URL,
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

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}


# ================= ROOT =================

@app.get("/")
def root():
    return {"status": "running"}


# ================= SECURITY =================

pwd = CryptContext(schemes=["bcrypt"])

oauth2 = OAuth2PasswordBearer(tokenUrl="token")


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
    data["exp"] = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_user(token: str = Depends(oauth2), db: Session = Depends(get_db)):
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    user = db.query(User).filter(
        User.username == payload["sub"]
    ).first()
    return user


# ================= SIGNUP =================

class Signup(BaseModel):

    username: str

    email: EmailStr

    password: str


@app.post("/signup")
def signup(data: Signup, db: Session = Depends(get_db)):

    if db.query(User).filter(
        or_(User.username == data.username, User.email == data.email)
    ).first():

        raise HTTPException(400, "Exists")

    user = User(
        username=data.username,
        email=data.email,
        password_hash=hash_password(data.password)
    )

    db.add(user)

    db.commit()

    return {"ok": True}


# ================= LOGIN =================

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):

    user = db.query(User).filter(
        User.username == form.username
    ).first()

    if not verify_password(form.password, user.password_hash):

        raise HTTPException(401)

    return {
        "access_token": create_token(
            {"sub": user.username}
        )
    }


# ================= CHAT =================

@app.post("/ai/chat")
def chat(message: str = Form(...),
         user: User = Depends(get_user),
         db: Session = Depends(get_db)):

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": message}]
    )

    return {"reply": res.choices[0].message.content}


# ================= IMAGE =================

@app.post("/ai/image")
def image(prompt: str = Form(...),
          user: User = Depends(get_user)):

    img = client.images.generate(
        model="gpt-image-1",
        prompt=prompt
    )

    filename = f"{uuid.uuid4()}.png"

    path = f"{OUTPUT_DIR}/{filename}"

    with open(path, "wb") as f:
        f.write(base64.b64decode(img.data[0].b64_json))

    return {"image": f"/outputs/{filename}"}


# ================= RUN =================

if __name__ == "__main__":

    uvicorn.run("backendapp:app",
                host="0.0.0.0",
                port=PORT)# 
