# ================= IMPORTS =================

from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File
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

BASE_URL = os.getenv("BASE_URL")

PORT = int(os.getenv("PORT", 8000))

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_SECRET")

JWT_ALGORITHM = "HS256"

JWT_EXPIRE_DAYS = 30


if not DATABASE_URL:
    raise Exception("DATABASE_URL not set")


if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )


# ================= OPENAI =================

client = None

if OPENAI_API_KEY:

    client = OpenAI(api_key=OPENAI_API_KEY)

    logger.info("OpenAI ready")

else:

    logger.error("OPENAI_API_KEY missing")


# ================= PLAN LIMITS =================

PLAN_LIMITS = {

    "free": {"chat": 20, "image": 3, "poster": 3, "humanize": 20},

    "basic": {"chat": -1, "image": 50, "poster": 50, "humanize": 100},

    "pro": {"chat": -1, "image": 200, "poster": 200, "humanize": -1},

    "mega": {"chat": -1, "image": -1, "poster": -1, "humanize": -1}

}


# ================= DATABASE =================

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

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

    allow_methods=["*"],

    allow_headers=["*"],

    allow_credentials=True

)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# ================= SECURITY =================

pwd_context = CryptContext(schemes=["bcrypt"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_db():

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def hash_password(password):

    return pwd_context.hash(password)


def verify_password(password, hashed):

    return pwd_context.verify(password, hashed)


def create_token(data):

    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)

    data.update({"exp": expire})

    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):

    try:

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        username = payload.get("sub")

    except:

        raise HTTPException(401, "Invalid token")

    user = db.query(User).filter(User.username == username).first()

    if not user:

        raise HTTPException(401, "User not found")

    return user


# ================= SIGNUP =================

class SignupRequest(BaseModel):

    username: str

    email: EmailStr

    password: str


@app.post("/signup")

def signup(data: SignupRequest, db: Session = Depends(get_db)):

    exists = db.query(User).filter(or_(User.username == data.username, User.email == data.email)).first()

    if exists:

        raise HTTPException(400, "User exists")

    user = User(

        username=data.username,

        email=data.email,

        password_hash=hash_password(data.password),

    )

    db.add(user)

    db.commit()

    return {"message": "Account created"}


# ================= LOGIN =================

@app.post("/token")

def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    user = db.query(User).filter(or_(User.username == form.username, User.email == form.username)).first()

    if not user:

        raise HTTPException(401, "Invalid login")

    if not verify_password(form.password, user.password_hash):

        raise HTTPException(401, "Invalid login")

    token = create_token({"sub": user.username})

    return {"access_token": token, "token_type": "bearer", "plan": user.plan}


# ================= CHAT =================

@app.post("/ai/chat")

async def chat(message: str = Form(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    if not client:
        raise HTTPException(500, "OpenAI not configured")

    res = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[{"role": "user", "content": message}]

    )

    user.chat_used += 1

    db.commit()

    return {"reply": res.choices[0].message.content}


# ================= HUMANIZE =================

@app.post("/ai/humanize")

async def humanize(message: str = Form(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    if not client:
        raise HTTPException(500, "OpenAI not configured")

    res = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=[{"role": "user", "content": f"Rewrite naturally:\n{message}"}]

    )

    user.humanize_used += 1

    db.commit()

    return {"reply": res.choices[0].message.content}


# ================= IMAGE =================

@app.post("/ai/image")

async def image(prompt: str = Form(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    result = client.images.generate(

        model="gpt-image-1",

        prompt=prompt,

        size="1024x1024"

    )

    filename = f"{uuid.uuid4()}.png"

    path = f"{OUTPUT_DIR}/{filename}"

    with open(path, "wb") as f:

        f.write(base64.b64decode(result.data[0].b64_json))

    user.image_used += 1

    db.commit()

    return {"image": f"{BASE_URL}/outputs/{filename}"}


# ================= POSTER =================

@app.post("/ai/poster")

async def poster(

    title: str = Form(...),

    description: str = Form(""),

    microscopic_details: str = Form(""),

    style: str = Form("cinematic"),

    user: User = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    prompt = f"{title}\n{description}\n{microscopic_details}\n{style}"

    result = client.images.generate(

        model="gpt-image-1",

        prompt=prompt,

        size="1024x1536"

    )

    filename = f"{uuid.uuid4()}.png"

    path = f"{OUTPUT_DIR}/{filename}"

    with open(path, "wb") as f:

        f.write(base64.b64decode(result.data[0].b64_json))

    user.poster_used += 1

    db.commit()

    return {"image": f"{BASE_URL}/outputs/{filename}"}


# ================= RUN =================

if __name__ == "__main__":

    uvicorn.run("backendapp:app", host="0.0.0.0", port=PORT)