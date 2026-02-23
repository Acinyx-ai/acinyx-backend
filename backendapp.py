# ================= IMPORTS =================

from fastapi import FastAPI, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from pydantic import BaseModel, EmailStr

from sqlalchemy import Column, Integer, String, create_engine, or_, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from passlib.context import CryptContext

from datetime import datetime, timedelta

from jose import jwt, JWTError

import os
import base64
import uuid
import logging
import requests

from openai import OpenAI

import uvicorn


# ================= CONFIG =================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acinyx")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
PAYSTACK_SECRET = os.getenv("PAYSTACK_SECRET")

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

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set")

if not PAYSTACK_SECRET:
    logger.warning("PAYSTACK_SECRET not set")


client = OpenAI(api_key=OPENAI_API_KEY)


# ================= PLAN LIMITS =================

PLAN_LIMITS = {

    "free": {"chat": 20, "poster": 3},

    "basic": {"chat": -1, "poster": 50},

    "pro": {"chat": -1, "poster": 200},

    "mega": {"chat": -1, "poster": -1},

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


# ================= MODELS =================

class User(Base):

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)

    username = Column(String, unique=True)

    email = Column(String, unique=True)

    password_hash = Column(String)

    plan = Column(String, default="free")

    chat_used = Column(Integer, default=0)

    poster_used = Column(Integer, default=0)



class ChatMemory(Base):

    __tablename__ = "chat_memory"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer)

    role = Column(String)

    content = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)



Base.metadata.create_all(bind=engine)


# ================= APP =================

app = FastAPI()

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)


os.makedirs("outputs", exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ================= SECURITY =================

pwd_context = CryptContext(schemes=["bcrypt"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def hash_password(password):

    return pwd_context.hash(password)


def verify_password(password, hashed):

    return pwd_context.verify(password, hashed)


def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close()


def create_access_token(data):

    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS)

    data.update({"exp": expire})

    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(

    token: str = Depends(oauth2_scheme),

    db: Session = Depends(get_db)

):

    try:

        payload = jwt.decode(

            token,

            JWT_SECRET,

            algorithms=[JWT_ALGORITHM]

        )

        username = payload.get("sub")

    except JWTError:

        raise HTTPException(401, "Invalid token")


    user = db.query(User).filter(

        User.username == username

    ).first()


    if not user:

        raise HTTPException(401, "Invalid token")


    return user


# ================= SIGNUP =================

class SignupRequest(BaseModel):

    username: str
    email: EmailStr
    password: str


@app.post("/signup")

def signup(data: SignupRequest, db: Session = Depends(get_db)):

    existing = db.query(User).filter(

        or_(

            User.username == data.username,

            User.email == data.email

        )

    ).first()


    if existing:

        raise HTTPException(400, "User exists")


    user = User(

        username=data.username,

        email=data.email,

        password_hash=hash_password(data.password)

    )


    db.add(user)

    db.commit()

    return {"message": "Account created"}


# ================= LOGIN =================

@app.post("/token")

def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    user = db.query(User).filter(

        or_(

            User.username == form.username,

            User.email == form.username

        )

    ).first()


    if not user:

        raise HTTPException(401, "Invalid login")


    if not verify_password(

        form.password,

        user.password_hash

    ):

        raise HTTPException(401, "Invalid login")


    token = create_access_token(

        {"sub": user.username}

    )


    return {

        "access_token": token,

        "token_type": "bearer",

        "plan": user.plan

    }


# ================= PAYSTACK =================

class PaystackRequest(BaseModel):

    plan: str
    amount: int


@app.post("/payments/paystack/init")

def paystack_init(

    data: PaystackRequest,

    user: User = Depends(get_current_user)

):

    try:

        if not PAYSTACK_SECRET:

            raise HTTPException(500, "Payment not configured")


        headers = {

            "Authorization": f"Bearer {PAYSTACK_SECRET}",

            "Content-Type": "application/json"

        }


        payload = {

            "email": user.email,

            "amount": data.amount,

            "metadata": {

                "plan": data.plan

            }

        }


        response = requests.post(

            "https://api.paystack.co/transaction/initialize",

            json=payload,

            headers=headers,

            timeout=30

        )


        result = response.json()


        if not result.get("status"):

            raise HTTPException(

                400,

                result.get("message")

            )


        return {

            "authorization_url":

            result["data"]["authorization_url"]

        }


    except Exception as e:

        logger.error(f"PAYSTACK ERROR: {str(e)}")

        raise HTTPException(

            500,

            "Payment initialization failed"

        )


# ================= CHAT =================

@app.post("/ai/chat")

async def chat(

    message: str = Form(""),

    user: User = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    try:

        limits = PLAN_LIMITS[user.plan]


        if limits["chat"] != -1 and user.chat_used >= limits["chat"]:

            raise HTTPException(403, "Chat limit reached")


        if not OPENAI_API_KEY:

            raise HTTPException(500, "AI not configured")


        response = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[

                {

                    "role": "user",

                    "content": message

                }

            ]

        )


        reply = response.choices[0].message.content


        user.chat_used += 1


        db.add(

            ChatMemory(

                user_id=user.id,

                role="assistant",

                content=reply

            )

        )


        db.commit()


        return {

            "reply": reply

        }


    except HTTPException:

        raise


    except Exception as e:

        logger.error(f"OPENAI ERROR: {str(e)}")

        raise HTTPException(

            500,

            "AI server error"

        )


# ================= HEALTH =================

@app.get("/health")

def health():

    return {

        "status": "ok"

    }


# ================= RUN =================

if __name__ == "__main__":

    uvicorn.run(

        "backendapp:app",

        host="0.0.0.0",

        port=8000

    )