from fastapi import FastAPI, HTTPException, Depends, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from pydantic import BaseModel, EmailStr

from sqlalchemy import Column, Integer, String, create_engine, or_, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from passlib.context import CryptContext

from datetime import datetime, timedelta

from jose import jwt, JWTError

import os, base64, uuid, logging

from openai import OpenAI

import uvicorn


# ================= CONFIG =================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acinyx")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")

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


client = OpenAI(api_key=OPENAI_API_KEY)


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

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, unique=True, index=True)

    email = Column(String, unique=True, index=True)

    password_hash = Column(String)

    plan = Column(String, default="free")

    chat_used = Column(Integer, default=0)

    poster_used = Column(Integer, default=0)


class ChatMemory(Base):

    __tablename__ = "chat_memory"

    id = Column(Integer, primary_key=True, index=True)

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

    return jwt.encode(

        data,

        JWT_SECRET,

        algorithm=JWT_ALGORITHM

    )


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

def signup(

    data: SignupRequest,

    db: Session = Depends(get_db)

):

    existing = db.query(User).filter(

        or_(

            User.username == data.username,

            User.email == data.email

        )

    ).first()


    if existing:

        raise HTTPException(

            400,

            "Username or email already exists"

        )


    user = User(

        username=data.username,

        email=data.email,

        password_hash=hash_password(data.password)

    )


    db.add(user)

    db.commit()


    return {

        "message": "Account created"

    }


# ================= LOGIN =================

@app.post("/token")

def login(

    form: OAuth2PasswordRequestForm = Depends(),

    db: Session = Depends(get_db)

):

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


# ================= CHAT =================

@app.post("/ai/chat")

async def chat(

    message: str = Form(""),

    mode: str = Form("chat"),

    user: User = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    if user.chat_used >= 50:

        raise HTTPException(403, "Limit reached")


    reply = None

    image_path = None


    # IMAGE MODE

    if mode == "image":

        img = client.images.generate(

            model="gpt-image-1",

            prompt=message,

            size="1024x1024"

        )

        image_bytes = base64.b64decode(

            img.data[0].b64_json

        )

        filename = f"{uuid.uuid4()}.png"

        path = f"outputs/{filename}"

        with open(path, "wb") as f:

            f.write(image_bytes)

        reply = "Here is your image."

        image_path = path


    # HUMANIZER MODE

    elif mode == "humanize":

        res = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[

                {

                    "role": "system",

                    "content":

                    "Rewrite this text to sound natural and human."

                },

                {

                    "role": "user",

                    "content": message

                }

            ]

        )

        reply = res.choices[0].message.content


    # NORMAL CHAT

    else:

        res = client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[

                {

                    "role": "user",

                    "content": message

                }

            ]

        )

        reply = res.choices[0].message.content


    db.add(ChatMemory(

        user_id=user.id,

        role="user",

        content=message

    ))


    db.add(ChatMemory(

        user_id=user.id,

        role="assistant",

        content=reply

    ))


    user.chat_used += 1


    db.commit()


    return {

        "reply": reply,

        "image": image_path

    }


# ================= POSTER =================

@app.post("/ai/poster")

async def poster(

    title: str = Form(...),

    user: User = Depends(get_current_user),

    db: Session = Depends(get_db)

):

    if user.poster_used >= 50:

        raise HTTPException(403, "Poster limit reached")


    img = client.images.generate(

        model="gpt-image-1",

        prompt=title,

        size="1024x1536"

    )


    image_bytes = base64.b64decode(

        img.data[0].b64_json

    )


    filename = f"{uuid.uuid4()}.png"

    path = f"outputs/{filename}"


    with open(path, "wb") as f:

        f.write(image_bytes)


    user.poster_used += 1

    db.commit()


    return {

        "image": path

    }


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