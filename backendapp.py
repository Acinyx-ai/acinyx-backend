from fastapi import (
    FastAPI, HTTPException, Depends,
    UploadFile, File, Form, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from sqlalchemy import Column, Integer, String, create_engine, or_, Text, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError

import os
import time
import base64
import logging
import requests
import hmac
import hashlib

from openai import OpenAI
from PIL import Image, ImageDraw
import uvicorn


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
if not PAYSTACK_SECRET_KEY:
    raise RuntimeError("PAYSTACK_SECRET_KEY not set")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_THIS_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24

client = OpenAI()


# -------------------------------------------------
# DATABASE (PERSISTENT FIX)
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "acinyx.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


# -------------------------------------------------
# MODELS
# -------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    plan = Column(String, default="free")
    chat_used = Column(Integer, default=0)
    poster_used = Column(Integer, default=0)


class ChatMemory(Base):
    __tablename__ = "chat_memory"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    role = Column(String(20))
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# -------------------------------------------------
# APP
# -------------------------------------------------

<<<<<<< HEAD
app = FastAPI(title="Acinyx.AI Backend", version="5.2.0")
=======
app = FastAPI(title="Acinyx.AI Backend", version="5.3.1")
>>>>>>> 599c4e6 (Fixed persistent database and finalized backend)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str):
    return pwd_context.verify(password, hashed)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode = data.copy()
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
            raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(401, "Invalid token")

    return user


# -------------------------------------------------
# PLANS
# -------------------------------------------------

PLANS = {
    "free": {"chat": 5, "poster": 2, "watermark": True},
    "basic": {"chat": 100, "poster": 20, "watermark": False},
    "pro": {"chat": 500, "poster": 100, "watermark": False},
    "mega": {"chat": 2000, "poster": 300, "watermark": False},
}


# -------------------------------------------------
# AUTH
# -------------------------------------------------

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
    db.refresh(user)

    return {"message": "Account created"}


@app.post("/token")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):

    user = db.query(User).filter(
        or_(User.username == form.username, User.email == form.username)
    ).first()

    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer",
        "plan": user.plan
    }


# -------------------------------------------------
# PAYSTACK INIT
# -------------------------------------------------

class PaystackInitBody(BaseModel):
    amount: int
    plan: str


@app.post("/payments/paystack/init")
def init_paystack_payment(
    body: PaystackInitBody,
    user: User = Depends(get_current_user)
):

    if body.plan not in PLANS:
        raise HTTPException(400, "Invalid plan")

    payload = {
        "email": user.email,
        "amount": body.amount,
        "callback_url": "https://acinyx-ai.vercel.app/dashboard",
        "metadata": {
            "username": user.username,
            "plan": body.plan
        }
    }

    r = requests.post(
        "https://api.paystack.co/transaction/initialize",
        json=payload,
        headers={
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json"
        },
        timeout=15
    )

    data = r.json()

    if not data.get("status"):
        raise HTTPException(400, data.get("message", "Paystack error"))

    return {
        "authorization_url": data["data"]["authorization_url"],
        "reference": data["data"]["reference"]
    }


# -------------------------------------------------
# PAYSTACK WEBHOOK
# -------------------------------------------------

@app.post("/payments/paystack/webhook")
async def paystack_webhook(request: Request, db: Session = Depends(get_db)):

    raw_body = await request.body()
    signature = request.headers.get("x-paystack-signature")

    expected = hmac.new(
        PAYSTACK_SECRET_KEY.encode(),
        raw_body,
        hashlib.sha512
    ).hexdigest()

    if not signature or not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid signature")

    payload = await request.json()

    if payload.get("event") != "charge.success":
        return {"status": "ignored"}

    data = payload.get("data", {})
    metadata = data.get("metadata", {})

    username = metadata.get("username")
    plan = metadata.get("plan")

    user = db.query(User).filter(User.username == username).first()

    if user and plan in PLANS:
        user.plan = plan
        user.chat_used = 0
        user.poster_used = 0
        db.commit()

    return {"status": "ok"}
# -------------------------------------------------
# CHAT WITH MEMORY
# -------------------------------------------------

MAX_HISTORY = 12


@app.post("/ai/chat")
async def ai_chat(
    message: str = Form(None),
    image: UploadFile = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    if message and len(message) > 4000:
        raise HTTPException(400, "Message too long")

    if user.chat_used >= PLANS[user.plan]["chat"]:
        raise HTTPException(403, "Chat limit reached")

    if not message and not image:
        raise HTTPException(422, "Message or image required")

    history = (
        db.query(ChatMemory)
        .filter(ChatMemory.user_id == user.id)
        .order_by(ChatMemory.created_at.desc())
        .limit(MAX_HISTORY)
        .all()
    )

    history.reverse()

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    messages = [
        {
            "role": "system",
            "content": f"You are Acinyx.AI. The current time is {now}."
        }
    ]

    for h in history:
        messages.append({"role": h.role, "content": h.content})

    # ---------------------------------------------
    # Handle image input
    # ---------------------------------------------

    if image:
        encoded = base64.b64encode(await image.read()).decode()
        mime = image.content_type or "image/png"

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": message or "Analyze this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}"
                    }
                }
            ]
        })

        db.add(ChatMemory(
            user_id=user.id,
            role="user",
            content=message or "[image]"
        ))

    else:
        messages.append({"role": "user", "content": message})

        db.add(ChatMemory(
            user_id=user.id,
            role="user",
            content=message
        ))

    # ---------------------------------------------
    # Generate AI response
    # ---------------------------------------------

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.6
    )

    reply = response.choices[0].message.content

    db.add(ChatMemory(
        user_id=user.id,
        role="assistant",
        content=reply
    ))

    # ---------------------------------------------
    # Usage deduction
    # ---------------------------------------------

    user.chat_used += 1
    db.commit()

    return {"reply": reply}


# -------------------------------------------------
<<<<<<< HEAD
# POSTER GENERATION (PROFESSIONAL STRUCTURED PROMPT)
=======
# POSTER GENERATION (FULL STRUCTURED PROMPT)
>>>>>>> 599c4e6 (Fixed persistent database and finalized backend)
# -------------------------------------------------

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

    if not title:
        raise HTTPException(400, "Title is required")

    image_size = SIZE_MAP.get(size, "1024x1536")

    # -------------------------------------------------
    # PROFESSIONAL STRUCTURED POSTER PROMPT
    # -------------------------------------------------

    prompt = f"""
Design a high-quality commercial marketing poster.

MAIN HEADLINE:
{title}

SUPPORTING MESSAGE:
{description}

VISUAL STYLE:
{style}

STRICT DESIGN REQUIREMENTS:
- Strong bold headline typography
- Clear visual hierarchy
- Professional layout composition
- Eye-catching modern marketing design
- Balanced spacing and alignment
- High contrast readability
- Proper text placement
- Realistic professional typography
- Clean background integration
- Suitable for social media and print
- Make it visually powerful and attention-grabbing
- It must look like a real professionally designed poster, not AI art

The result must be production-level professional.
"""

    # ---------------------------------------------
    # Generate image
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Watermark if required by plan
    # ---------------------------------------------

    if PLANS[user.plan]["watermark"]:
        im = Image.open(path).convert("RGBA")
        draw = ImageDraw.Draw(im)

        draw.text(
            (20, im.height - 40),
            "Acinyx.AI",
            fill=(255, 255, 255, 160)
        )

        im.save(path)

    # ---------------------------------------------
    # Usage deduction
    # ---------------------------------------------

    user.poster_used += 1
    db.commit()

    return {"poster_url": f"/outputs/{filename}"}


# -------------------------------------------------
# RUN SERVER
# -------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backendapp:app",
        host="0.0.0.0",
        port=8000
    )
