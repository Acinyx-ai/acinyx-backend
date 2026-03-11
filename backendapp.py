# ================= IMPORTS =================

from fastapi import FastAPI, HTTPException, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse, FileResponse

from pydantic import BaseModel, EmailStr

from sqlalchemy import Column, Integer, String, create_engine, or_, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.sql import func

from passlib.context import CryptContext

from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError, ExpiredSignatureError

import os
import base64
import uuid
import logging
import requests
import hmac
import hashlib

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

# Paystack Configuration
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY")
PAYSTACK_PUBLIC_KEY = os.getenv("PAYSTACK_PUBLIC_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

if not DATABASE_URL:
    raise Exception("DATABASE_URL missing")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY missing")
if not PAYSTACK_SECRET_KEY:
    raise Exception("PAYSTACK_SECRET_KEY missing")

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

PLAN_PRICES = {
    "basic": 25000,  # in kobo (250 KES = 25000 kobo)
    "pro": 50000,    # 500 KES = 50000 kobo
    "mega": 150000   # 1500 KES = 150000 kobo
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
    plan_expiry = Column(DateTime, nullable=True)
    chat_used = Column(Integer, default=0)
    image_used = Column(Integer, default=0)
    poster_used = Column(Integer, default=0)
    humanize_used = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())


class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    email = Column(String)
    plan = Column(String)
    amount = Column(Integer)
    reference = Column(String, unique=True)
    status = Column(String, default="pending")  # pending, success, failed
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


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
            "/payments/paystack/init",
            "/payments/paystack/webhook",
            "/payments/verify/{reference}",
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
    
    # Check if plan has expired
    if user.plan != "free" and user.plan_expiry and user.plan_expiry < datetime.utcnow():
        user.plan = "free"
        user.plan_expiry = None
        db.commit()
    
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
        "plan_expiry": user.plan_expiry.isoformat() if user.plan_expiry else None,
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


@app.post("/ai/image")
async def image(prompt: str = Form(...),
                user: User = Depends(get_current_user),
                db: Session = Depends(get_db)):
    try:
        if not check_usage_limit(user, "image"):
            raise HTTPException(429, "Image generation limit reached. Please upgrade your plan.")
        
        # Try DALL-E 3 first, fall back to DALL-E 2
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                n=1
            )
        except Exception as e:
            logger.warning(f"DALL-E 3 failed, falling back to DALL-E 2: {e}")
            response = client.images.generate(
                model="dall-e-2",
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


@app.post("/ai/poster")
async def poster(prompt: str = Form(...),
                 style: str = Form("realistic"),
                 user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    try:
        if not check_usage_limit(user, "poster"):
            raise HTTPException(429, "Poster generation limit reached. Please upgrade your plan.")
        
        enhanced_prompt = f"Create a professional poster: {prompt}. Style: {style}, high quality, marketing poster design"
        
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


# ================= PAYMENT ENDPOINTS =================

class PaymentInit(BaseModel):
    plan: str
    amount: int  # in kobo


@app.post("/payments/paystack/init")
async def init_payment(
    payment_data: PaymentInit,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Initialize Paystack payment"""
    try:
        if payment_data.plan not in ["basic", "pro", "mega"]:
            raise HTTPException(400, "Invalid plan")
        
        # Verify amount matches plan price
        expected_amount = PLAN_PRICES.get(payment_data.plan)
        if not expected_amount or payment_data.amount != expected_amount:
            raise HTTPException(400, "Invalid amount for selected plan")
        
        # Generate unique reference
        reference = f"ACINYX-{uuid.uuid4().hex[:8].upper()}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Create payment record
        payment = Payment(
            user_id=user.id,
            email=user.email,
            plan=payment_data.plan,
            amount=payment_data.amount,
            reference=reference,
            status="pending"
        )
        db.add(payment)
        db.commit()
        
        # Initialize Paystack transaction
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "email": user.email,
            "amount": payment_data.amount,
            "reference": reference,
            "callback_url": f"{FRONTEND_URL}/payment/verify?reference={reference}",
            "metadata": {
                "user_id": user.id,
                "username": user.username,
                "plan": payment_data.plan
            }
        }
        
        response = requests.post(
            "https://api.paystack.co/transaction/initialize",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            logger.error(f"Paystack init failed: {response.text}")
            raise HTTPException(500, "Failed to initialize payment")
        
        data = response.json()
        
        if not data.get("status"):
            raise HTTPException(500, data.get("message", "Payment initialization failed"))
        
        logger.info(f"✅ Payment initialized for user: {user.username}, reference: {reference}")
        
        return {
            "authorization_url": data["data"]["authorization_url"],
            "reference": reference,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment init error: {str(e)}")
        raise HTTPException(500, f"Payment initialization failed: {str(e)}")


@app.post("/payments/paystack/webhook")
async def paystack_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Paystack webhook"""
    try:
        # Get signature from header
        signature = request.headers.get("x-paystack-signature")
        
        # Get request body
        body = await request.body()
        
        # Verify webhook signature
        hash = hmac.new(
            PAYSTACK_SECRET_KEY.encode(),
            body,
            hashlib.sha512
        ).hexdigest()
        
        if signature != hash:
            logger.warning("Invalid webhook signature")
            return JSONResponse(status_code=400, content={"status": "invalid signature"})
        
        # Parse webhook data
        data = await request.json()
        
        event = data.get("event")
        webhook_data = data.get("data", {})
        
        if event == "charge.success":
            reference = webhook_data.get("reference")
            status = webhook_data.get("status")
            amount = webhook_data.get("amount")
            
            # Find payment record
            payment = db.query(Payment).filter(Payment.reference == reference).first()
            
            if not payment:
                logger.error(f"Payment not found for reference: {reference}")
                return JSONResponse(status_code=404, content={"status": "payment not found"})
            
            if status == "success":
                # Update payment status
                payment.status = "success"
                
                # Find user and upgrade plan
                user = db.query(User).filter(User.id == payment.user_id).first()
                if user:
                    # Set plan and expiry (30 days from now)
                    user.plan = payment.plan
                    user.plan_expiry = datetime.utcnow() + timedelta(days=30)
                    
                    # Reset usage counters for new billing period
                    user.chat_used = 0
                    user.image_used = 0
                    user.poster_used = 0
                    user.humanize_used = 0
                    
                    db.commit()
                    
                    logger.info(f"✅ User {user.username} upgraded to {payment.plan} plan")
            
            db.commit()
        
        return JSONResponse(status_code=200, content={"status": "success"})
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error"})


@app.get("/payments/verify/{reference}")
async def verify_payment(reference: str, db: Session = Depends(get_db)):
    """Verify payment status"""
    try:
        # Check local payment record
        payment = db.query(Payment).filter(Payment.reference == reference).first()
        
        if not payment:
            raise HTTPException(404, "Payment not found")
        
        # Verify with Paystack
        headers = {
            "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"
        }
        
        response = requests.get(
            f"https://api.paystack.co/transaction/verify/{reference}",
            headers=headers,
            timeout=10
        )
        
        if response.status_code != 200:
            logger.error(f"Paystack verify failed: {response.text}")
            raise HTTPException(500, "Failed to verify payment")
        
        data = response.json()
        
        if not data.get("status"):
            raise HTTPException(400, data.get("message", "Payment verification failed"))
        
        paystack_status = data["data"]["status"]
        
        # Update payment status if needed
        if paystack_status == "success" and payment.status != "success":
            payment.status = "success"
            
            # Update user plan
            user = db.query(User).filter(User.id == payment.user_id).first()
            if user:
                user.plan = payment.plan
                user.plan_expiry = datetime.utcnow() + timedelta(days=30)
                user.chat_used = 0
                user.image_used = 0
                user.poster_used = 0
                user.humanize_used = 0
            
            db.commit()
        
        return {
            "status": payment.status,
            "plan": payment.plan,
            "amount": payment.amount,
            "reference": payment.reference,
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment verification error: {str(e)}")
        raise HTTPException(500, f"Verification failed: {str(e)}")


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
    logger.info(f"💰 Paystack: {'configured' if PAYSTACK_SECRET_KEY else 'missing'}")
    logger.info(f"🌐 Base URL: {BASE_URL or 'not set'}")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"📁 Poster directory: {POSTER_DIR}")
    
    # Test OpenAI API connection
    try:
        client.models.list()
        logger.info("✅ OpenAI API connection successful")
    except Exception as e:
        logger.error(f"❌ OpenAI API connection failed: {e}")
    
    # Test Paystack API connection
    try:
        headers = {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"}
        response = requests.get("https://api.paystack.co/balance", headers=headers, timeout=10)
        if response.status_code == 200:
            logger.info("✅ Paystack API connection successful")
        else:
            logger.error(f"❌ Paystack API connection failed: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Paystack API connection failed: {e}")


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
