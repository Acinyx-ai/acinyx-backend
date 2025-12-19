from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv
from passlib.context import CryptContext

import uvicorn
import os
import jwt
import time
import shutil
import requests

# -------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "change_this_secret")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠ WARNING: OPENAI_API_KEY not set. AI will run in ECHO mode.")

# -------------------------------------------------
# PASSWORD HASHING
# -------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

# -------------------------------------------------
# APP INIT
# -------------------------------------------------
app = FastAPI(
    title="Acinyx.AI API",
    version="0.1.0",
    description="Backend API for Acinyx AI services"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------------------------------------------------
# ROOT
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Acinyx.AI backend",
        "message": "Backend is live 🚀",
        "version": "0.1.0"
    }

# -------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Acinyx.AI backend",
        "version": "0.1.0"
    }

# -------------------------------------------------
# IN-MEMORY DATABASE (MVP)
# -------------------------------------------------
USERS = {
    "demo": {
        "password": hash_password("password123")
    }
}

CREDITS = {
    "demo": 20
}

# -------------------------------------------------
# MODELS
# -------------------------------------------------
class User(BaseModel):
    username: str
    password: str

class TokenResp(BaseModel):
    access_token: str
    token_type: str = "bearer"

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def create_token(username: str):
    payload = {
        "sub": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 24  # 24h
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        if username in USERS:
            return username
        raise HTTPException(status_code=401, detail="Invalid user")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# -------------------------------------------------
# AUTH
# -------------------------------------------------
@app.post("/signup")
def signup(user: User):
    if user.username in USERS:
        raise HTTPException(status_code=400, detail="User exists")

    USERS[user.username] = {
        "password": hash_password(user.password)
    }
    CREDITS[user.username] = 20
    return {"ok": True, "message": "User created"}

@app.post("/token", response_model=TokenResp)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token(form_data.username)
    return {"access_token": token}

# -------------------------------------------------
# USER INFO
# -------------------------------------------------
@app.get("/me")
def me(user: str = Depends(get_current_user)):
    return {
        "username": user,
        "credits": CREDITS.get(user, 0)
    }

# -------------------------------------------------
# AI CHAT
# -------------------------------------------------
@app.post("/ai/chat")
def ai_chat(payload: dict, user: str = Depends(get_current_user)):
    if CREDITS.get(user, 0) < 1:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty prompt")

    CREDITS[user] -= 1

    if not OPENAI_API_KEY:
        return {"response": f"ECHO: {text}", "credits_left": CREDITS[user]}

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": text}]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=30
    )

    if r.status_code == 429:
        raise HTTPException(status_code=429, detail="AI busy. Try again later.")
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="OpenAI API error")

    content = r.json()["choices"][0]["message"]["content"]
    return {"response": content, "credits_left": CREDITS[user]}

# -------------------------------------------------
# WHATSAPP AGENT (SIMULATION)
# -------------------------------------------------
@app.post("/agent/whatsapp")
def whatsapp_agent(payload: dict, user: str = Depends(get_current_user)):
    if CREDITS.get(user, 0) < 1:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    message = payload.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    CREDITS[user] -= 1

    return {
        "reply": f"🤖 Acinyx WhatsApp Agent received: '{message}'",
        "credits_left": CREDITS[user]
    }

# -------------------------------------------------
# AI IMAGE
# -------------------------------------------------
@app.post("/ai/image")
def ai_image(payload: dict, user: str = Depends(get_current_user)):
    if CREDITS.get(user, 0) < 2:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    prompt = payload.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    CREDITS[user] -= 2

    if not OPENAI_API_KEY:
        return {
            "image_url": "https://placehold.co/600x400?text=Acinyx+AI+Image",
            "credits_left": CREDITS[user]
        }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {"prompt": prompt, "size": "1024x1024"}

    r = requests.post(
        "https://api.openai.com/v1/images/generations",
        json=body,
        headers=headers,
        timeout=30
    )

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="Image API error")

    url = r.json()["data"][0]["url"]
    return {"image_url": url, "credits_left": CREDITS[user]}

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
@app.post("/upload")
def upload(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    path = os.path.join(uploads_dir, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"url": f"/uploads/{file.filename}"}

# -------------------------------------------------
# RUN (LOCAL ONLY)
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("backendapp:app", host="0.0.0.0", port=port)
