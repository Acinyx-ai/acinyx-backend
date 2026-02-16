from fastapi import FastAPI,HTTPException,Depends,UploadFile,File,Form,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm
from pydantic import BaseModel

from sqlalchemy import Column,Integer,String,create_engine,or_,Text,ForeignKey,DateTime
from sqlalchemy.orm import sessionmaker,declarative_base,Session

from passlib.context import CryptContext
from datetime import datetime,timedelta
from jose import jwt,JWTError

import os,time,base64,logging,requests,hmac,hashlib,uuid

from openai import OpenAI
from PIL import Image,ImageDraw
import uvicorn

# CONFIG

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger("acinyx")

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PAYSTACK_SECRET_KEY=os.getenv("PAYSTACK_SECRET_KEY")
NEWS_API_KEY=os.getenv("NEWS_API_KEY")

JWT_SECRET=os.getenv("JWT_SECRET","CHANGE_THIS_SECRET")
JWT_ALGORITHM="HS256"
JWT_EXPIRE_MINUTES=60*24

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

if not PAYSTACK_SECRET_KEY:
    raise RuntimeError("PAYSTACK_SECRET_KEY not set")

client=OpenAI()

# DATABASE (PostgreSQL + SQLite fallback)

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH=os.path.join(BASE_DIR,"acinyx.db")

DATABASE_URL=os.getenv("DATABASE_URL")

if DATABASE_URL:

    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL=DATABASE_URL.replace("postgres://","postgresql://",1)

    logger.info("Using PostgreSQL")

    engine=create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )

else:

    logger.info("Using SQLite")

    engine=create_engine(
        f"sqlite:///{SQLITE_PATH}",
        connect_args={"check_same_thread":False},
        pool_pre_ping=True
    )

SessionLocal=sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base=declarative_base()

# MODELS

class User(Base):
    __tablename__="users"
    id=Column(Integer,primary_key=True,index=True)
    username=Column(String(100),unique=True,index=True,nullable=False)
    email=Column(String(150),unique=True,index=True,nullable=False)
    password_hash=Column(String,nullable=False)
    plan=Column(String,default="free")
    chat_used=Column(Integer,default=0)
    poster_used=Column(Integer,default=0)

class ChatMemory(Base):
    __tablename__="chat_memory"
    id=Column(Integer,primary_key=True,index=True)
    user_id=Column(Integer,ForeignKey("users.id"),index=True)
    role=Column(String(20))
    content=Column(Text)
    created_at=Column(DateTime,default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# APP

app=FastAPI(title="Acinyx.AI Backend",version="6.0.0")

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# STATIC

os.makedirs("outputs",exist_ok=True)
app.mount("/outputs",StaticFiles(directory="outputs"),name="outputs")

# SECURITY

pwd_context=CryptContext(schemes=["bcrypt"],deprecated="auto")
oauth2_scheme=OAuth2PasswordBearer(tokenUrl="token")

# HELPERS

def hash_password(password:str):
    return pwd_context.hash(password)

def verify_password(password:str,hashed:str):
    return pwd_context.verify(password,hashed)

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data:dict):
    expire=datetime.utcnow()+timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode=data.copy()
    to_encode.update({"exp":expire})
    return jwt.encode(to_encode,JWT_SECRET,algorithm=JWT_ALGORITHM)

def get_current_user(token:str=Depends(oauth2_scheme),db:Session=Depends(get_db)):
    try:
        payload=jwt.decode(token,JWT_SECRET,algorithms=[JWT_ALGORITHM])
        username=payload.get("sub")
        if not username:
            raise HTTPException(401,"Invalid token")
    except JWTError:
        raise HTTPException(401,"Invalid token")

    user=db.query(User).filter(User.username==username).first()

    if not user:
        raise HTTPException(401,"Invalid token")

    return user
# PLANS

PLANS={
    "free":{"chat":5,"poster":2,"watermark":True},
    "basic":{"chat":100,"poster":20,"watermark":False},
    "pro":{"chat":500,"poster":100,"watermark":False},
    "mega":{"chat":2000,"poster":300,"watermark":False},
}

# AUTH

class SignupBody(BaseModel):
    username:str
    email:str
    password:str

@app.post("/signup")
def signup(data:SignupBody,db:Session=Depends(get_db)):

    if db.query(User).filter(User.username==data.username).first():
        raise HTTPException(400,"User exists")

    if db.query(User).filter(User.email==data.email).first():
        raise HTTPException(400,"Email already registered")

    user=User(
        username=data.username,
        email=data.email,
        password_hash=hash_password(data.password)
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"New user created: {user.username}")

    return {"message":"Account created"}

@app.post("/token")
def login(form:OAuth2PasswordRequestForm=Depends(),db:Session=Depends(get_db)):

    user=db.query(User).filter(
        or_(User.username==form.username,User.email==form.username)
    ).first()

    if not user or not verify_password(form.password,user.password_hash):
        raise HTTPException(401,"Invalid credentials")

    token=create_access_token({"sub":user.username})

    return{
        "access_token":token,
        "token_type":"bearer",
        "plan":user.plan
    }

# PAYSTACK INIT

class PaystackInitBody(BaseModel):
    amount:int
    plan:str

@app.post("/payments/paystack/init")
def init_paystack_payment(body:PaystackInitBody,user:User=Depends(get_current_user)):

    if body.plan not in PLANS:
        raise HTTPException(400,"Invalid plan")

    payload={
        "email":user.email,
        "amount":body.amount,
        "callback_url":"https://acinyx-ai.vercel.app/dashboard",
        "metadata":{
            "username":user.username,
            "plan":body.plan
        }
    }

    r=requests.post(
        "https://api.paystack.co/transaction/initialize",
        json=payload,
        headers={
            "Authorization":f"Bearer {PAYSTACK_SECRET_KEY}",
            "Content-Type":"application/json"
        },
        timeout=20
    )

    data=r.json()

    if not data.get("status"):
        raise HTTPException(400,data.get("message"))

    return{
        "authorization_url":data["data"]["authorization_url"],
        "reference":data["data"]["reference"]
    }

# PAYSTACK WEBHOOK

@app.post("/payments/paystack/webhook")
async def paystack_webhook(request:Request,db:Session=Depends(get_db)):

    raw_body=await request.body()
    signature=request.headers.get("x-paystack-signature")

    expected=hmac.new(
        PAYSTACK_SECRET_KEY.encode(),
        raw_body,
        hashlib.sha512
    ).hexdigest()

    if not signature or not hmac.compare_digest(expected,signature):
        raise HTTPException(400,"Invalid signature")

    payload=await request.json()

    if payload.get("event")!="charge.success":
        return{"status":"ignored"}

    metadata=payload.get("data",{}).get("metadata",{})
    username=metadata.get("username")
    plan=metadata.get("plan")

    if not username or not plan:
        logger.error("Missing metadata")
        return{"status":"error"}

    user=db.query(User).filter(User.username==username).first()

    if not user:
        logger.error("User not found")
        return{"status":"error"}

    if plan not in PLANS:
        logger.error("Invalid plan")
        return{"status":"error"}

    user.plan=plan
    user.chat_used=0
    user.poster_used=0

    db.commit()

    logger.info(f"Plan upgraded {username}->{plan}")

    return{"status":"ok"}

# CHAT

MAX_HISTORY=12

@app.post("/ai/chat")
async def ai_chat(
    message:str=Form(None),
    image:UploadFile=File(None),
    user:User=Depends(get_current_user),
    db:Session=Depends(get_db)
):

    if user.chat_used>=PLANS[user.plan]["chat"]:
        raise HTTPException(403,"Chat limit reached")

    if not message and not image:
        raise HTTPException(400,"Message required")

    history=db.query(ChatMemory)\
        .filter(ChatMemory.user_id==user.id)\
        .order_by(ChatMemory.created_at.desc())\
        .limit(MAX_HISTORY)\
        .all()

    history.reverse()

    messages=[{
        "role":"system",
        "content":"You are Acinyx.AI"
    }]

    for h in history:
        messages.append({
            "role":h.role,
            "content":h.content
        })

    if image:

        image_bytes=await image.read()
        encoded=base64.b64encode(image_bytes).decode()

        messages.append({
            "role":"user",
            "content":[
                {"type":"text","text":message or "Describe image"},
                {
                    "type":"image_url",
                    "image_url":{
                        "url":f"data:{image.content_type};base64,{encoded}"
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

        messages.append({
            "role":"user",
            "content":message
        })

        db.add(ChatMemory(
            user_id=user.id,
            role="user",
            content=message
        ))

    response=client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )

    reply=response.choices[0].message.content

    db.add(ChatMemory(
        user_id=user.id,
        role="assistant",
        content=reply
    ))

    user.chat_used+=1
    db.commit()

    image_url=None

    if reply and "/outputs/" in reply:
        start=reply.find("/outputs/")
        end=reply.find(".png",start)
        if end!=-1:
            image_url=reply[start:end+4]

    return{
        "reply":reply,
        "image":image_url
    }

# POSTER GENERATION

SIZE_MAP={
    "portrait":"1024x1536",
    "square":"1024x1024",
    "landscape":"1536x1024"
}

@app.post("/ai/poster/ai-generate")
async def ai_poster(
    title:str=Form(...),
    description:str=Form(""),
    style:str=Form("modern"),
    size:str=Form("portrait"),
    user:User=Depends(get_current_user),
    db:Session=Depends(get_db)
):

    if user.poster_used>=PLANS[user.plan]["poster"]:
        raise HTTPException(403,"Limit reached")

    prompt=f"""
Create professional marketing poster
Title:{title}
Description:{description}
Style:{style}
"""

    img=client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=SIZE_MAP.get(size,"1024x1536")
    )

    image_bytes=base64.b64decode(img.data[0].b64_json)

    filename=f"{uuid.uuid4()}.png"
    path=f"outputs/{filename}"

    with open(path,"wb") as f:
        f.write(image_bytes)

    user.poster_used+=1
    db.commit()

    return{
        "reply":f"/outputs/{filename}",
        "image":f"/outputs/{filename}"
    }

# NEWS

@app.get("/news")
def news():

    if not NEWS_API_KEY:
        return[]

    r=requests.get(
        "https://newsapi.org/v2/top-headlines",
        params={
            "country":"us",
            "apiKey":NEWS_API_KEY
        }
    )

    return r.json()

# HEALTH

@app.get("/health")
def health():
    return{"status":"ok"}

# SERVER

if __name__=="__main__":

    uvicorn.run(
        "backendapp:app",
        host="0.0.0.0",
        port=8000
    )
