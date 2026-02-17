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
import os,base64,logging,requests,hmac,hashlib,uuid
from openai import OpenAI
from PIL import Image,ImageDraw
import uvicorn

# ---------------- CONFIG ----------------

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger("acinyx")

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PAYSTACK_SECRET_KEY=os.getenv("PAYSTACK_SECRET_KEY")
NEWS_API_KEY=os.getenv("NEWS_API_KEY")

JWT_SECRET=os.getenv("JWT_SECRET","CHANGE_THIS_SECRET")
JWT_ALGORITHM="HS256"
JWT_EXPIRE_DAYS=30

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

if not PAYSTACK_SECRET_KEY:
    raise RuntimeError("PAYSTACK_SECRET_KEY not set")

client=OpenAI()

# ---------------- DATABASE ----------------

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH=os.path.join(BASE_DIR,"acinyx.db")
DATABASE_URL=os.getenv("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL=DATABASE_URL.replace("postgres://","postgresql://",1)

    engine=create_engine(DATABASE_URL,pool_pre_ping=True)

else:
    engine=create_engine(
        f"sqlite:///{SQLITE_PATH}",
        connect_args={"check_same_thread":False},
        pool_pre_ping=True
    )

SessionLocal=sessionmaker(bind=engine,autoflush=False,autocommit=False)
Base=declarative_base()

# ---------------- MODELS ----------------

class User(Base):

    __tablename__="users"

    id=Column(Integer,primary_key=True,index=True)
    username=Column(String(100),unique=True,index=True)
    email=Column(String(150),unique=True,index=True)
    password_hash=Column(String)

    plan=Column(String,default="free")

    chat_used=Column(Integer,default=0)
    poster_used=Column(Integer,default=0)


class ChatMemory(Base):

    __tablename__="chat_memory"

    id=Column(Integer,primary_key=True,index=True)

    user_id=Column(Integer,ForeignKey("users.id"))

    role=Column(String(20))
    content=Column(Text)

    created_at=Column(DateTime,default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# ---------------- APP ----------------

app=FastAPI(title="Acinyx.AI Backend",version="FINAL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

os.makedirs("outputs",exist_ok=True)
app.mount("/outputs",StaticFiles(directory="outputs"),name="outputs")

# ---------------- SECURITY ----------------

pwd_context=CryptContext(schemes=["bcrypt"],deprecated="auto")
oauth2_scheme=OAuth2PasswordBearer(tokenUrl="token")

def hash_password(p:str):
    return pwd_context.hash(p)

def verify_password(p:str,h:str):
    return pwd_context.verify(p,h)


def get_db():

    db=SessionLocal()

    try:
        yield db
    finally:
        db.close()


def create_access_token(data:dict):

    expire=datetime.utcnow()+timedelta(days=JWT_EXPIRE_DAYS)

    data.update({"exp":expire})

    return jwt.encode(data,JWT_SECRET,algorithm=JWT_ALGORITHM)


def get_current_user(token:str=Depends(oauth2_scheme),db:Session=Depends(get_db)):

    try:
        payload=jwt.decode(token,JWT_SECRET,algorithms=[JWT_ALGORITHM])
        username=payload.get("sub")

    except JWTError:
        raise HTTPException(401,"Invalid token")

    user=db.query(User).filter(User.username==username).first()

    if not user:
        raise HTTPException(401,"Invalid token")

    return user


# ---------------- PLANS ----------------

PLANS={

"free":{"chat":50,"poster":2,"watermark":True},

"basic":{"chat":100,"poster":20,"watermark":False},

"pro":{"chat":500,"poster":100,"watermark":False},

"mega":{"chat":2000,"poster":300,"watermark":False},

}


# ---------------- SIGNUP ----------------

class SignupBody(BaseModel):

    username:str
    email:str
    password:str


@app.post("/signup")

def signup(data:SignupBody,db:Session=Depends(get_db)):

    if db.query(User).filter(User.username==data.username).first():

        raise HTTPException(400,"User exists")


    user=User(

        username=data.username,

        email=data.email,

        password_hash=hash_password(data.password)

    )

    db.add(user)

    db.commit()

    db.refresh(user)

    return{"message":"Account created"}


# ---------------- LOGIN ----------------

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


# ---------------- PAYSTACK INIT ----------------

class PaystackInitBody(BaseModel):

    amount:int

    plan:str


@app.post("/payments/paystack/init")

def init_payment(body:PaystackInitBody,user:User=Depends(get_current_user)):

    if body.plan not in PLANS:

        raise HTTPException(400,"Invalid plan")

    payload={

        "email":user.email,

        "amount":body.amount,

        "callback_url":"https://acinyx-frontend.vercel.app/dashboard",

        "metadata":{"username":user.username,"plan":body.plan}

    }

    r=requests.post(

        "https://api.paystack.co/transaction/initialize",

        json=payload,

        headers={"Authorization":f"Bearer {PAYSTACK_SECRET_KEY}"}

    )

    return r.json()["data"]


# ---------------- PAYSTACK WEBHOOK ----------------

@app.post("/payments/paystack/webhook")

async def webhook(request:Request,db:Session=Depends(get_db)):

    raw=await request.body()

    sig=request.headers.get("x-paystack-signature")

    expected=hmac.new(

        PAYSTACK_SECRET_KEY.encode(),

        raw,

        hashlib.sha512

    ).hexdigest()

    if not sig or not hmac.compare_digest(expected,sig):

        raise HTTPException(400,"Invalid signature")


    payload=await request.json()

    meta=payload["data"]["metadata"]

    user=db.query(User).filter(User.username==meta["username"]).first()

    user.plan=meta["plan"]

    user.chat_used=0

    user.poster_used=0

    db.commit()

    return{"status":"ok"}


# ---------------- CHAT ----------------

MAX_HISTORY=12


@app.post("/ai/chat")

async def chat(

    message:str=Form(...),

    user:User=Depends(get_current_user),

    db:Session=Depends(get_db)

):

    if user.chat_used>=PLANS[user.plan]["chat"]:

        raise HTTPException(403,"Limit reached")


    history=db.query(ChatMemory).filter(

        ChatMemory.user_id==user.id

    ).limit(MAX_HISTORY).all()


    msgs=[{"role":"system","content":"You are Acinyx.AI"}]


    for h in history:

        msgs.append({"role":h.role,"content":h.content})


    msgs.append({"role":"user","content":message})


    if "generate image" in message.lower():

        img=client.images.generate(

            model="gpt-image-1",

            prompt=message,

            size="1024x1024"

        )

        image_bytes=base64.b64decode(img.data[0].b64_json)

        filename=f"{uuid.uuid4()}.png"

        path=f"outputs/{filename}"

        open(path,"wb").write(image_bytes)

        reply=f"/outputs/{filename}"

    else:

        res=client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=msgs

        )

        reply=res.choices[0].message.content


    db.add(ChatMemory(user_id=user.id,role="user",content=message))

    db.add(ChatMemory(user_id=user.id,role="assistant",content=reply))


    user.chat_used+=1

    db.commit()


    return{"reply":reply}


# ---------------- POSTER ----------------

@app.post("/ai/poster")

async def poster(

    title:str=Form(...),

    user:User=Depends(get_current_user),

    db:Session=Depends(get_db)

):

    img=client.images.generate(

        model="gpt-image-1",

        prompt=title,

        size="1024x1536"

    )

    image_bytes=base64.b64decode(img.data[0].b64_json)

    filename=f"{uuid.uuid4()}.png"

    path=f"outputs/{filename}"

    open(path,"wb").write(image_bytes)


    if PLANS[user.plan]["watermark"]:

        im=Image.open(path)

        draw=ImageDraw.Draw(im)

        draw.text((20,20),"Acinyx.AI")

        im.save(path)


    user.poster_used+=1

    db.commit()


    return{"image":path}


# ---------------- HEALTH ----------------

@app.get("/health")

def health():

    return{"status":"ok"}


# ---------------- SERVER ----------------

if __name__=="__main__":

    uvicorn.run(

        "backendapp:app",

        host="0.0.0.0",

        port=8000

    )
