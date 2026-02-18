from fastapi import FastAPI,HTTPException,Depends,Form,Request,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm

from pydantic import BaseModel

from sqlalchemy import Column,Integer,String,create_engine,or_,Text,ForeignKey,DateTime
from sqlalchemy.orm import sessionmaker,declarative_base,Session

from passlib.context import CryptContext

from datetime import datetime,timedelta

from jose import jwt,JWTError

import os,base64,requests,hmac,hashlib,uuid,logging

from openai import OpenAI

from PIL import Image,ImageDraw

import uvicorn


# CONFIG

logging.basicConfig(level=logging.INFO)

logger=logging.getLogger("acinyx")

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

DATABASE_URL=os.getenv("DATABASE_URL")

PAYSTACK_SECRET_KEY=os.getenv("PAYSTACK_SECRET_KEY")

JWT_SECRET=os.getenv("JWT_SECRET","CHANGE_THIS_SECRET")

JWT_ALGORITHM="HS256"

JWT_EXPIRE_DAYS=30


if DATABASE_URL.startswith("postgres://"):

    DATABASE_URL=DATABASE_URL.replace("postgres://","postgresql://",1)


client=OpenAI(api_key=OPENAI_API_KEY)



# DATABASE

engine=create_engine(DATABASE_URL,pool_pre_ping=True,pool_recycle=300)

SessionLocal=sessionmaker(bind=engine)

Base=declarative_base()



class User(Base):

    __tablename__="users"

    id=Column(Integer,primary_key=True)

    username=Column(String,unique=True)

    email=Column(String,unique=True)

    password_hash=Column(String)

    plan=Column(String,default="free")

    chat_used=Column(Integer,default=0)

    poster_used=Column(Integer,default=0)



class ChatMemory(Base):

    __tablename__="chat_memory"

    id=Column(Integer,primary_key=True)

    user_id=Column(Integer)

    role=Column(String)

    content=Column(Text)

    created_at=Column(DateTime,default=datetime.utcnow)



Base.metadata.create_all(bind=engine)



# APP

app=FastAPI()

app.add_middleware(

CORSMiddleware,

allow_origins=["*"],

allow_methods=["*"],

allow_headers=["*"],

allow_credentials=True

)


os.makedirs("outputs",exist_ok=True)

app.mount("/outputs",StaticFiles(directory="outputs"),name="outputs")



# SECURITY

pwd_context=CryptContext(schemes=["bcrypt"])

oauth2_scheme=OAuth2PasswordBearer(tokenUrl="token")



def get_db():

    db=SessionLocal()

    try:

        yield db

    finally:

        db.close()



def create_access_token(data):

    expire=datetime.utcnow()+timedelta(days=JWT_EXPIRE_DAYS)

    data.update({"exp":expire})

    return jwt.encode(data,JWT_SECRET,algorithm=JWT_ALGORITHM)



def get_current_user(token:str=Depends(oauth2_scheme),db:Session=Depends(get_db)):

    try:

        payload=jwt.decode(token,JWT_SECRET,algorithms=[JWT_ALGORITHM])

        username=payload.get("sub")

    except:

        raise HTTPException(401)


    user=db.query(User).filter(User.username==username).first()

    if not user:

        raise HTTPException(401)

    return user




# CHAT WITH IMAGE + HUMANIZER SUPPORT

@app.post("/ai/chat")

async def chat(

message:str=Form(""),

mode:str=Form("chat"),

image:UploadFile|None=File(None),

user:User=Depends(get_current_user),

db:Session=Depends(get_db)

):


if user.chat_used>=50:

    raise HTTPException(403,"Limit reached")



reply=None

image_path=None



# IMAGE GENERATION MODE

if mode=="image":

    img=client.images.generate(

    model="gpt-image-1",

    prompt=message,

    size="1024x1024"

    )



    image_bytes=base64.b64decode(img.data[0].b64_json)

    filename=f"{uuid.uuid4()}.png"

    path=f"outputs/{filename}"


    with open(path,"wb") as f:

        f.write(image_bytes)


    image_path=path

    reply="Here is your image."



# HUMANIZER MODE

elif mode=="humanize":

    res=client.chat.completions.create(

    model="gpt-4.1-mini",

    messages=[

    {

    "role":"system",

    "content":"Rewrite this text to sound completely human and natural."

    },

    {"role":"user","content":message}

    ]

    )



    reply=res.choices[0].message.content



# NORMAL CHAT

else:

    res=client.chat.completions.create(

    model="gpt-4.1-mini",

    messages=[{"role":"user","content":message}]

    )


    reply=res.choices[0].message.content



# SAVE MEMORY

db.add(ChatMemory(user_id=user.id,role="user",content=message))

db.add(ChatMemory(user_id=user.id,role="assistant",content=reply))

user.chat_used+=1

db.commit()



return {

"reply":reply,

"image":image_path

}




# POSTER

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


with open(path,"wb") as f:

    f.write(image_bytes)



user.poster_used+=1

db.commit()



return {"image":path}




@app.get("/health")

def health():

    return {"status":"ok"}




if __name__=="__main__":

    uvicorn.run("backendapp:app",host="0.0.0.0",port=8000)
