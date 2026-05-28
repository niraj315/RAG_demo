from dotenv import load_dotenv
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from rag import ask

load_dotenv()

app =FastAPI(
    title= "RAG CHATBOT",
    description="Ask question about your pdf document.",
    version="1.0.0",
)

#Request/Response Schemas

class chatRequest(BaseModel):
    question : str

class chatRespnse(BaseModel):
    answer :str
    sources : list[str]

@app.get("/")
def root():
    return {"messaage" : "RAG chatbot is running. POST to /chat to ask a question."}

@app.post("/chat", response_model=chatRespnse)
def chat(request: chatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400,detail="Questin can not be empty")
    
    result = ask(request.question)
    return chatRespnse(
        answer = result["answer"],
        sources= result["sources"]
    )