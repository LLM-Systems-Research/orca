from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from enum import Enum
from dataclasses import dataclass
from scheduler import Scheduler
import pandas as pd

app = FastAPI()

scheduler = Scheduler()


### Pydantic models
class Prompt_Request(BaseModel):
    prompt: str
    
class Batch_Prompt_Request(BaseModel):
    prompts: list[str]
    
@app.post("/generate")
def read_root(request: Prompt_Request):
    scheduler.add_request(prompt=request.prompt)
    return {"Hello World"}

@app.post("/batch_process")
def process_request(request: Batch_Prompt_Request):
    # Process the request here
    scheduler.add_request_batch(request.prompts)
    return {"message": "Request processed successfully"}
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)