# main.py dosyasının geçici test içeriği

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "UniAsistan API çalışıyor!"}