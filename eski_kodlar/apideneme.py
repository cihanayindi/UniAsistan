from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # veya ["http://localhost"] gibi kısıtlayabilirsin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/message")
async def receive_message(data: dict):
    message = data.get("message", "")
    return {"reply": f"Sunucu mesajını aldı: '{message}'"}
