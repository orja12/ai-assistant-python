from fastapi import FastAPI
from routers import ocr

app = FastAPI(title="AI Assistant App")

# ربط الروترات هنا
app.include_router(ocr.router)

@app.get("/")
def home():
    return {"message": "AI app is running!"}
