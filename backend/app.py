from fastapi import FastAPI

app = FastAPI(title="AI Assistant App")

@app.get("/")
def home():
    return {"message": "AI app is running!"}
