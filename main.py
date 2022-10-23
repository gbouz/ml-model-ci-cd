from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def get_items():
    return {
        "msg": "Called inference API root.",
        "success": True,
    }