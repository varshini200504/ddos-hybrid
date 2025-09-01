from fastapi import FastAPI

# This must be named exactly "app"
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
