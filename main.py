import uvicorn
from routes import *
from middleware.middleware import add_middleware
from fastapi import FastAPI

app = FastAPI()

app.middleware("http")(add_middleware)
app.include_router(index_router)
app.include_router(summarization_router)

if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000, reload=True)