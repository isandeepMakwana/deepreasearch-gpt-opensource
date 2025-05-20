import uvicorn
from .api import app

if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8001, reload=True)
