import sys
print("sys.path =", sys.path)

import uvicorn
from src.api import app

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8080, reload=True)
