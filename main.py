import uvicorn
from app.core.env import settings

def main():
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)


if __name__ == "__main__":
    main()
