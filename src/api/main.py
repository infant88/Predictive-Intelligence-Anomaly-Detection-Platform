from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import config
from src.logger import logger
from src.database.connection import init_db
from src.api.routes import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {config.app_name} v{config.version}...")
    init_db()
    yield
    logger.info("Shutting down platform API...")

app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="Enterprise ML platform predicting business demand, detecting unusual patterns, and explaining prediction drivers.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {
        "app": config.app_name,
        "version": config.version,
        "docs_url": "/docs",
        "health_check": "/api/health"
    }
