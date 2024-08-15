from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routes import trading_bot, strategy, bot, backtesting
from app.core.config import settings
from app.db.database import initialize_database
from app.services.trading_engine_manager import trading_engine_manager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")
app.include_router(trading_bot.router)
app.include_router(strategy.router)
app.include_router(bot.router)
app.include_router(backtesting.router)


@app.on_event("startup")
async def startup_event():
    print("Starting database initialization...")
    initialize_database()
    print("Database initialization completed")
    await trading_engine_manager.run()


@app.on_event("shutdown")
async def shutdown_event():
    print("Stopping all bots...")
    await trading_engine_manager.stop_all_bots()
    trading_engine_manager.shutdown()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
