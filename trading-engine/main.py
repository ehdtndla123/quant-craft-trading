from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import trading_bot, strategy, backtesting
from app.db.database import initialize_database, tunnel
from app.services.trading_engine_manager import trading_engine_manager
from app.admin import setup_admin
from app.db.database_manager import db_manager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://quant-craft.site"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trading_bot.router)
app.include_router(strategy.router)
app.include_router(backtesting.router)
setup_admin(app)


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
    import atexit

    atexit.register(db_manager.stop_tunnel)
    uvicorn.run(app, host="0.0.0.0", port=8000)
