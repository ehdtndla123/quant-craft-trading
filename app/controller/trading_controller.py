from fastapi import APIRouter, HTTPException, Depends
from app.service.trading import TradingService

router = APIRouter()

trading_service = None

def get_trading_service():
    global trading_service
    if trading_service is None:
        raise HTTPException(status_code=500, detail="Trading service not initialized")
    return trading_service

@router.post("/init")
async def init_trading_service(strategy_name: str, symbol: str, timeframe: str, initial_balance: float, is_dry_run: bool, api_key: str = None, secret_key: str = None):
    global trading_service
    trading_service = TradingService(strategy_name, symbol, timeframe, initial_balance, is_dry_run, api_key, secret_key)
    return {"message": "Trading service initialized successfully"}

@router.post("/start")
async def start_trading(service: TradingService = Depends(get_trading_service)):
    service.start()
    return {"message": "Trading started"}

@router.post("/stop")
async def stop_trading(service: TradingService = Depends(get_trading_service)):
    service.stop()
    return {"message": "Trading stopped"}

@router.get("/status")
async def get_trading_status(service: TradingService = Depends(get_trading_service)):
    return service.get_status()

@router.post("/close")
async def close_trading_service(service: TradingService = Depends(get_trading_service)):
    await service.close()
    global trading_service
    trading_service = None
    return {"message": "Trading service closed"}