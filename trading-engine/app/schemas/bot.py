from pydantic import BaseModel
from datetime import datetime


class BotBase(BaseModel):
    dry_run: bool = True
    name: str
    cash: float


class BotCreate(BotBase):
    pass


class BotUpdate(BotBase):
    pass


class BotInDB(BotBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
