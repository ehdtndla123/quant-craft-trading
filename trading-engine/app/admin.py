from sqladmin import Admin, ModelView
from app.db.models import Bot, Strategy, TradingBot, Order, Trade, User, Backtesting
from app.db.database import engine


class BotAdmin(ModelView, model=Bot):
    column_list = [Bot.id, Bot.name, Bot.dry_run, Bot.cash]


class StrategyAdmin(ModelView, model=Strategy):
    column_list = [Strategy.id, Strategy.name, Strategy.description]


class TradingBotAdmin(ModelView, model=TradingBot):
    column_list = [TradingBot.id, TradingBot.bot_id, TradingBot.strategy_id]


class OrderAdmin(ModelView, model=Order):
    column_list = [Order.id, Order.size, Order.status]


class TradeAdmin(ModelView, model=Trade):
    column_list = [Trade.id, Trade.size, Trade.entry_price, Trade.exit_price, Trade.status]


class UserAdmin(ModelView, model=User):
    column_list = [User.id, User.username]


class BacktestingAdmin(ModelView, model=Backtesting):
    column_list = [Backtesting.id, Backtesting.strategy_name, Backtesting.start_date, Backtesting.end_date]


def setup_admin(app):
    admin = Admin(app, engine)
    admin.add_view(BotAdmin)
    admin.add_view(StrategyAdmin)
    admin.add_view(TradingBotAdmin)
    admin.add_view(OrderAdmin)
    admin.add_view(TradeAdmin)
    admin.add_view(UserAdmin)
    admin.add_view(BacktestingAdmin)
