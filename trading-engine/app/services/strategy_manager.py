import importlib
from typing import Type
from app.model.Strategy import Strategy


class StrategyManager:
    @staticmethod
    def get_strategy(strategy_name: str) -> Type[Strategy]:
        try:
            strategy_module = importlib.import_module(f"app.strategy.strategy")
            strategy_class = getattr(strategy_module, strategy_name)
            if not issubclass(strategy_class, Strategy):
                raise ValueError(f"{strategy_name} is not a valid Strategy subclass")
            return strategy_class
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Strategy {strategy_name} not found: {str(e)}")
