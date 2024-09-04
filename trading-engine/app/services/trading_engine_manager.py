import asyncio
import multiprocessing
from multiprocessing import Queue
import queue
import time
from typing import Dict, List
from app.services.trading_engine import TradingEngine
from app.services import trading_bot_service
from app.db.database import SessionLocal
from app.db.models import TradingBotStatus


def worker_process(worker_id: int, task_queue: Queue, result_queue: Queue):
    asyncio.run(async_worker(worker_id, task_queue, result_queue))


async def async_worker(worker_id: int, task_queue: Queue, result_queue: Queue):
    bot_processes = {}
    bot_engines = {}
    last_heartbeat = time.time()
    while True:
        try:
            trading_bot_id, action, params = task_queue.get(timeout=1)
        except queue.Empty:
            await asyncio.sleep(1)
            if time.time() - last_heartbeat > 30:
                result_queue.put(("heartbeat", None, worker_id))
                last_heartbeat = time.time()
            continue

        if action == "start":
            trading_bot = params
            engine = TradingEngine(trading_bot)
            bot_engines[trading_bot_id] = engine
            task = asyncio.create_task(engine.run())
            bot_processes[trading_bot_id] = task
            result_queue.put(("started", trading_bot_id, worker_id))
        elif action == "stop":
            if trading_bot_id in bot_processes:
                bot_processes[trading_bot_id].cancel()
                await bot_processes[trading_bot_id]
                del bot_processes[trading_bot_id]
                del bot_engines[trading_bot_id]
                result_queue.put(("stopped", trading_bot_id, worker_id))
        elif action == "status":
            is_running = trading_bot_id in bot_processes and not bot_processes[trading_bot_id].done()
            result_queue.put(
                ("status", trading_bot_id, TradingBotStatus.RUNNING if is_running else TradingBotStatus.STOPPED))
        elif action == "get_engine_data":
            if trading_bot_id in bot_engines:
                engine = bot_engines[trading_bot_id]
                engine_data = {
                    "equity": engine.broker_service.equity,
                    "last_price": engine.broker_service.last_price,
                }
                result_queue.put(("engine_data", trading_bot_id, engine_data))
            else:
                result_queue.put(("engine_data", trading_bot_id, None))


class TradingEngineManager:
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.task_queues: List[Queue] = [Queue() for _ in range(self.num_workers)]
        self.result_queue = Queue()
        self.workers: List[multiprocessing.Process] = []
        self.bot_worker_map: Dict[int, int] = {}
        self.worker_last_heartbeat: Dict[int, float] = {}

    def start_workers(self):
        for i in range(self.num_workers):
            self._start_worker(i)

    def _start_worker(self, worker_id: int):
        worker = multiprocessing.Process(target=worker_process,
                                         args=(worker_id, self.task_queues[worker_id], self.result_queue))
        worker.start()
        self.workers.append(worker)
        self.worker_last_heartbeat[worker_id] = time.time()


    async def start_bot(self, trading_bot_id: int):
        db = SessionLocal()
        try:
            trading_bot = trading_bot_service.get_trading_bot_with_relations(db, trading_bot_id)
            if trading_bot and trading_bot.status == TradingBotStatus.PENDING:
                worker_id = hash(trading_bot.id) % self.num_workers
                self.task_queues[worker_id].put((trading_bot.id, "start", trading_bot))
                trading_bot_service.update_trading_bot_status(db, trading_bot.id, TradingBotStatus.RUNNING)
                self.bot_worker_map[trading_bot.id] = worker_id
        finally:
            db.close()

    async def stop_bot(self, trading_bot_id: int):
        db = SessionLocal()
        try:
            if trading_bot_id in self.bot_worker_map:
                worker_id = self.bot_worker_map[trading_bot_id]
                self.task_queues[worker_id].put((trading_bot_id, "stop", None))
                trading_bot_service.update_trading_bot_status(db, trading_bot_id, TradingBotStatus.STOPPING)
        finally:
            db.close()

    async def get_bot_status(self, trading_bot_id: int) -> TradingBotStatus:
        db = SessionLocal()
        try:
            trading_bot = trading_bot_service.get_trading_bot(db, trading_bot_id)
            return trading_bot.status if trading_bot else None
        finally:
            db.close()

    async def manage_results(self):
        db = SessionLocal()
        try:
            while True:
                try:
                    action, bot_id, data = self.result_queue.get(timeout=1)
                    if action == "heartbeat":
                        self.worker_last_heartbeat[data] = time.time()
                    elif action == "started":
                        trading_bot_service.update_trading_bot_status(db, bot_id, TradingBotStatus.RUNNING)
                    elif action == "stopped":
                        trading_bot_service.update_trading_bot_status(db, bot_id, TradingBotStatus.STOPPED)
                        if bot_id in self.bot_worker_map:
                            del self.bot_worker_map[bot_id]
                except queue.Empty:
                    await asyncio.sleep(1)
        finally:
            db.close()

    async def monitor_workers(self):
        while True:
            for worker_id, worker in enumerate(self.workers):
                if not worker.is_alive() or time.time() - self.worker_last_heartbeat.get(worker_id, 0) > 60:
                    self._restart_worker(worker_id)
            await asyncio.sleep(30)

    def _restart_worker(self, worker_id: int):
        old_worker = self.workers[worker_id]
        old_worker.terminate()
        old_worker.join(timeout=5)
        self._start_worker(worker_id)
        self._reassign_bots(worker_id)

    def _reassign_bots(self, worker_id: int):
        db = SessionLocal()
        try:
            affected_bots = [bot_id for bot_id, assigned_worker in self.bot_worker_map.items() if
                             assigned_worker == worker_id]
            for bot_id in affected_bots:
                trading_bot = trading_bot_service.get_trading_bot(db, bot_id)
                if trading_bot and trading_bot.status == TradingBotStatus.RUNNING:
                    self.task_queues[worker_id].put((bot_id, "start", trading_bot))
        finally:
            db.close()

    async def run(self):
        self.start_workers()
        asyncio.create_task(self.manage_results())
        asyncio.create_task(self.monitor_workers())
        asyncio.create_task(self.check_pending_bots())

    async def check_pending_bots(self):
        while True:
            db = SessionLocal()
            try:
                pending_bots = trading_bot_service.get_trading_bots_by_status(db, TradingBotStatus.PENDING)
                for bot in pending_bots:
                    await self.start_bot(bot.id)
            finally:
                db.close()
            await asyncio.sleep(60)

    async def stop_all_bots(self):
        db = SessionLocal()
        try:
            running_bots = trading_bot_service.get_trading_bots_by_status(db, TradingBotStatus.RUNNING)
            for bot in running_bots:
                await self.stop_bot(bot.id)
        finally:
            db.close()

    def shutdown(self):
        asyncio.run(self.stop_all_bots())
        for worker in self.workers:
            worker.terminate()
            worker.join()


trading_engine_manager = TradingEngineManager()
