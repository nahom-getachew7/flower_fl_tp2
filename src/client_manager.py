from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional
import threading
import time
import random

class CustomClientManager(ClientManager):
    def __init__(self):
        super().__init__()
        self.clients: Dict[str, ClientProxy] = {}
        self.lock = threading.Lock()

    def num_available(self) -> int:
        with self.lock:
            return len(self.clients)

    def register(self, client: ClientProxy) -> bool:
        with self.lock:
            if client.cid not in self.clients:
                self.clients[client.cid] = client
                return True
            return False

    def unregister(self, client: ClientProxy) -> None:
        with self.lock:
            if client.cid in self.clients:
                del self.clients[client.cid]

    def all(self) -> Dict[str, ClientProxy]:
        with self.lock:
            return dict(self.clients)

    def wait_for(self, num_clients: int, timeout: int) -> bool:
        start_time = time.time()
        while True:
            with self.lock:
                if len(self.clients) >= num_clients:
                    return True
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.5)

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[object] = None
    ) -> List[ClientProxy]:
        with self.lock:
            available = list(self.clients.values())
            if min_num_clients is not None and len(available) < min_num_clients:
                raise ValueError(
                    f"There should be at least need {min_num_clients} but you have {len(available)})"
                )
            return random.sample(available, min(num_clients, len(available)))