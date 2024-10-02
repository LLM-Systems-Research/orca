from orca.engine_py.engine import ORCAExecutionEngine
from threading import Thread
import random
import torch
from enum import Enum
from dataclasses import dataclass

class RequestState(Enum):
    INITIATION = 1
    RUNNING = 2
    INCREMENT = 3
    COMPLETED = 4

# Request Object
@dataclass
class Request: 
    def __init__(self,  prompt: str, request_id: int = 0, max_tokens: int = 100):
        self.state = RequestState.INITIATION
        self.max_tokens = max_tokens
        self.prompt = prompt
        self.request_id = request_id
        self.response = ""
        self.tokens_generated = 0


class Scheduler:
    def __init__(self, engine: ORCAExecutionEngine, n_workers: int = 4, max_batch_size: int = 16, n_kv_slots: int = 2000) -> None:
        self.request_pool = []
        self.n_workers = n_workers
        self.engine = engine
        self.MAX_BATCH_SIZE = max_batch_size
        self.n_kv_slots = n_kv_slots
        self.current_request_id = 0
    
    # Algorithm to select requests for batch processing
    def select(self, n_rsrv: int) -> tuple[list[Request], int]:
        """Select a batch of requests to process based on the current request pool and the number of reserved slots."""
        batch = []
        request_pool = [req for req in self.request_pool if req.state != RequestState.RUNNING]
        request_pool.sort(key=lambda x: x.request_id)  # Sort by arrival time 

        for req in request_pool:
            if len(batch) == self.MAX_BATCH_SIZE:
                break
            if req.state == RequestState.INITIATION:
                new_n_rsrv = n_rsrv + req.max_tokens
                if new_n_rsrv > self.n_kv_slots: 
                    break
                n_rsrv = new_n_rsrv
            batch.append(req)

        return batch, n_rsrv

    def add_request(self, prompt: str):
        """Add a single request to the request pool."""
        max_tokens = len(prompt.split())  # Use word count as max_tokens for simplicity
        new_request = Request(prompt, max_tokens, self.current_request_id)
        self.request_pool.append(new_request)
        self.current_request_id += 1
        print(f"Request {self.current_request_id} submitted: {prompt[:30]}...")
    
    def add_request_batch(self, prompts: list[str]):
        """Add a batch of requests to the request pool."""
        for prompt in prompts:
            self.add_request(prompt)

    def process_requests(self):
        """Start the scheduler to handle the request pool. This will run until interrupted."""
        while True:
            self.orca_scheduling()
    
    def orca_scheduling(self):
        n_scheduled = 0
        n_rsrv = 0
        threads = []  # To store threads and their corresponding batches

        while self.request_pool:
            batch, n_rsrv = self.select(n_rsrv)

            if not batch:  # If no valid batch is found, exit the loop
                break

            # Create tensor input for the execution engine based on batch size and input dimensions
            input_dim = 64  # Define your input dimension
            input_tensor = torch.randn(len(batch), 3, input_dim)  # Example tensor shape

            # Create a thread for processing the batch using the execution engine
            thread = Thread(target=lambda b=batch: self.engine.execute(input_tensor))
            thread.start()
            
            # Store the thread along with its batch in the list
            threads.append((thread, batch))
            print(f"Scheduling batch of {len(batch)} requests")

            for req in batch:
                req.state = RequestState.RUNNING
            
            n_scheduled += 1

            # If all worker threads are engaged, wait for any thread to complete
            if n_scheduled >= self.n_workers:
                while True:
                    for idx, (thread, batch) in enumerate(threads):
                        if not thread.is_alive():  # If thread has finished
                            completed_batch = batch
                            # Process the completed batch state updates
                            for req in completed_batch:
                                if req.tokens_generated >= 2000:
                                    n_rsrv -= req.max_tokens
                                    print(f"Request {req.request_id} completed.")

                            # Remove the completed thread from the list
                            threads.pop(idx)
                            n_scheduled -= len(completed_batch)
                            break
                    else:
                        continue
                    break

            # Remove completed requests from the pool
            request_pool = [req for req in self.request_pool if req.state != RequestState.COMPLETED]

            print(f"Requests Yet To Be Processed: {len(request_pool)}")