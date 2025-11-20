import time


class PipelineBenchmarker:
    def __init__(self, function_to_benchmark: callable):
        self.function_to_benchmark = function_to_benchmark

    def time_execution(self):
        start_time = time.time()
        result = self.function_to_benchmark()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.4f} seconds")
        return execution_time, result
