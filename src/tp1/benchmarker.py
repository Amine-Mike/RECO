import time


class PipelineBenchmarker:
    def __init__(self, function_to_benchmark: callable):
        self.function_to_benchmark = function_to_benchmark

    def time_execution(self):
        """
        Compare time taken to accomplish the whole pipeline given a model
        in seconds !
        """
        start_time = time.time()
        result = self.function_to_benchmark()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {execution_time:.4f} seconds")
        return execution_time, result

    def gradient_smoothness(self):
        """
        Compare smoothness of my gradients based on a given model
        The following models will be compared:
        LSTM, BI-LSTM, MLP, GRU
        """
        pass
