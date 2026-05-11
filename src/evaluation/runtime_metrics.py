import time
import psutil
import logging

logger = logging.getLogger(__name__)

class RuntimeMetrics:
    """Handles hardware utilization monitoring and execution timing."""

    def get_hardware_metrics(self) -> dict:
        """Captures current CPU, RAM, and GPU utilization."""
        metrics = {
            "gpu_utilization_pct": 0,
            "vram_usage_mb": 0.0,
            "cpu_utilization_pct": psutil.cpu_percent(),
            "ram_usage_mb": psutil.virtual_memory().used / (1024 * 1024)
        }
        # GPU metrics typically require torch.cuda or pynvml
        # This is a placeholder for the logic previously in RAGMetrics
        return metrics

    def measure_duration(self, start_time: float) -> float:
        """Calculates duration in seconds from a perf_counter start time."""
        return time.perf_counter() - start_time

    def calculate_token_speed(self, text: str, duration: float) -> float:
        """Calculates generation speed in tokens per second."""
        if duration <= 0:
            return 0.0
        tokens = len(text.split())
        return tokens / duration