"""
Runtime performance monitoring for RAG pipeline.

Tracks execution timing, hardware utilization, and token throughput
for performance regression detection and capacity planning.
"""

import time
import psutil
import logging

logger = logging.getLogger(__name__)


class RuntimeMetrics:
    """Captures execution timing and hardware utilization metrics for RAG pipeline.

    Responsibilities:
    - Measure query latency at pipeline stages
    - Track CPU/RAM/GPU utilization
    - Calculate token generation throughput

    Attributes:
        gpu_utilization_pct: Current GPU utilization percentage (0 if unavailable)
        vram_usage_mb: Current VRAM consumption in megabytes
        cpu_utilization_pct: Current CPU utilization percentage
        ram_usage_mb: Current RAM usage in megabytes
    """

    def get_hardware_metrics(self) -> dict:
        """Capture current CPU, RAM, and GPU utilization.

        Returns:
            Dict with gpu_utilization_pct, vram_usage_mb, cpu_utilization_pct, ram_usage_mb.
        """
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
        """Calculate elapsed time from perf_counter start timestamp.

        Args:
            start_time: Value from time.perf_counter().

        Returns:
            Duration in seconds.
        """
        return time.perf_counter() - start_time

    def calculate_token_speed(self, text: str, duration: float) -> float:
        """Calculate token generation throughput.

        Args:
            text: Generated response text.
            duration: Generation time in seconds.

        Returns:
            Tokens per second (0 if duration <= 0).
        """
        if duration <= 0:
            return 0.0
        tokens = len(text.split())
        return tokens / duration