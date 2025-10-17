# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.
import cProfile
import importlib
from functools import wraps
from typing import Any, Callable

from dora_utils.node import DoraNode
from omegaconf import DictConfig

from judo.optimizers import register_optimizer
from judo.tasks import register_task


class TimeProfiler:
    """Context manager for tracking time.

    This is a context manager that can be used to measure the time taken to execute a block of code.
    """

    def __init__(self, name: str = "") -> None:
        """Initialize the time profiler."""
        self.dump_file = f"{name}_stats.prof"
        self._pr = cProfile.Profile()
        self._call_count = 0

    def profile_call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Profile a function call."""
        self._pr.enable()
        try:
            return fn(*args, **kwargs)
        finally:
            self._pr.disable()
            self._call_count += 1

    def dump_stats(self) -> None:
        """Dump the stats to a file."""
        self._pr.dump_stats(self.dump_file)


class TimedDoraNode(DoraNode):
    """A DoraNode that tracks the time taken to execute a block of code."""

    def __init__(self, node_id: str, max_workers: int | None = None, node_name: str = "") -> None:
        """Initialize the timed Dora node."""
        super().__init__(node_id=node_id, max_workers=max_workers)
        self.time_profiler = TimeProfiler(node_name)
        self.node_name = node_name

    @staticmethod
    def record_stats(func: Callable) -> Callable:
        """Decorator to catch a KeyboardInterrupt and dumps current stats to the configured dump file."""

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return func(self, *args, **kwargs)
            except KeyboardInterrupt:
                print(f"Dumping stats for {self.node_name} to file...")
                self.time_profiler.dump_stats()

        return wrapper

    def time_call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Time a function call."""
        self.time_profiler.profile_call(func, *args, **kwargs)

    def cleanup(self) -> None:
        """Cleanup the node and write stats to file."""
        self.time_profiler.dump_stats()
        super().cleanup()


def get_class_from_string(class_path: str) -> type:
    """Get a class from a string path."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def register_tasks_from_cfg(task_registration_cfg: DictConfig) -> None:
    """Register custom tasks."""
    for task_name in task_registration_cfg.keys():
        task_dict = task_registration_cfg.get(task_name, {})
        assert set(task_dict.keys()) == {"task", "config"}, (
            "Task registration must be a dict with keys 'task' and 'config'."
        )
        assert isinstance(task_dict["task"], str), "Task must be a string path to the task class."
        assert isinstance(task_dict["config"], str), "Task config must be a string path to the config class."
        task_cls = get_class_from_string(task_dict["task"])
        task_config_cls = get_class_from_string(task_dict["config"])
        register_task(str(task_name), task_cls, task_config_cls)


def register_optimizers_from_cfg(optimizer_registration_cfg: DictConfig) -> None:
    """Register custom optimizers."""
    for optimizer_name in optimizer_registration_cfg.keys():
        optimizer_dict = optimizer_registration_cfg.get(optimizer_name, {})
        assert set(optimizer_dict.keys()) == {"optimizer", "config"}, (
            "Optimizer registration must be a dict with keys 'optimizer' and 'config'."
        )
        assert isinstance(optimizer_dict["optimizer"], str), "Optimizer must be a string path to the optimizer class."
        assert isinstance(optimizer_dict["config"], str), "Optimizer config must be a string path to the config class."
        optimizer_cls = get_class_from_string(optimizer_dict["optimizer"])
        optimizer_config_cls = get_class_from_string(optimizer_dict["config"])
        register_optimizer(str(optimizer_name), optimizer_cls, optimizer_config_cls)
