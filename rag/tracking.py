"""Thin wrapper around the Klique (ClearML) task SDK.

Every run is a tracked task. With CLEARML_OFFLINE_MODE=1 the same calls run
without a server and write the session to disk, which is how the local flow and
the tests execute them. RAG_TRACKING=off skips task creation entirely.
"""
from __future__ import annotations

import os


def tracking_enabled() -> bool:
    return os.environ.get("RAG_TRACKING", "on").lower() not in ("off", "0", "false")


def start_task(project: str, name: str, task_type: str = "data_processing", params: dict | None = None):
    """Return the current task (inside a pipeline step) or create one.
    Parameters are connected so they are editable after Clone, as in the labs."""
    if not tracking_enabled():
        return None
    from clearml import Task

    task = Task.current_task() or Task.init(
        project_name=project, task_name=name, task_type=task_type,
        auto_connect_frameworks=False, auto_resource_monitoring=False)
    if params:
        task.connect(params, name="Args")
    return task


def report_values(task, values: dict) -> None:
    """Report single-value scalars (shown in the task's SCALARS tab and usable
    by the Trigger Manager's metric triggers)."""
    for key, value in values.items():
        print("metric %s = %s" % (key, value))
        if task is not None and isinstance(value, (int, float)):
            task.get_logger().report_single_value(key, value)


def close_task(task) -> None:
    if task is not None:
        task.flush(wait_for_uploads=True)
        task.close()
