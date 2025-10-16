"""DAG module for managing dependencies between segmentation tasks."""

from .dag_setting import (
    set_dependency,
    remove_dependency,
    get_dependencies,
    get_edge_distance,
    get_edge_time,
    get_edge_cost
)

__all__ = [
    "set_dependency",
    "remove_dependency",
    "get_dependencies",
    "get_edge_distance",
    "get_edge_time",
    "get_edge_cost"
]