import yaml

def get_task(task_config):
    """Instantiates the task named in ``task_config["name"]`` from its params."""
    if task_config["name"] == "dvsgesture":
        from .dvsgesture import DVSGesture
        return DVSGesture(**task_config["params"])
    if task_config["name"] == "shd":
        from .shd import SHD
        return SHD(**task_config["params"])
    raise ValueError(f"Unknown task name: {task_config['name']}")