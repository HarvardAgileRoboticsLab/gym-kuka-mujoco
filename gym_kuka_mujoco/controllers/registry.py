controller_registry = dict()

def register_controller(cls, name):
    assert name not in controller_registry, "Controller name: {} is already registered".format(name)
    controller_registry[name] = cls
