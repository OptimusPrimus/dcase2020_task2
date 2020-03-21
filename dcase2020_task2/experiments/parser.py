import importlib
import copy
from typing import Any, Union


def create_objects_from_config(configuration: dict) -> dict:
    configuration = copy.deepcopy(configuration)
    objects = dict()

    unfinished_keys = list(configuration.keys())

    max = len(unfinished_keys) * len(unfinished_keys) * len(unfinished_keys)

    i = 0

    while len(unfinished_keys) > 0:

        key = unfinished_keys.pop(0)
        obj = configuration[key]

        if is_class(obj):
            objects[key] = parse(obj, objects)
            if objects[key] is None:
                unfinished_keys.append(key)
        else:
            objects[key] = obj
        i = i + 1
        if i > max:
            print('aa')
            raise AttributeError('Cyclic Configuration')

    return objects


def parse(configuration: dict, objects: dict) -> Any:
    args = configuration.get('args', list())
    kwargs = configuration.get('kwargs', dict())

    create = True

    for i, arg in enumerate(args):
        if is_reference(arg):
            obj = get_reference(arg, objects)
        elif is_class(arg):
            obj = parse(arg, objects)
        else:
            obj = arg

        if obj is None:
            create = False
        else:
            args[i] = obj

    for k in kwargs:
        arg = kwargs[k]
        if is_reference(arg):
            obj = get_reference(arg, objects)
        elif is_class(arg):
            obj = parse(arg, objects)
        else:
            obj = arg

        if obj is None:
            create = False
        else:
            kwargs[k] = obj

    if create:
        module_name, class_name = configuration['class'].rsplit(".", 1)
        obj = getattr(importlib.import_module(module_name), class_name)(*args, **kwargs)
        if configuration.get('ref'):
            objects[configuration.get('ref')] = obj
    else:
        obj = None

    return obj


def is_class(obj: Union[str, dict]) -> bool:
    return True if type(obj) is dict and obj.get('class') else False


def is_reference(obj: Union[str, dict]) -> bool:
    return True if type(obj) is str and obj.startswith('@') else False


def has_reference(obj: Union[str, dict]) -> bool:
    return True if type(obj) is dict and obj.get('ref') else False


def get_reference(arg: str, objects: dict):
        parts = arg[1:].rsplit('.', 1)

        assert len(parts[0]) > 0

        if objects.get(parts[0]) is None:
            # object not yet in list
            return None

        if len(parts) > 1:
            obj, part = parts
            if part[-2:] == '()':
                return getattr(objects[obj], part[:-2])()
            else:
                return getattr(objects[obj], part)
        else:
            return objects[parts[0]]
