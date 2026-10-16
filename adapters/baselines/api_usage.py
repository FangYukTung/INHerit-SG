"""Shared response accounting for KeySG compatibility wrappers."""

def guarded_chat(create, record, **kwargs):
    result = create(**kwargs)
    record(result, kwargs["model"])
    return result
