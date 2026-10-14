# events.py
# A tiny event bus. This is what keeps the pose side and the audio side decoupled:
#   - the pose side only ever calls  emit("pose_held", {...})
#   - the audio side only ever calls subscribe("pose_held", its_function)
# Neither side imports or knows about the other.

_listeners = {}   # event name -> list of functions to call when that event fires


def subscribe(event_name, callback):
    if event_name not in _listeners:
        _listeners[event_name] = []
    _listeners[event_name].append(callback)


def emit(event_name, data):
    if event_name not in _listeners:
        return
    for callback in _listeners[event_name]:
        callback(data)