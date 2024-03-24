from .preamble import *
''' The following is a simple patch to the dataclass library to dump dataclasses natively '''
from json import JSONEncoder
_json_encoder_default = JSONEncoder.default
def _default_json_encoder_patch(self, o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    elif isinstance(o, (datetime.datetime, datetime.date, datetime.time)):
        return o.isoformat()
    elif isinstance(o, datetime.timedelta):
        return (datetime.datetime.min + o).time().isoformat()
    elif isinstance(o, Path):
        return str(o)
    return _json_encoder_default(self, o)
JSONEncoder.default = _default_json_encoder_patch
