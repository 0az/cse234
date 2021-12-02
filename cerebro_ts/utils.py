import datetime


def timestamp():
    datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')


def positive_int(s: str) -> int:
    i = int(s)
    if i <= 0:
        raise ValueError(f'Expected positive integer, got {s}')
    return i
