import re

FIELD = re.compile(r"([a-z]+)(-?\d+)")


def encode(fields):
    return "_".join("%s%d" % (key, value) for key, value in fields)


def decode(identifier):
    return {key: int(value) for key, value in FIELD.findall(identifier)}
