def message(msg: str, title="[DumpUnet]"):
    return f"{title} {msg}" if msg is not None and msg != "" else ""
