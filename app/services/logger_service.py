class LoggerService:

    def log(self, *msg):
        print("[LOG]", *msg)

    def error(self, *msg):
        print("[ERROR]", *msg)

    def latency(self, name: str, ms: float):
        print(f"[LATENCY] {name}: {ms} ms")

logger = LoggerService()
