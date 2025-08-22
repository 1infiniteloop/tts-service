# handler.py
import runpod

def handler(event):
    inp = event.get("input", {})
    return {"ok": True, "echo": inp}

runpod.serverless.start({"handler": handler})