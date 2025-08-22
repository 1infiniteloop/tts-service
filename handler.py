import runpod

def handler(event):
    # sanity: echo back what we got
    return {"ok": True, "echo": event}

runpod.serverless.start({"handler": handler})