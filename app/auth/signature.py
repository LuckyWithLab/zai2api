import base64
import hashlib
import hmac

from ..config import SECRET


def generate_signature(prompt: str, timestamp: str, request_id: str, user_id: str) -> str:
    sign_fields = {
        "requestId": request_id,
        "timestamp": timestamp,
        "user_id": user_id,
    }
    sorted_payload = ",".join(
        part
        for kv in sorted(sign_fields.items(), key=lambda x: x[0])
        for part in kv
    )

    prompt_b64 = base64.b64encode(prompt.strip().encode("utf-8")).decode("ascii")
    bucket = str(int(timestamp) // (5 * 60 * 1000))

    round1 = hmac.new(
        SECRET.encode("utf-8"),
        bucket.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    msg = f"{sorted_payload}|{prompt_b64}|{timestamp}"
    signature = hmac.new(
        round1.encode("utf-8"),
        msg.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return signature
