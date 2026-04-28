import uvicorn

from glm2api.app import app
from glm2api.config import SECRET

if __name__ == "__main__":
    import os

    if not SECRET:
        print("[!] ZAI_SECRET 未设置，检查 .env 文件")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"[*] GLM-5.1 OpenAI Proxy → http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
