# zai2api

将 [chat.z.ai](https://chat.z.ai)（智谱清言/GLM）转换为 OpenAI 兼容 API 的代理服务。

## 功能

- OpenAI 兼容的 `/v1/chat/completions` 接口
- 支持 GLM-5.1、GLM-5、GLM-4.7 等模型
- 流式和非流式响应
- 自动登录获取认证 token（绕过 guest token 限流）
- CV 缺口检测 + 非线性滑块公式自动过验证码

## 快速开始

```bash
pip install -r requirements.txt
```

配置 `.env`：

```env
ZAI_SECRET=your_secret
API_KEY=your_api_key
ZAI_EMAIL=your_email
ZAI_PASSWORD=your_password
PORT=8000
```

启动：

```bash
python main.py
```

## 使用

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.1",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 自动登录

当 guest token 被限流时，服务会自动通过 Playwright + Camoufox 浏览器登录 chat.z.ai：

1. CV 检测缺口位置（轮廓特征匹配 + Sobel 备用）
2. 阿里云滑块非线性公式计算拖动距离（DOM 实测 50 点拟合，误差 < 0.5px）
3. 模拟人类拖动获取认证 token

## 项目结构

```
app/
├── auth/
│   ├── gap_detector.py   # CV 缺口检测
│   ├── login.py          # 自动登录流程
│   ├── token.py          # token 管理
│   ├── signature.py      # 请求签名
│   └── chat.py           # 聊天认证
├── routes/
│   ├── chat.py           # /v1/chat/completions
│   └── models.py         # /v1/models
├── zai/
│   ├── client.py         # 上游请求
│   └── payload.py        # 请求构造
└── sse/                  # SSE 流处理
```

## License

MIT
