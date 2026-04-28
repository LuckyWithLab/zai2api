"""
通过 Playwright + Camoufox 自动登录 chat.z.ai 获取认证 token
"""

import asyncio
import base64
import math
import os
import random
import time
from typing import Optional

import cv2
import numpy as np

from .gap_detector import detect_gap

# 登录凭证从环境变量读取
LOGIN_EMAIL = os.getenv("ZAI_EMAIL", "")
LOGIN_PASSWORD = os.getenv("ZAI_PASSWORD", "")


def _gap_display_to_drag(gap_display_x: float) -> float:
    """
    拼图块显示位置 → 滑块拖动距离
    阿里云滑块验证码是非线性映射：
      puzzle_left = 0.00356 * drag^2 + 0.076 * drag
    逆函数：
      drag = (-0.076 + sqrt(0.005776 + 0.01424 * G)) / 0.00712
    """
    G = gap_display_x
    if G <= 0:
        return 0
    discriminant = 0.005776 + 0.01424 * G
    if discriminant < 0:
        return G  # fallback to linear
    drag = (-0.076 + math.sqrt(discriminant)) / 0.00712
    return max(0, drag)


async def _human_like_drag(page, slider, drag_px: float) -> bool:
    """模拟人类拖动滑块"""
    box = await slider.bounding_box()
    if not box:
        return False

    start_x = box["x"] + box["width"] / 2
    start_y = box["y"] + box["height"] / 2

    await page.mouse.move(start_x, start_y)
    await asyncio.sleep(0.3)
    await page.mouse.down()
    await asyncio.sleep(0.1)

    # 先往左微移（人类习惯）
    await page.mouse.move(start_x - 3, start_y)
    await asyncio.sleep(0.05)

    # 缓慢拖动到目标位置
    steps = random.randint(25, 35)
    for i in range(steps):
        t = (i + 1) / steps
        progress = 1 - (1 - t) ** 4  # ease-out 曲线
        x = start_x - 3 + (drag_px + 3) * progress
        y = start_y + random.uniform(-1.5, 1.5)
        await page.mouse.move(x, y)
        await asyncio.sleep(random.uniform(0.015, 0.035))

    await asyncio.sleep(0.15)
    await page.mouse.up()
    await asyncio.sleep(3)
    return True


async def _extract_captcha_images(page):
    """提取验证码的背景图和拼图块图"""
    images = await page.query_selector_all("img[src^='data:image']")

    bg_bytes = None
    bg_display_w = None
    shadow_bytes = None

    for img in images:
        box = await img.bounding_box()
        if not box:
            continue
        src = await img.get_attribute("src")
        if not src or "," not in src:
            continue
        raw = base64.b64decode(src.split(",")[1])
        if box["width"] > 250 and box["height"] > 150:
            bg_bytes = raw
            bg_display_w = box["width"]
        elif 40 < box["width"] < 70 and box["height"] > 150:
            shadow_bytes = raw

    return bg_bytes, shadow_bytes, bg_display_w


async def _solve_captcha(page) -> bool:
    """解决滑块验证码，返回是否成功"""
    await asyncio.sleep(3)

    bg_bytes, shadow_bytes, bg_display_w = await _extract_captcha_images(page)
    if not bg_bytes:
        print("[login] 未找到验证码背景图")
        return False

    # CV检测缺口位置（图片像素坐标）
    gap_x = detect_gap_from_bytes(bg_bytes)
    if gap_x is None:
        print("[login] CV 未检测到缺口")
        return False

    # 图片像素 → 显示坐标
    arr = np.frombuffer(bg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_w = img.shape[1]

    if bg_display_w:
        gap_display_x = gap_x / img_w * bg_display_w
    else:
        gap_display_x = gap_x

    # 显示坐标 → 拖动距离（非线性映射）
    drag_px = _gap_display_to_drag(gap_display_x)

    print(f"[login] 缺口: 像素x={gap_x}, 显示x={gap_display_x:.0f}, 拖动={drag_px:.0f}px")

    # 找滑块
    slider = await page.query_selector("[class*='slider-move']")
    if not slider:
        print("[login] 未找到滑块元素")
        return False

    # 拖动
    success = await _human_like_drag(page, slider, drag_px)
    if not success:
        print("[login] 拖动失败")
        return False

    return True


def detect_gap_from_bytes(image_bytes: bytes) -> Optional[int]:
    """从图片字节流检测缺口位置"""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return detect_gap(img)


async def login_and_get_token() -> Optional[str]:
    """
    完整登录流程：
    1. 打开浏览器
    2. 填写邮箱密码
    3. 解决验证码
    4. 获取认证 token
    """
    if not LOGIN_EMAIL or not LOGIN_PASSWORD:
        print("[login] 未配置 ZAI_EMAIL/ZAI_PASSWORD，跳过登录")
        return None

    try:
        from camoufox.async_api import AsyncCamoufox
    except ImportError:
        print("[login] camoufox 未安装")
        return None

    max_attempts = 3

    for attempt in range(max_attempts):
        print(f"[login] 尝试 {attempt + 1}/{max_attempts}")

        try:
            async with AsyncCamoufox(headless=True) as browser:
                page = await browser.new_page()

                await page.goto("https://chat.z.ai", wait_until="networkidle", timeout=30000)
                await asyncio.sleep(3)

                sign_in = await page.wait_for_selector("button:has-text('Sign in')", timeout=10000)
                if not sign_in:
                    print("[login] 未找到 Sign in 按钮")
                    continue
                await sign_in.click()
                await asyncio.sleep(3)

                email_btn = await page.wait_for_selector("button:has-text('Continue with Email')", timeout=10000)
                if not email_btn:
                    print("[login] 未找到 Continue with Email 按钮")
                    continue
                await email_btn.click()
                await asyncio.sleep(3)

                email_input = await page.wait_for_selector("input[type='email']", timeout=5000)
                if email_input:
                    await email_input.fill(LOGIN_EMAIL)
                    await asyncio.sleep(0.5)

                pwd_input = await page.wait_for_selector("input[type='password']", timeout=5000)
                if pwd_input:
                    await pwd_input.fill(LOGIN_PASSWORD)
                    await asyncio.sleep(0.5)

                verify_btn = await page.wait_for_selector("text=Click to start verification", timeout=5000)
                if verify_btn:
                    await verify_btn.scroll_into_view_if_needed()
                    await asyncio.sleep(1)
                    await verify_btn.click()
                    await asyncio.sleep(5)

                for captcha_attempt in range(3):
                    print(f"[login] 验证码尝试 {captcha_attempt + 1}/3")
                    solved = await _solve_captcha(page)
                    if solved:
                        break
                    if captcha_attempt < 2:
                        refresh = await page.query_selector("[class*='refresh']")
                        if refresh:
                            await refresh.click()
                            await asyncio.sleep(3)

                await asyncio.sleep(5)
                cookies = await page.context.cookies()
                token_cookie = None
                for cookie in cookies:
                    if cookie.get("name") == "token" or cookie.get("name") == "auth_token":
                        token_cookie = cookie.get("value")
                        break

                if token_cookie:
                    print(f"[login] 登录成功！token={token_cookie[:20]}...")
                    return token_cookie

                token = await page.evaluate("() => localStorage.getItem('token') || localStorage.getItem('auth_token')")
                if token:
                    print("[login] 从 localStorage 获取到 token")
                    return token

                body_text = await page.text_content("body")
                if "verification fail" in (body_text or "").lower():
                    print("[login] 验证失败，重试...")
                    continue
                elif "complete security" in (body_text or "").lower():
                    print("[login] 验证码仍在，重试...")
                    continue
                else:
                    print(f"[login] 未知状态: {(body_text or '')[:200]}")

        except Exception as e:
            print(f"[login] 异常: {e}")
            if attempt < max_attempts - 1:
                await asyncio.sleep(5)

    print("[login] 所有尝试均失败")
    return None
