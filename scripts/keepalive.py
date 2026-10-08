"""
scripts/keepalive.py
เปิดเว็บแอปด้วยเบราว์เซอร์จริง (ผ่าน Playwright) แล้วกดปุ่มปลุกถ้าเจอว่าแอป sleep อยู่

ทำไมต้องใช้เบราว์เซอร์จริง ทำไม ping ธรรมดาใช้ไม่ได้:
Streamlit Community Cloud ให้แอปพัก (sleep) เองถ้าไม่มี "traffic จริง" นานเกิน 12 ชม.
แต่การยิง HTTP GET ธรรมดา (curl, requests, cron ping ทั่วไป) จะโดนแค่หน้า static shell
เปล่าๆ ไม่ถึงตัว container ของแอปจริง จึง "ไม่" นับเป็น traffic ที่ reset ตัวจับเวลาได้
ต้องเป็นเบราว์เซอร์จริงที่โหลดหน้าเว็บ รัน JS และต่อ WebSocket เท่านั้นถึงจะถือว่าเป็น
การเข้าใช้งานจริง (นี่คือข้อจำกัดที่ยืนยันจาก GitHub issue ของทีม Streamlit เอง ไม่ใช่
เราเข้าใจผิด)

Usage:
    STREAMLIT_APP_URL=https://xxx.streamlit.app python scripts/keepalive.py
"""
import asyncio
import os
import sys

from playwright.async_api import async_playwright

APP_URL = os.environ.get("STREAMLIT_APP_URL", "").strip()
WAKE_BUTTON_TEXT = "Yes, get this app back up!"


async def visit_and_wake():
    if not APP_URL:
        print("[keepalive] ไม่พบ STREAMLIT_APP_URL -- ตั้งเป็น repository variable ก่อน")
        sys.exit(1)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        print(f"[keepalive] กำลังเปิด {APP_URL} ...")
        try:
            await page.goto(APP_URL, wait_until="domcontentloaded", timeout=120_000)
        except Exception as e:
            # แอปที่กำลังตื่นช้าๆ อาจทำให้ timeout ได้บ้าง -- ไม่ถือเป็น error ร้ายแรง
            # (รอบถัดไปจะลองใหม่เองตาม schedule) แค่ log ไว้แล้วจบแบบไม่ fail job
            print(f"[keepalive] โหลดหน้าไม่ทันเวลา (ไม่ร้ายแรง จะลองใหม่รอบหน้า): {e}")
            await browser.close()
            return

        await page.wait_for_timeout(5_000)  # ให้เวลา Streamlit เช็คสถานะแอปก่อน

        wake_button = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
        try:
            is_sleeping = await wake_button.count() > 0
        except Exception:
            is_sleeping = False

        if is_sleeping:
            print("[keepalive] แอปกำลังหลับอยู่ -- กดปุ่มปลุก...")
            await wake_button.click()
            await page.wait_for_timeout(15_000)  # รอให้เริ่ม boot
            print("[keepalive] กดปลุกแล้ว แอปกำลังตื่น (รอบหน้าจะเช็คซ้ำว่าตื่นเต็มที่รึยัง)")
        else:
            print("[keepalive] แอปตื่นอยู่แล้ว ไม่ต้องทำอะไรเพิ่ม")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(visit_and_wake())
