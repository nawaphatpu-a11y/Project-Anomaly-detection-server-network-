"""เปิดเว็บแอปด้วย Playwright แล้วกดปุ่มปลุกถ้าแอป sleep อยู่"""
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
            print(f"[keepalive] โหลดหน้าไม่ทันเวลา (ไม่ร้ายแรง จะลองใหม่รอบหน้า): {e}")
            await browser.close()
            return

        await page.wait_for_timeout(5_000)

        wake_button = page.get_by_role("button", name=WAKE_BUTTON_TEXT)
        try:
            is_sleeping = await wake_button.count() > 0
        except Exception:
            is_sleeping = False

        if is_sleeping:
            print("[keepalive] แอปกำลังหลับอยู่ -- กดปุ่มปลุก...")
            await wake_button.click()
            await page.wait_for_timeout(15_000)
            print("[keepalive] กดปลุกแล้ว แอปกำลังตื่น (รอบหน้าจะเช็คซ้ำว่าตื่นเต็มที่รึยัง)")
        else:
            print("[keepalive] แอปตื่นอยู่แล้ว ไม่ต้องทำอะไรเพิ่ม")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(visit_and_wake())
