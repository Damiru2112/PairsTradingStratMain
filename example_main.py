from telegram_notifier import notifier
import time

def main():
    print("--- Starting Telegram Notifier Example ---")

    # 1. Simple Notification
    print("\nSending standard notification...")
    notifier.notify("🚀 System Startup: Trading Bot is ON.")
    
    # Allow time for async thread to fire
    time.sleep(1)

    # 2. Simulate an Error
    print("\nSending error alert...")
    notifier.notify("⚠️ ERROR: Connection lost to Data Provider.")

    # 3. Demonstrate notify_once (Anti-Spam)
    print("\nTesting Anti-Spam (notify_once)...")
    key = "TEST_ERROR_1"
    
    print("Attempt 1: Should send.")
    notifier.notify_once(key, "🛑 Critical Error: Database Locked (1/3)", cooldown_minutes=1)
    
    print("Attempt 2: Should be suppressed.")
    notifier.notify_once(key, "🛑 Critical Error: Database Locked (2/3)", cooldown_minutes=1)
    
    print("Attempt 3: Should be suppressed.")
    notifier.notify_once(key, "🛑 Critical Error: Database Locked (3/3)", cooldown_minutes=1)

    print("\n--- Example Finished (Check your Telegram) ---")
    
    # Keep script alive briefly to let background threads finish
    time.sleep(2)

if __name__ == "__main__":
    main()
