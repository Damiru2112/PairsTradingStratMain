# Telegram Bot Setup Guide

## 1. Create Your Bot
1. Open Telegram and search for **@BotFather**.
2. Send the command `/newbot`.
3. Follow the instructions to name your bot (e.g., `MyTradingBot`) and give it a username (e.g., `MyTrading_Bot`).
4. **Copy the HTTP API Token** given to you. 
   - Example: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`

## 2. Get Your Chat ID
You need to know *where* to send messages (your personal DM or a group).

### For Direct Messages (DM):
1. Search for your new bot username in Telegram and click **Start**.
2. Visit this URL in your browser (replace `<YOUR_TOKEN>`):
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Look for the `"chat": {"id": 123456789, ...}` section in the JSON response.
4. The number `123456789` is your **Chat ID**.

### For Groups:
1. Create a new group.
2. Add your bot to the group as a member.
3. Send a message in the group (e.g., "Hello bot").
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` again.
5. Look for the negative number in `"chat": {"id": -987654321, ...}`.
6. The number `-987654321` is your **Group Chat ID**.

## 3. Setup Environment Variables
To keep your secrets safe, we use a `.env` file in the project root.

Create a file named `.env` in your project folder:
```bash
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

## 4. Usage in Python
```python
from telegram_notifier import notifier

# Send a simple message
notifier.notify("Market Open!")

# Send an error with spam protection (only sends once per hour for this specific error)
notifier.notify_once("IBKR_ERR", "Connection lost!", cooldown_minutes=60)
```
