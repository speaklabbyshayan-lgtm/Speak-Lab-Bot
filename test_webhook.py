import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()
WABA_ID = os.environ.get("WABA_ID", "1566677701731038")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "1281579466704018")

async def test_webhook():
    url = "https://speak-lab-bot-production.up.railway.app/webhook"
    
    payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": WABA_ID, # WABA_ID
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "1234567890",
                        "phone_number_id": WHATSAPP_PHONE_NUMBER_ID
                    },
                    "contacts": [{
                        "profile": {
                            "name": "Test User"
                        },
                        "wa_id": "923014497532"
                    }],
                    "messages": [{
                        "from": "923014497532",
                        "id": "wamid.HBgL...",
                        "timestamp": "1720454400",
                        "text": {
                            "body": "Hi there! I am testing the webhook."
                        },
                        "type": "text"
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            print(f"Status Code: {response.status_code}")
            print(f"Response Body: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_webhook())
