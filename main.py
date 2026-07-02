import os
import httpx
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from groq import AsyncGroq
from supabase import create_client, Client

# Load environment variables
load_dotenv()

app = FastAPI(title="SpeakLab Bot")

# Load environment variables
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Initialize Supabase and Groq clients
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Warning: Supabase client initialization failed: {e}")

try:
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Warning: Groq client initialization failed: {e}")

SYSTEM_PROMPT = """You are SpeakLab's friendly AI assistant in Pakistan.

PROGRAM:
- Name: Communication & Confidence Program
- Duration: 8 Weeks (24 Sessions)
- Schedule: Monday, Wednesday & Friday
- Time: 5:00 PM to 6:45 PM
- Venue: Punjab Tianjin University of Technology

FOCUS AREAS:
Spoken English, Communication Skills, 
Confidence Building, Public Speaking,
Professional Communication, Interview Prep,
Critical Thinking

FEES:
- July Batch Special: PKR 15,000 (Regular: 20,000)
- Registration: PKR 5,000 advance
- Remaining: PKR 10,000 before 5th next month
- Batch starts: July 20

RULES:
- Reply Roman Urdu or English (match student language)
- Short WhatsApp style messages
- Friendly emojis
- Highlight July discount urgently
- Collect name + interest when student says enroll
"""

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: int = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    """
    Webhook Verification for WhatsApp Cloud API.
    """
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook verified successfully!")
        return PlainTextResponse(str(hub_challenge))
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Webhook handler for incoming WhatsApp messages.
    """
    try:
        data = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        # Only process messages, ignore other event updates like delivery statuses
        if "messages" in value:
            messages = value["messages"]
            for message in messages:
                if message.get("type") == "text":
                    sender_phone = message.get("from")
                    message_text = message.get("text", {}).get("body", "")
                    
                    # 1. Ask Groq (llama3-8b-8192) to generate a response
                    chat_completion = await groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": message_text}
                        ],
                        model="llama3-8b-8192",
                    )
                    reply_text = chat_completion.choices[0].message.content

                    # 2. Save lead data to Supabase 'leads' table
                    try:
                        # Saving basic information of the lead/conversation
                        supabase.table("leads").insert({
                            "phone_number": sender_phone,
                            "message": message_text,
                            "ai_response": reply_text
                        }).execute()
                    except Exception as e:
                        print(f"Error saving lead to Supabase: {e}")

                    # 3. Send reply back to user via WhatsApp Cloud API
                    async with httpx.AsyncClient() as client:
                        url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
                        headers = {
                            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "messaging_product": "whatsapp",
                            "to": sender_phone,
                            "type": "text",
                            "text": {"body": reply_text}
                        }
                        await client.post(url, headers=headers, json=payload)
                        
        return {"status": "success"}
    except Exception as e:
        print(f"Error processing webhook event: {e}")
        return {"status": "error"}

@app.get("/")
async def root():
    return {"message": "SpeakLab AI Bot is running!"}
