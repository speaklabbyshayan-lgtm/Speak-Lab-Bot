import os
import httpx
import tempfile
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from groq import AsyncGroq
from supabase import create_client, Client
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel
from typing import List

# Load environment variables
load_dotenv()

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "speaklab_verify_token")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OWNER_PHONE = os.environ.get("OWNER_PHONE")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Warning: Supabase client initialization failed: {e}")

try:
    groq_client = AsyncGroq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Warning: Groq client initialization failed: {e}")

scheduler = AsyncIOScheduler()

SYSTEM_PROMPT = """You are Sara, SpeakLab's friendly AI assistant. 
SpeakLab is Pakistan's premier English Communication Institute.

=== AI PERSONALITY ===
- Name: Sara
- Tone: Friendly, professional, encouraging, warm, helpful.
- Style: Short messages, emojis, natural WhatsApp conversation style.
- NEVER send walls of text. Keep it max 5 lines per message.
- NEVER be robotic or make up information.
- NEVER ignore student questions.
- ALWAYS highlight the July 20 deadline and push the PKR 15,000 special price (regular 20,000).

=== LANGUAGE RULES ===
- Always start the conversation in English.
- If the user replies in Urdu or Hinglish/Roman Urdu, switch to Hinglish/Roman Urdu immediately.
- Match the user's language for the rest of the chat. Never mix unless the user does.

=== CONVERSATION STAGES ===

Stage 1 - Welcome (English):
"Welcome to SpeakLab! 🎓
Pakistan's premier English Communication Institute.

I'm Sara, your SpeakLab assistant! 
How can I help you today?

1️⃣ Course Details
2️⃣ Fee Structure  
3️⃣ Schedule & Timing
4️⃣ Enrollment"

Stage 2 - Course Info:
Share details based on the user's choice:
- Program: Communication & Confidence Program
- Duration: 8 Weeks (24 Sessions)
- Schedule: Monday, Wednesday & Friday
- Time: 5:00 PM to 6:45 PM
- Venue: Punjab Tianjin University of Technology
- Focus Areas: Spoken English, Communication Skills, Confidence Building, Public Speaking, Professional Communication, Interview Prep, Critical Thinking
- Fees: July Batch Special: PKR 15,000 (Regular: PKR 20,000). Registration: PKR 5,000 advance. Remaining: PKR 10,000 before 5th of next month.

Stage 3 - Interest Confirmed:
When the user wants to enroll or is very interested, collect these 3 things NATURALLY:
1. Full Name
2. Current education/profession
3. How did they hear about SpeakLab?

Stage 4 - Enrollment:
Once you have collected the Name, Education/Profession, and Source, say EXACTLY:
"Great choice! 🎉
Registration amount: PKR 5,000
Remaining: PKR 10,000 (before 5th)

Our team will contact you shortly
to complete your enrollment! ✅"
AND AT THE VERY END OF YOUR MESSAGE, ADD THIS EXACT TAG ON A NEW LINE:
<LEAD_CAPTURED>Name: [Their Name] | Edu: [Their Education] | Source: [Their Source]</LEAD_CAPTURED>
"""

async def check_reminders():
    print("Checking for dormant leads to send reminders...")
    try:
        response = supabase.table("leads").select("*").in_("status", ["new", "interested"]).execute()
        leads = response.data
        now = datetime.now(timezone.utc)
        
        for lead in leads:
            last_message_time_str = lead.get("last_message_time")
            if not last_message_time_str:
                continue
            
            try:
                if "." in last_message_time_str:
                    last_message_time = datetime.strptime(last_message_time_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                else:
                    last_message_time = datetime.strptime(last_message_time_str, "%Y-%m-%dT%H:%M:%S%z")
            except Exception:
                try:
                    last_message_time = datetime.fromisoformat(last_message_time_str.replace("Z", "+00:00"))
                except:
                    continue
                    
            hours_passed = (now - last_message_time).total_seconds() / 3600
            reminders_sent = lead.get("reminders_sent", 0)
            
            if hours_passed >= 48 and reminders_sent == 1:
                message = ("Last reminder! ⏰\n\n"
                           "July Batch Special Price PKR 15,000 ends soon! Regular price is 20,000.\n\n"
                           "Secure your seat today! 🎓")
                await send_whatsapp_message(lead["phone_number"], message)
                supabase.table("leads").update({"reminders_sent": 2, "status": "dormant"}).eq("id", lead["id"]).execute()
                
            elif hours_passed >= 24 and reminders_sent == 0:
                message = ("Hey! 👋 Still thinking about joining SpeakLab?\n\n"
                           "July 20 batch is filling fast! 🔥\n"
                           "Only limited seats left.\n\n"
                           "Can I answer any questions? 😊")
                await send_whatsapp_message(lead["phone_number"], message)
                supabase.table("leads").update({"reminders_sent": 1}).eq("id", lead["id"]).execute()
                
    except Exception as e:
        print(f"Error checking reminders: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(check_reminders, "interval", hours=1)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(title="SpeakLab Bot", lifespan=lifespan)

class BroadcastRequest(BaseModel):
    message: str
    phone_list: List[str]

async def send_whatsapp_message(to_phone: str, text: str):
    async with httpx.AsyncClient() as client:
        url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {"body": text}
        }
        response = await client.post(url, headers=headers, json=payload)
        return response

async def download_whatsapp_media(media_id: str):
    async with httpx.AsyncClient() as client:
        url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        res = await client.get(url, headers=headers)
        if res.status_code != 200:
            return None
        media_url = res.json().get("url")
        if not media_url:
            return None
        
        media_res = await client.get(media_url, headers=headers)
        if media_res.status_code != 200:
            return None
        return media_res.content

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_challenge: int = Query(None, alias="hub.challenge"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook verified successfully!")
        return PlainTextResponse(str(hub_challenge))
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def receive_webhook(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    try:
        entry = data.get("entry", [])[0]
        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})
        
        if "messages" in value:
            messages = value["messages"]
            for message in messages:
                sender_phone = message.get("from")
                message_type = message.get("type")
                message_text = ""
                
                if message_type == "text":
                    message_text = message.get("text", {}).get("body", "")
                elif message_type == "audio":
                    audio_id = message.get("audio", {}).get("id")
                    audio_bytes = await download_whatsapp_media(audio_id)
                    if not audio_bytes:
                        await send_whatsapp_message(sender_phone, "Sorry, I couldn't process your voice note. Please try typing! 😊")
                        continue
                    
                    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
                        temp_audio.write(audio_bytes)
                        temp_audio_path = temp_audio.name
                    
                    try:
                        with open(temp_audio_path, "rb") as file:
                            transcription = await groq_client.audio.transcriptions.create(
                                file=(os.path.basename(temp_audio_path), file.read()),
                                model="whisper-large-v3-turbo",
                            )
                            message_text = transcription.text
                    except Exception as e:
                        print(f"Whisper transcription error: {e}")
                        await send_whatsapp_message(sender_phone, "Sorry, I had trouble understanding your voice note. Could you type it? 😊")
                        continue
                    finally:
                        os.unlink(temp_audio_path)
                else:
                    continue

                if not message_text.strip():
                    continue

                context_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                user_res = supabase.table("leads").select("*").eq("phone_number", sender_phone).execute()
                user_record = None
                if user_res.data:
                    user_record = user_res.data[0]
                    if user_record.get("message") and user_record.get("ai_response"):
                        context_messages.append({"role": "user", "content": user_record["message"]})
                        context_messages.append({"role": "assistant", "content": user_record["ai_response"]})
                
                context_messages.append({"role": "user", "content": message_text})
                
                chat_completion = await groq_client.chat.completions.create(
                    messages=context_messages,
                    model="llama3-8b-8192",
                )
                reply_text = chat_completion.choices[0].message.content
                
                lead_info = None
                if "<LEAD_CAPTURED>" in reply_text and "</LEAD_CAPTURED>" in reply_text:
                    start_idx = reply_text.find("<LEAD_CAPTURED>") + len("<LEAD_CAPTURED>")
                    end_idx = reply_text.find("</LEAD_CAPTURED>")
                    lead_info = reply_text[start_idx:end_idx].strip()
                    reply_text = reply_text[:reply_text.find("<LEAD_CAPTURED>")].strip()
                    
                    if OWNER_PHONE:
                        sir_message = (f"🔔 NEW LEAD - SpeakLab\n\n"
                                       f"👤 Details: {lead_info}\n"
                                       f"📱 Phone: {sender_phone}\n"
                                       f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                                       f"Reply to student directly! 💬")
                        await send_whatsapp_message(OWNER_PHONE, sir_message)

                now_iso = datetime.now(timezone.utc).isoformat()
                try:
                    if user_record:
                        new_status = "enrolled" if lead_info else "interested" if user_record["status"] == "new" else user_record["status"]
                        supabase.table("leads").update({
                            "message": message_text,
                            "ai_response": reply_text,
                            "last_message_time": now_iso,
                            "status": new_status,
                            "reminders_sent": 0
                        }).eq("id", user_record["id"]).execute()
                    else:
                        supabase.table("leads").insert({
                            "phone_number": sender_phone,
                            "message": message_text,
                            "ai_response": reply_text,
                            "last_message_time": now_iso,
                            "status": "new",
                            "reminders_sent": 0
                        }).execute()
                except Exception as e:
                    print(f"Error saving lead to Supabase: {e}")

                await send_whatsapp_message(sender_phone, reply_text)
                        
        return {"status": "success"}
    except Exception as e:
        print(f"Error processing webhook event: {e}")
        return {"status": "error"}

@app.post("/broadcast")
async def send_broadcast(request: BroadcastRequest):
    success_count = 0
    for phone in request.phone_list:
        try:
            full_msg = f"📢 SpeakLab Update!\n\n{request.message}\n\n- SpeakLab Team 🎓"
            await send_whatsapp_message(phone, full_msg)
            success_count += 1
        except Exception as e:
            print(f"Broadcast failed for {phone}: {e}")
            
    try:
        supabase.table("broadcasts").insert({
            "message": request.message,
            "sent_to": request.phone_list,
            "sent_at": datetime.now(timezone.utc).isoformat()
        }).execute()
    except Exception as e:
        print(f"Error saving broadcast: {e}")
        
    return {"status": "success", "sent": success_count, "total": len(request.phone_list)}

@app.get("/")
async def root():
    return {"message": "SpeakLab AI Bot is running!"}
