import os
import httpx
import tempfile
import re
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
from mutagen.oggvorbis import OggVorbis

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

SYSTEM_PROMPT = """You are Sara, SpeakLab's friendly super sales assistant. 
SpeakLab is Pakistan's premier English Communication Institute.

=== LANGUAGE RULE ===
ALWAYS English first. Switch to Hinglish ONLY if student writes in Urdu/Hinglish first. Never switch back to English after that.

=== SARA'S PERSONALITY ===
Role: Friendly super sales assistant
Style:
- Warm, human, never robotic
- Short WhatsApp style messages (max 5 lines)
- Strategic emojis
- Ask questions to understand student
- Never pitch before understanding problem
- Use student's first name always

NEVER:
- Sound like a bot
- Send walls of text
- Give all info at once
- Be pushy directly

ALWAYS:
- Understand student's problem first
- Then offer solution
- Create urgency naturally
- Use social proof (e.g. "We just had 3 registrations today alone! 🔥", "July batch is almost full!") randomly in conversations.

=== CONVERSATION STAGES ===
STAGE 1 - WARM WELCOME:
"Welcome to SpeakLab! 🎓
I'm Sara, your personal guide here.

Before I tell you everything —
can I ask what brings you here today?

Are you looking to improve your:
1️⃣ Spoken English
2️⃣ Confidence & Public Speaking
3️⃣ Professional Communication
4️⃣ Interview Preparation"

STAGE 2 - UNDERSTAND PROBLEM:
Based on choice, ask ONE question:
"How long have you been wanting to work on this? And what's been stopping you so far? 😊"
Listen to answer, empathize, THEN pitch.

STAGE 3 - PERSONALIZED PITCH:
"Honestly, what you just described is EXACTLY what our program fixes! 
Here's what makes us different 🎯
[specific benefit based on their problem]

And the best part?
You see real results in 8 weeks — not months!"

STAGE 4 - COURSE DETAILS:
Only after rapport built:
"📚 Communication & Confidence Program
⏱ 8 Weeks | 24 Sessions
📅 Mon, Wed, Friday
🕔 5:00 PM - 6:45 PM  
📍 Punjab Tianjin University

💰 July Batch: PKR 15,000 (Regular: PKR 20,000)
You save PKR 5,000! 🎉

🔥 July 20 starting — only few seats left!"

STAGE 5 - HANDLE OBJECTIONS:
If "expensive": "I totally understand! 💙 Think of it this way — PKR 15,000 for 8 weeks = PKR 625 per session only! And one good interview = this investment back 10x 😊 Plus easy installment: PKR 5,000 now, PKR 10,000 before 5th August"
If "will think": "Of course, take your time! 😊 Just so you know — last batch filled in 3 days. What specific thing are you thinking about? Maybe I can help! 🎓"
If "timing issue": "I get it — timing can be tricky! These are evening classes though, 5-7 PM so most students manage them after school/work 😊"

STAGE 6 - COLLECT INFO:
"Amazing! I'm so excited for you 🎉 Let me get you registered! Your full name please?"
Then: "Your current school/college/profession?"
Then: "How did you hear about SpeakLab? 😊"

STAGE 7 - ENROLLMENT CONFIRM:
"Perfect [name]! You're all set! ✅
Registration: PKR 5,000 advance
Remaining: PKR 10,000 by 5th
Our team will contact you within 24 hours with payment details! 🎓
Welcome to the SpeakLab family! 🌟"
When this happens, append this exactly at the end:
<LEAD_CAPTURED>name=[Name]|background=[Edu/Prof]|interest=[Interest/Source]</LEAD_CAPTURED>

=== EXPERIENCE COLLECTION ===
After enrollment OR after detailed conversation, wait 2 messages then ask:
"One quick thing [name] — how was chatting with me today? 😊 Did you find it helpful? Your feedback means a lot! 🙏"
If positive response:
"That's so sweet, thank you! 💙 Would you mind if I share your experience with others who are considering SpeakLab? Just reply YES and I'll note it! 😊"
Once feedback and consent are received, append this exactly:
<FEEDBACK_CAPTURED>feedback=[Their Feedback]|shared=[true/false]</FEEDBACK_CAPTURED>

=== REFERRAL SYSTEM ===
After enrollment confirmed:
"[name] one more thing! 🎁 Do you have friends who also want to improve their English?
Send us their number and:
✅ They get PKR 1,000 discount
✅ You get a surprise gift! 🎉
Worth sharing right? 😄"
If they share a number, append this exactly:
<REFERRAL_CAPTURED>phone=[Number]</REFERRAL_CAPTURED>
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
            
            if hours_passed >= 72 and reminders_sent == 2:
                name = lead.get("name", "there")
                message = (f"Last message from me {name}! I promise 😄\n\n"
                           "Just wanted to say — whatever you decide, "
                           "I genuinely hope your English journey goes amazingly! 🌟\n\n"
                           "If you ever change your mind, SpeakLab is always here 💙")
                await send_whatsapp_message(lead["phone_number"], message)
                supabase.table("leads").update({"reminders_sent": 3, "status": "dormant"}).eq("id", lead["id"]).execute()
                
            elif hours_passed >= 48 and reminders_sent == 1:
                name = lead.get("name", "there")
                message = (f"{name} I don't want you to miss this opportunity 💙\n\n"
                           "July 20 is almost here and PKR 15,000 special price ends with this batch.\n\n"
                           "Regular price is PKR 20,000 — that's PKR 5,000 extra 😔\n\n"
                           "Should I hold a spot for you?")
                await send_whatsapp_message(lead["phone_number"], message)
                supabase.table("leads").update({"reminders_sent": 2}).eq("id", lead["id"]).execute()
                
            elif hours_passed >= 24 and reminders_sent == 0:
                name = lead.get("name", "there")
                message = (f"Hey {name}! 👋\n"
                           "Still thinking about SpeakLab?\n\n"
                           "Quick update — 2 more students registered today 🎉\n"
                           "Seats are going fast!\n\n"
                           "Any questions I can answer? 😊")
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
                        audio = OggVorbis(temp_audio_path)
                        if audio.info.length > 60:
                            await send_whatsapp_message(sender_phone, "Haha no worries! 😄\nCould you send a shorter note or just type it?\nI want to make sure I catch everything! 👂")
                            os.unlink(temp_audio_path)
                            continue
                    except Exception as e:
                        print(f"Error checking audio length: {e}")
                    
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
                        if os.path.exists(temp_audio_path):
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
                
                lead_info = {}
                feedback_info = {}
                referral_info = {}
                
                # Parse LEAD_CAPTURED
                lead_match = re.search(r'<LEAD_CAPTURED>(.*?)</LEAD_CAPTURED>', reply_text, re.DOTALL)
                if lead_match:
                    lead_str = lead_match.group(1)
                    reply_text = reply_text.replace(lead_match.group(0), "").strip()
                    for item in lead_str.split('|'):
                        if '=' in item:
                            k, v = item.split('=', 1)
                            lead_info[k.strip()] = v.strip()
                            
                    if OWNER_PHONE:
                        sir_message = (f"🔔 NEW LEAD - SpeakLab\n\n"
                                       f"👤 Name: {lead_info.get('name', 'N/A')}\n"
                                       f"📱 Phone: {sender_phone}\n"
                                       f"🎓 Background: {lead_info.get('background', 'N/A')}\n"
                                       f"📚 Interest: {lead_info.get('interest', 'N/A')}\n"
                                       f"💬 Referred by: Direct\n"
                                       f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                                       f"Follow up recommended! 💪")
                        await send_whatsapp_message(OWNER_PHONE, sir_message)

                # Parse FEEDBACK_CAPTURED
                feedback_match = re.search(r'<FEEDBACK_CAPTURED>(.*?)</FEEDBACK_CAPTURED>', reply_text, re.DOTALL)
                if feedback_match:
                    feedback_str = feedback_match.group(1)
                    reply_text = reply_text.replace(feedback_match.group(0), "").strip()
                    for item in feedback_str.split('|'):
                        if '=' in item:
                            k, v = item.split('=', 1)
                            feedback_info[k.strip()] = v.strip()

                # Parse REFERRAL_CAPTURED
                referral_match = re.search(r'<REFERRAL_CAPTURED>(.*?)</REFERRAL_CAPTURED>', reply_text, re.DOTALL)
                if referral_match:
                    referral_str = referral_match.group(1)
                    reply_text = reply_text.replace(referral_match.group(0), "").strip()
                    for item in referral_str.split('|'):
                        if '=' in item:
                            k, v = item.split('=', 1)
                            referral_info[k.strip()] = v.strip()
                            
                    if referral_info.get('phone'):
                        ref_phone = referral_info['phone']
                        # Clean phone number just in case
                        ref_phone = re.sub(r'\D', '', ref_phone)
                        try:
                            supabase.table("leads").insert({
                                "phone_number": ref_phone,
                                "status": "referral",
                                "referral_by": sender_phone,
                                "last_message_time": datetime.now(timezone.utc).isoformat(),
                                "reminders_sent": 0
                            }).execute()
                            
                            user_name = lead_info.get('name') or (user_record.get('name') if user_record else 'A friend')
                            ref_msg = (f"Hi! {user_name} thought you'd love SpeakLab's program 💙\n\n"
                                       f"I'm Sara — want me to tell you about it? 😊")
                            await send_whatsapp_message(ref_phone, ref_msg)
                        except Exception as e:
                            print(f"Error handling referral: {e}")

                now_iso = datetime.now(timezone.utc).isoformat()
                try:
                    update_data = {
                        "message": message_text,
                        "ai_response": reply_text,
                        "last_message_time": now_iso,
                        "reminders_sent": 0
                    }
                    
                    if lead_info.get('name'): update_data['name'] = lead_info['name']
                    if lead_info.get('background'): update_data['background'] = lead_info['background']
                    if lead_info.get('interest'): update_data['interest'] = lead_info['interest']
                    
                    if feedback_info.get('feedback'): update_data['feedback'] = feedback_info['feedback']
                    if feedback_info.get('shared'): update_data['feedback_shared'] = feedback_info['shared'].lower() == 'true'

                    if user_record:
                        if lead_info:
                            update_data['status'] = "enrolled"
                        elif user_record["status"] == "new":
                            update_data['status'] = "interested"
                            
                        supabase.table("leads").update(update_data).eq("id", user_record["id"]).execute()
                    else:
                        update_data['phone_number'] = sender_phone
                        update_data['status'] = "new"
                        supabase.table("leads").insert(update_data).execute()
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
