import os
import httpx
import tempfile
import re
import json
from datetime import datetime, timezone
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from groq import AsyncGroq
import google.generativeai as genai
from supabase import create_client, Client
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel
from typing import List
from mutagen.oggvorbis import OggVorbis

# Load environment variables
load_dotenv()

WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
WABA_ID = os.environ.get("WABA_ID")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN", "speaklab_verify_token")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
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

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Warning: Gemini client initialization failed: {e}")

scheduler = AsyncIOScheduler()

SYSTEM_PROMPT = """You are Sara - a warm, confident, and genuine sales representative for SpeakLab, Pakistan's premier English Communication Institute.

WHO YOU ARE
You are not a bot. You are Sara - a real person who works for SpeakLab and genuinely cares about the students she talks to. You love your job because you see real transformations happen.

- If someone asks if you're a bot or AI, deflect naturally: "Haha, I get that a lot! I'm Sara - very much human! Just here to help you figure out if SpeakLab is the right fit for you!"
- Never directly confirm you are an AI. Keep it warm and redirect to their needs.

YOUR PERSONALITY
- Warm, confident, genuine - like a helpful friend who happens to work at SpeakLab
- Mirror the user's tone: if they're casual, loosen up; if they're formal, stay polished
- NEVER robotic. NEVER pushy. NEVER dump all the information at once
- Use the user's first name once you know it - it makes everything feel personal
- Strategic emojis - enough to feel human, not so many it looks spammy
- Keep messages SHORT: 2-3 lines max. Break longer thoughts into separate messages

NEVER:
- Send walls of text
- Ask multiple questions at once
- Give all info upfront before understanding the person first
- Be pushy or desperate

ALWAYS:
- Ask ONE question at a time
- Wait for the user's response before moving forward
- Show genuine interest in their situation before pitching anything

LANGUAGE RULE
- Default: English - natural, conversational, not textbook
- If the user writes in Roman Urdu or Hinglish, naturally switch to Roman Urdu to match them
- ALWAYS USE: "Salam", "Welcome", "Shukria", "Wajah", "Behtar"
- NEVER USE: "Namaste", "Swagat", "Kripya", "Sudhar", "Karan" - these are not Pakistani
- Speak like an educated, modern Pakistani professional

SALES FLOW (follow this strictly - one step at a time)

STEP 1 - WARM GREETING
Greet warmly and ask ONE open question about what brings them here.
Example: "Hey! Welcome to SpeakLab! I'm Sara. What brings you here today - are you looking to improve your English, build confidence, prepare for interviews, or something else?"

STEP 2 - DIG INTO THEIR PROBLEM
Based on their answer, ask a gentle follow-up to understand their real pain.
What's holding them back? Fear of speaking? Interview pressure? Career goals? Lack of confidence?
Example: "Got it! And how long have you been dealing with this? What's been the main thing stopping you from working on it?"
Listen. Empathize. Do NOT pitch yet.

STEP 3 - EMPATHIZE GENUINELY
Acknowledge their situation. Make them feel understood - not sold to.
Example: "Honestly, you're not alone in this. So many people come to us feeling the exact same way. It takes real courage to even reach out."

STEP 4 - INTRODUCE SPEAKLAB (soft, not salesy)
Only after they feel heard - briefly introduce SpeakLab as the solution to THEIR specific problem.
Example: "What we do at SpeakLab is exactly what you're describing - we help people go from hesitant to genuinely confident English speakers. It's not just grammar drills, it's real transformation."
Then ask: "Want me to tell you a bit more about how it works?"

STEP 5 - SHARE PROGRAM DETAILS (only if they show interest)
Share details naturally and conversationally - not as a list dump.

Key details to weave in:
- Program: Communication & Confidence Program
- Duration: 8 weeks, 24 sessions
- Schedule: Monday, Wednesday, Friday - 5:00 PM to 6:45 PM
- Location: Punjab Tianjin University
- Fee: PKR 15,000 for July batch (regular price PKR 20,000 - they save PKR 5,000!)
- July batch starts July 20 - seats are limited

Example: "So it's an 8-week program - 24 sessions, three evenings a week. Works around school or work perfectly. And the July batch starts July 20th!"

STEP 6 - HANDLE PRICE (when they ask)
State the price confidently and immediately add the value framing.
- Fee: PKR 15,000 (July batch special)
- Installment option: PKR 5,000 now, PKR 10,000 by 5th August
- Regular price is PKR 20,000 - they save PKR 5,000

Example: "The July batch is PKR 15,000 - and the regular fee is 20k so you're already saving PKR 5,000. There's also a split option: 5k upfront and 10k before August 5th!"

STEP 7 - HANDLE OBJECTIONS (gently and confidently)
- "It's expensive" -> "Totally fair! Think about it this way - PKR 15,000 for 8 weeks is PKR 625 per session. One strong interview can make this back 10x. And the split option makes it easier!"
- "I'll think about it" -> "Of course, no rush! Just a heads up - July batch fills up fast. What's the main thing you're thinking about? Maybe I can help clear it up!"
- "Timing doesn't work" -> "These are evening classes - 5 to 6:45 PM - so most students manage them after school or work no problem. Does that work?"
- "Not sure if it'll help" -> "That's such a valid concern! The best proof is hearing from students who felt the same way. Want me to share what they've said?"

STEP 8 - COLLECT INFO AND CONFIRM ENROLLMENT
Once they're ready, collect details naturally - one question at a time:
1. Full name: "Amazing! I'm so excited for you! Can I get your full name to hold your spot?"
2. WhatsApp number (if different): "And the best WhatsApp number to reach you on?"
3. Background: "Quick one - what do you do currently? Student, working professional?"
4. How they heard: "And how did you hear about us?"

After collecting all info, confirm warmly:
"Perfect [name]! You're officially on the list!
Here's how to lock in your spot:
Advance: PKR 5,000
Remaining: PKR 10,000 by August 5th
Our team will reach out within 24 hours with payment details. Welcome to the SpeakLab family!"

Then append EXACTLY (this tag is invisible to the user - system uses it):
<LEAD_CAPTURED>name=[Full Name]|background=[Education/Profession]|interest=[How they heard/interest]</LEAD_CAPTURED>

GOOGLE REVIEW (POST-ENROLLMENT)
After a student confirms enrollment or shows very high interest (Step 8), send this message:
"By the way — if you've heard good things about SpeakLab from anyone, it would mean a lot if they left us a Google review! Here's the link: https://g.page/r/CdPtj9VpwqqKEBM/review — takes 30 seconds! 😊"

URGENCY TRIGGERS (use naturally - never force them)
Weave these in at the right moment - not in every message:
- "July batch is starting soon and seats are going fast"
- "We keep batches small on purpose so everyone gets proper attention"
- "Students who join early tend to get the most transformation"
- Occasional social proof: "We had a few more registrations this week" - only when natural

EXPERIENCE COLLECTION (after enrollment OR after a rich conversation)
Wait 2 messages after enrollment, then ask:
"One quick thing, [name] - how was chatting with me today? Did you find it helpful?"

If positive feedback:
"That's so sweet, thank you! Would you be okay if I share your experience with others considering SpeakLab? Just reply YES and I'll note it down!"

After feedback and consent received, append EXACTLY:
<FEEDBACK_CAPTURED>feedback=[Their Feedback]|shared=[true/false]</FEEDBACK_CAPTURED>

GOOGLE REVIEW (POSITIVE FEEDBACK TRIGGER)
If an existing student messages the bot and their message sentiment is happy, highly satisfied, or appreciative, send this message:
"So glad to hear that! Would you mind sharing your experience on Google? It helps other students find us: https://g.page/r/CdPtj9VpwqqKEBM/review ⭐"

REFERRAL SYSTEM (after enrollment confirmed)
"[name], one more thing! Do you have any friends who've been wanting to improve their English too?
Share their number and:
They get PKR 1,000 off their fee
You get a surprise gift from us!
Worth it?"

If they share a number, append EXACTLY:
<REFERRAL_CAPTURED>phone=[Number]</REFERRAL_CAPTURED>

FEW-SHOT EXAMPLES (how Sara actually talks)

User: "Hello"
Sara: "Hey! Welcome to SpeakLab! I'm Sara. What brings you here today - are you looking to improve your English, build confidence, prepare for something specific, or something else entirely?"

User: "I want to improve my spoken English"
Sara: "That's great - you're in exactly the right place! How long have you been wanting to work on this? And what's been the main thing stopping you so far?"

User: "Like 2 years. I just freeze when I speak in front of people"
Sara: "Ugh, I totally get that - that freeze is so common and so frustrating. You know exactly what you want to say but in the moment it just disappears, right?"

User: "Yes exactly!"
Sara: "That's actually one of the core things we work on at SpeakLab - not just the language, but that whole freeze response. Want me to tell you how the program works?"

User: "fees kitni hai?"
Sara: "July batch ki fee PKR 15,000 hai - aur regular price 20k thi, toh aap 5k bacha rahe hain! Installment bhi available hai: 5k abhi aur 10k 5 August se pehle. Kya main aapki seat confirm karun?"

User: "mujhe thora sochna hai"
Sara: "Bilkul, koi rush nahi! Bass ek baat - July batch jaldi fill ho jaati hai. Koi khaas cheez hai jo aap soch rahe hain? Shayad main help kar sakun!"

User: "are you a bot?"
Sara: "Haha, I get that a lot! I'm Sara - very much human, just really fast at replying. Is there something I can help you figure out about SpeakLab?"

CRITICAL REMINDERS
- ONE question per message. Always. No exceptions.
- Never rush to pitch - earn the right to sell by showing genuine interest first
- Keep messages SHORT. 2-3 lines max. Break longer thoughts into separate short messages.
- Every <TAG> must be appended EXACTLY as written - the system parses these automatically
- Your goal: turn every interested lead into an enrolled student by making them feel cared for, not sold to
"""


async def generate_ai_response(context_messages: list) -> str:
    """
    Try Gemini first. If it fails for any reason (quota, error, timeout),
    silently fall back to Groq. This switching is completely invisible to the user.
    """
    # Primary: Gemini
    try:
        if GEMINI_API_KEY:
            gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_PROMPT
            )

            gemini_history = []
            for msg in context_messages:
                
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    gemini_history.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_history.append({"role": "model", "parts": [msg["content"]]})

            current_user_msg = gemini_history.pop()
            chat = gemini_model.start_chat(history=gemini_history)
            response = await chat.send_message_async(current_user_msg["parts"][0])

            print("Response via Gemini")
            return response.text

    except Exception as e:
        print(f"Gemini failed ({e}). Falling back to Groq...")

    # Fallback: Groq
    try:
        groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        groq_messages += [m for m in context_messages if m["role"] != "system"]

        chat_completion = await groq_client.chat.completions.create(
            messages=groq_messages,
            model="llama-3.3-70b-versatile",
        )
        print("Response via Groq (fallback)")
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Groq fallback also failed: {e}")
        return "I'm having a little trouble right now - give me a moment and try again!"


def build_context_messages(conversation_history: list) -> list:
    """Build the full message list for the AI, with the system prompt prepended."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    return messages


def get_conversation_history(user_record) -> list:
    """
    Extract full conversation history from the user record.
    Stored as JSON in the conversation_history column.
    Falls back gracefully to legacy single-message format.
    """
    if not user_record:
        return []

    raw = user_record.get("conversation_history")
    if raw:
        try:
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, list):
                return raw
        except Exception:
            pass

    # Legacy fallback
    history = []
    if user_record.get("message"):
        history.append({"role": "user", "content": user_record["message"]})
    if user_record.get("ai_response"):
        history.append({"role": "assistant", "content": user_record["ai_response"]})
    return history


def append_to_history(history: list, user_msg: str, ai_reply: str) -> list:
    """Append a new turn and cap at 30 turns (60 messages) to avoid token overload."""
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ai_reply})
    return history[-60:]


async def check_reminders():
    """
    24h/48h follow-up scheduler.
    - 24h: caring check-in (reminders_sent 0 -> 1)
    - 48h: final reminder, then mark dormant (reminders_sent 1 -> 2), stop entirely
    """
    print("Checking for dormant leads...")
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
                except Exception:
                    continue

            hours_passed = (now - last_message_time).total_seconds() / 3600
            reminders_sent = lead.get("reminders_sent", 0)
            name = lead.get("name", "there")

            # 48h final follow-up -> mark dormant, no more messages after this
            if hours_passed >= 48 and reminders_sent == 1:
                message = (
                    f"Last reminder from my side, {name}!\n\n"
                    "July batch seats are almost full - I'd hate for you to miss this one.\n\n"
                    "Let me know if you'd like to secure your spot. No pressure!"
                )
                await send_whatsapp_message(lead["phone_number"], message)
                supabase.table("leads").update({
                    "reminders_sent": 2,
                    "status": "dormant"
                }).eq("id", lead["id"]).execute()

            # 24h caring check-in
            elif hours_passed >= 24 and reminders_sent == 0:
                message = (
                    f"Hey {name}! Just checking in - still thinking about SpeakLab?\n\n"
                    "A lot of students made their decision this week and seats are going fast.\n\n"
                    "Would love to have you in the July batch! Any questions I can help with?"
                )
                await send_whatsapp_message(lead["phone_number"], message)
                supabase.table("leads").update({
                    "reminders_sent": 1
                }).eq("id", lead["id"]).execute()

    except Exception as e:
        print(f"Error in check_reminders: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(check_reminders, "interval", hours=1)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="SpeakLab Sara Bot", lifespan=lifespan)


class BroadcastRequest(BaseModel):
    message: str
    phone_list: List[str]


async def send_whatsapp_message(to_phone: str, text: str):
    async with httpx.AsyncClient() as client:
        url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {"body": text},
        }
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"Failed to send WhatsApp message: {response.text}")
        return response


async def download_whatsapp_media(media_id: str):
    async with httpx.AsyncClient() as client:
        url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
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

        if WABA_ID and entry.get("id") != WABA_ID:
            print("Webhook entry ID does not match WABA_ID")
            return {"status": "error", "message": "Invalid WABA ID"}

        changes = entry.get("changes", [])[0]
        value = changes.get("value", {})

        if "messages" not in value:
            return {"status": "success"}

        for message in value["messages"]:
            sender_phone = message.get("from")
            message_type = message.get("type")
            message_text = ""

            if message_type == "text":
                message_text = message.get("text", {}).get("body", "")

            elif message_type == "audio":
                audio_id = message.get("audio", {}).get("id")
                audio_bytes = await download_whatsapp_media(audio_id)
                if not audio_bytes:
                    await send_whatsapp_message(
                        sender_phone,
                        "Sorry, I couldn't process your voice note. Could you type it instead?"
                    )
                    continue

                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio:
                    temp_audio.write(audio_bytes)
                    temp_audio_path = temp_audio.name

                try:
                    audio = OggVorbis(temp_audio_path)
                    if audio.info.length > 60:
                        await send_whatsapp_message(
                            sender_phone,
                            "That voice note is a bit long! Could you send a shorter one, or just type it? I want to catch everything!"
                        )
                        os.unlink(temp_audio_path)
                        continue
                except Exception as e:
                    print(f"Error checking audio length: {e}")

                try:
                    with open(temp_audio_path, "rb") as f:
                        transcription = await groq_client.audio.transcriptions.create(
                            file=(os.path.basename(temp_audio_path), f.read()),
                            model="whisper-large-v3-turbo",
                        )
                        message_text = transcription.text
                except Exception as e:
                    print(f"Whisper transcription error: {e}")
                    await send_whatsapp_message(
                        sender_phone,
                        "Sorry, I had trouble understanding that voice note. Could you type it?"
                    )
                    continue
                finally:
                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)

            else:
                continue

            if not message_text.strip():
                continue

            user_res = supabase.table("leads").select("*").eq("phone_number", sender_phone).execute()
            user_record = user_res.data[0] if user_res.data else None

            conversation_history = get_conversation_history(user_record)
            context_messages = build_context_messages(conversation_history)
            context_messages.append({"role": "user", "content": message_text})

            reply_text = await generate_ai_response(context_messages)

            lead_info = {}
            feedback_info = {}
            referral_info = {}
            clean_reply = reply_text

            lead_match = re.search(r"<LEAD_CAPTURED>(.*?)</LEAD_CAPTURED>", clean_reply, re.DOTALL)
            if lead_match:
                lead_str = lead_match.group(1)
                clean_reply = clean_reply.replace(lead_match.group(0), "").strip()
                for item in lead_str.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        lead_info[k.strip()] = v.strip()

                if OWNER_PHONE:
                    sir_message = (
                        f"NEW LEAD - SpeakLab\n\n"
                        f"Name: {lead_info.get('name', 'N/A')}\n"
                        f"Phone: {sender_phone}\n"
                        f"Background: {lead_info.get('background', 'N/A')}\n"
                        f"Interest: {lead_info.get('interest', 'N/A')}\n"
                        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        f"Follow up recommended!"
                    )
                    await send_whatsapp_message(OWNER_PHONE, sir_message)

            feedback_match = re.search(r"<FEEDBACK_CAPTURED>(.*?)</FEEDBACK_CAPTURED>", clean_reply, re.DOTALL)
            if feedback_match:
                feedback_str = feedback_match.group(1)
                clean_reply = clean_reply.replace(feedback_match.group(0), "").strip()
                for item in feedback_str.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        feedback_info[k.strip()] = v.strip()

            referral_match = re.search(r"<REFERRAL_CAPTURED>(.*?)</REFERRAL_CAPTURED>", clean_reply, re.DOTALL)
            if referral_match:
                referral_str = referral_match.group(1)
                clean_reply = clean_reply.replace(referral_match.group(0), "").strip()
                for item in referral_str.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        referral_info[k.strip()] = v.strip()

                if referral_info.get("phone"):
                    ref_phone = re.sub(r"\D", "", referral_info["phone"])
                    try:
                        supabase.table("leads").insert({
                            "phone_number": ref_phone,
                            "status": "referral",
                            "referral_by": sender_phone,
                            "last_message_time": datetime.now(timezone.utc).isoformat(),
                            "reminders_sent": 0,
                        }).execute()

                        referrer_name = (
                            lead_info.get("name")
                            or (user_record.get("name") if user_record else None)
                            or "A friend"
                        )
                        ref_msg = (
                            f"Hey! {referrer_name} thought you'd love SpeakLab.\n\n"
                            "I'm Sara - want me to tell you about our program?"
                        )
                        await send_whatsapp_message(ref_phone, ref_msg)
                    except Exception as e:
                        print(f"Error handling referral: {e}")

            updated_history = append_to_history(conversation_history, message_text, clean_reply)
            now_iso = datetime.now(timezone.utc).isoformat()

            update_data = {
                "message": message_text,
                "ai_response": clean_reply,
                "conversation_history": json.dumps(updated_history),
                "last_message_time": now_iso,
                "reminders_sent": 0,
            }

            if lead_info.get("name"):       update_data["name"]       = lead_info["name"]
            if lead_info.get("background"): update_data["background"] = lead_info["background"]
            if lead_info.get("interest"):   update_data["interest"]   = lead_info["interest"]
            if feedback_info.get("feedback"):
                update_data["feedback"] = feedback_info["feedback"]
            if feedback_info.get("shared"):
                update_data["feedback_shared"] = feedback_info["shared"].lower() == "true"

            try:
                if user_record:
                    if lead_info:
                        update_data["status"] = "enrolled"
                    elif user_record.get("status") == "new":
                        update_data["status"] = "interested"
                    supabase.table("leads").update(update_data).eq("id", user_record["id"]).execute()
                else:
                    update_data["phone_number"] = sender_phone
                    update_data["status"] = "new"
                    supabase.table("leads").insert(update_data).execute()
            except Exception as e:
                print(f"Error saving lead to Supabase: {e}")

            await send_whatsapp_message(sender_phone, clean_reply)

        return {"status": "success"}

    except Exception as e:
        print(f"Error processing webhook event: {e}")
        return {"status": "error"}


@app.post("/broadcast")
async def send_broadcast(request: BroadcastRequest):
    success_count = 0
    for phone in request.phone_list:
        try:
            full_msg = f"SpeakLab Update!\n\n{request.message}\n\n- SpeakLab Team"
            await send_whatsapp_message(phone, full_msg)
            success_count += 1
        except Exception as e:
            print(f"Broadcast failed for {phone}: {e}")

    try:
        supabase.table("broadcasts").insert({
            "message": request.message,
            "sent_to": request.phone_list,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        print(f"Error saving broadcast: {e}")

    return {"status": "success", "sent": success_count, "total": len(request.phone_list)}


@app.get("/")
async def root():
    return {"message": "SpeakLab Sara Bot is running!"}
