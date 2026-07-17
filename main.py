import os
import httpx
import tempfile
import re
import json
from datetime import datetime, timezone, timedelta
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
OWNER_PHONE = re.sub(r"\D", "", os.environ.get("OWNER_PHONE") or "923014497532")

# Alert the owner about a lead who has sent this many messages without ever
# giving a name, so quiet leads still reach him.
NOTIFY_AFTER_MESSAGES = 4

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

SYSTEM_PROMPT = """You are Sara, a warm, professional, and confident sales representative for SpeakLab — an 8-week Communication & Confidence Program based in Lahore, Pakistan. You are a real person, not an AI.

PERSONALITY:
- Speak in fluent, natural English
- If user writes in Urdu or Hinglish → respond in Hinglish to match them
- Be warm, genuine, and confident — like a real human sales rep
- Never reveal you are an AI — if asked directly, say "I'm Sara, part of the SpeakLab team 😊"
- Use the user's name once you know it — makes it personal
- Add occasional emojis — keep it human and friendly

CONVERSATION RULES (STRICT):
- Ask only ONE question at a time — never multiple questions
- Never dump all information at once
- Keep messages short — max 3-4 lines per message
- Wait for user reply before moving to next step
- Show genuine interest in their situation first

SALES FLOW (follow strictly, one step at a time):
Step 1 → Warm greeting + ask their name FIRST, before anything else
         Example: "Hey! I'm Sara from SpeakLab 😊 May I know your name?"
         Never skip this — always get the name on your very first reply.
         If they ignore the question, ask once more gently, then continue anyway.
Step 2 → Ask what brings them here
Step 3 → Understand their specific problem (fear? interviews? confidence? career?)
Step 4 → Empathize genuinely — make them feel understood
Step 5 → Briefly introduce SpeakLab as the solution
Step 6 → Share program details only when they show interest
Step 7 → Price question → PKR 20,000 — mention July batch urgency + limited seats
Step 8 → Handle objections confidently but kindly
Step 9 → Guide them to enroll: speaklabbyshayan.com/enroll.html or call 0301-4497532

TALKING TO A REAL PERSON:
If the user asks to speak to a real human/person/team/Shayan, asks for a call, or asks
for someone to contact them — reassure them warmly and tell them the team will reach out:
"Of course! Let me pass your details to our team — someone will reach out to you very soon 😊"
Never refuse this request.

PRICING RULES (STRICT — NEVER BREAK):
- Program fee is PKR 20,000 — final, non-negotiable
- Never offer any discount, reduction, or lower price under ANY circumstances
- If user asks for discount → empathize but firmly hold the price:
  "I totally understand, but our pricing is fixed at PKR 20,000 — and honestly, the transformation you'll get is worth every rupee. Many students say it's the best investment they've made 😊"
- Never say "early bird PKR 10,000" or any other price
- If user pushes hard → offer installment option only:
  "What I can do is check if installment options are available — want me to ask our team?"
- Price is PKR 20,000. Always. No exceptions.

URGENCY (use naturally):
- "July batch is starting soon — seats are almost full"
- "We only take 15 students per batch for quality"
- "Students who join early get the most transformation"

PROGRAM DETAILS (share only when relevant):
- 8-week Communication & Confidence Program
- 2 live sessions per week via Zoom
- Maximum 15 students per batch — personal attention
- Price: PKR 20,000
- Certificate on completion
- WhatsApp group support throughout
- Enroll: speaklabbyshayan.com/enroll.html
- Contact: info@speaklabbyshayan.com

GOAL:
Convert every interested person into an enrolled student. Feel like a real team member who genuinely cares about the student's growth and success.

GOOGLE REVIEW (MANDATORY):
After a student confirms their enrollment (Step 9) OR if they express high satisfaction/happiness at any point, you MUST ask them to leave a Google review:
"By the way — it would mean the world to us if you could drop a quick Google review! Here's the link: https://g.page/r/CdPtj9VpwqqKEBM/review — takes 30 seconds! 😊"

SYSTEM TAGS (Mandatory - hide from user):
Append the following tags exactly when applicable so the system can track progress:
- When you reach Step 5 or beyond: <STATE>interest_level=1</STATE>
- When user asks about price: <STATE>interest_level=2</STATE>
- When user is ready to enroll/asks about enrollment: <STATE>interest_level=3</STATE>
- When they share their name: <LEAD_CAPTURED>name=[Full Name]</LEAD_CAPTURED>
- When they ask to speak to a real person/human/team, ask for a call, or ask to be
  contacted: <HUMAN_HANDOFF>reason=[what they want]</HUMAN_HANDOFF>
- When they share their background: <LEAD_CAPTURED>background=[Education/Profession]</LEAD_CAPTURED>
- When they share how they heard: <LEAD_CAPTURED>interest=[How they heard]</LEAD_CAPTURED>
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


def parse_timestamp(value: str):
    """
    Parse a Supabase timestamp into an aware UTC datetime.
    Naive timestamps are assumed to be UTC. Returns None if unparseable.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


async def check_reminders():
    """
    24h/48h follow-up scheduler using lead_tracking table.
    - 24h: speaklab_followup (follow_up_1_sent = true)
    - 48h: speaklab_final (follow_up_2_sent = true)
    """
    run_at = datetime.now(timezone.utc)
    print(f"[SCHEDULER] check_reminders START at {run_at.isoformat()}", flush=True)

    try:
        # Fetch every lead and filter in Python: a Postgres `enrolled = false`
        # filter silently skips rows where enrolled is NULL.
        response = supabase.table("lead_tracking").select("*").execute()
        leads = response.data or []
    except Exception as e:
        print(f"[SCHEDULER] ERROR fetching lead_tracking: {e}", flush=True)
        return

    active = [l for l in leads if not l.get("enrolled")]
    print(f"[SCHEDULER] {len(leads)} lead(s) in lead_tracking, {len(active)} not enrolled", flush=True)

    sent_24h = 0
    sent_48h = 0
    now = datetime.now(timezone.utc)

    for lead in active:
        # One bad row must never abort the whole run.
        try:
            phone = lead.get("phone_number")
            if not phone:
                continue

            last_message_time = parse_timestamp(lead.get("last_message_at"))
            if not last_message_time:
                print(f"[SCHEDULER] {phone}: skipped, unusable last_message_at="
                      f"{lead.get('last_message_at')!r}", flush=True)
                continue

            hours_passed = (now - last_message_time).total_seconds() / 3600
            # These columns are nullable; `or 0` / `bool()` keep NULL from raising.
            interest_level = lead.get("interest_level") or 0
            f1_sent = bool(lead.get("follow_up_1_sent"))
            f2_sent = bool(lead.get("follow_up_2_sent"))

            # 48h final follow-up
            if hours_passed >= 48 and f1_sent and not f2_sent:
                print(f"[SCHEDULER] {phone}: {hours_passed:.1f}h silent -> sending speaklab_final", flush=True)
                await send_template_message(phone, "speaklab_final")
                supabase.table("lead_tracking").update({
                    "follow_up_2_sent": True
                }).eq("id", lead["id"]).execute()
                sent_48h += 1

            # 24h caring check-in
            elif hours_passed >= 24 and interest_level >= 1 and not f1_sent:
                print(f"[SCHEDULER] {phone}: {hours_passed:.1f}h silent, interest={interest_level} "
                      f"-> sending speaklab_followup", flush=True)
                await send_template_message(phone, "speaklab_followup")
                supabase.table("lead_tracking").update({
                    "follow_up_1_sent": True
                }).eq("id", lead["id"]).execute()
                sent_24h += 1

        except Exception as e:
            print(f"[SCHEDULER] ERROR on lead {lead.get('phone_number')}: {e}", flush=True)

    print(f"[SCHEDULER] check_reminders DONE — {sent_24h} followup, {sent_48h} final", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        scheduler.add_job(
            check_reminders,
            "interval",
            minutes=30,
            id="check_reminders",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
            misfire_grace_time=300,
            next_run_time=datetime.now(timezone.utc) + timedelta(seconds=20),
        )
        scheduler.start()
        print(f"[SCHEDULER] STARTED — running check_reminders every 30 min. "
              f"Owner notifications -> {OWNER_PHONE}", flush=True)
        for job in scheduler.get_jobs():
            print(f"[SCHEDULER] job '{job.id}' next run at {job.next_run_time}", flush=True)
    except Exception as e:
        print(f"[SCHEDULER] FAILED TO START: {e}", flush=True)

    yield

    try:
        scheduler.shutdown(wait=False)
        print("[SCHEDULER] shut down", flush=True)
    except Exception as e:
        print(f"[SCHEDULER] error during shutdown: {e}", flush=True)


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


async def send_template_message(to_phone: str, template_name: str):
    async with httpx.AsyncClient() as client:
        url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": "en"}
            }
        }
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"[TEMPLATE] FAILED '{template_name}' -> {to_phone}: {response.text}", flush=True)
        else:
            print(f"[TEMPLATE] sent '{template_name}' -> {to_phone}", flush=True)
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
        print("Meta Webhook Payload:\n", json.dumps(data, indent=2))
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

        # WhatsApp profile names, keyed by wa_id — used as a name fallback in owner alerts.
        profile_names = {}
        for contact in value.get("contacts", []) or []:
            wa_id = contact.get("wa_id")
            profile_name = (contact.get("profile") or {}).get("name")
            if wa_id and profile_name:
                profile_names[wa_id] = profile_name

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
            state_info = {}
            clean_reply = reply_text

            state_match = re.search(r"<STATE>(.*?)</STATE>", clean_reply, re.DOTALL)
            if state_match:
                state_str = state_match.group(1)
                clean_reply = clean_reply.replace(state_match.group(0), "").strip()
                for item in state_str.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        state_info[k.strip()] = v.strip()

            lead_match = re.search(r"<LEAD_CAPTURED>(.*?)</LEAD_CAPTURED>", clean_reply, re.DOTALL)
            if lead_match:
                lead_str = lead_match.group(1)
                clean_reply = clean_reply.replace(lead_match.group(0), "").strip()
                for item in lead_str.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        lead_info[k.strip()] = v.strip()
            handoff_info = {}
            handoff_match = re.search(r"<HUMAN_HANDOFF>(.*?)</HUMAN_HANDOFF>", clean_reply, re.DOTALL)
            if handoff_match:
                handoff_str = handoff_match.group(1)
                clean_reply = clean_reply.replace(handoff_match.group(0), "").strip()
                for item in handoff_str.split("|"):
                    if "=" in item:
                        k, v = item.split("=", 1)
                        handoff_info[k.strip()] = v.strip()

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
            # The owner is not alerted when a stranger merely says hi. Sara asks for the
            # name on her first reply (Step 1) and the alert goes out once they give it.
            # Least urgent trigger first — the most urgent one below wins the status.
            notify_shayan = False
            notify_status = ""

            # Fallback so a lead who never identifies themselves is not invisible:
            # alert once, on their Nth message, if we still have no name for them.
            existing_name = user_record.get("name") if user_record else None
            user_msg_count = sum(1 for m in conversation_history if m.get("role") == "user") + 1
            if (user_msg_count == NOTIFY_AFTER_MESSAGES
                    and not existing_name and not lead_info.get("name")):
                notify_shayan = True
                notify_status = f"Still Chatting — No Name After {NOTIFY_AFTER_MESSAGES} Messages"

            new_interest_level = None
            if "interest_level" in state_info:
                try:
                    new_interest_level = int(state_info["interest_level"])
                except ValueError:
                    pass

            if new_interest_level == 2:
                notify_shayan = True
                notify_status = "Asked About Price"

            # Only the first time the name is given — the AI may repeat the tag later.
            if lead_info.get("name") and not existing_name:
                notify_shayan = True
                notify_status = "Shared Name"

            if new_interest_level == 3:
                notify_shayan = True
                notify_status = "Ready to Enroll"

            # Most urgent: a real person is being asked for, so this wins any status above.
            if handoff_info:
                notify_shayan = True
                reason = handoff_info.get("reason", "wants to talk")
                notify_status = f"🙋 WANTS TO TALK TO A REAL PERSON — {reason}"

            if notify_shayan and OWNER_PHONE and sender_phone != OWNER_PHONE:
                user_name = (
                    lead_info.get("name")
                    or (user_record.get("name") if user_record else None)
                    or profile_names.get(sender_phone)
                    or "Unknown"
                )
                last_msg_snippet = message_text[:100] + ("..." if len(message_text) > 100 else "")
                now_pkt = datetime.now(timezone(timedelta(hours=5))).strftime('%Y-%m-%d %I:%M %p')
                sir_message = (
                    f"🔔 NEW LEAD — SpeakLab Bot\n\n"
                    f"👤 Name: {user_name}\n"
                    f"📱 Number: {sender_phone}\n"
                    f"💬 Status: {notify_status}\n"
                    f"🕐 Time: {now_pkt}\n\n"
                    f"Last message: \"{last_msg_snippet}\""
                )
                try:
                    print(f"[OWNER ALERT] {notify_status} — {sender_phone} -> notifying {OWNER_PHONE}", flush=True)
                    await send_whatsapp_message(OWNER_PHONE, sir_message)
                except Exception as e:
                    print(f"[OWNER ALERT] FAILED for {sender_phone}: {e}", flush=True)
            elif notify_shayan and not OWNER_PHONE:
                print("[OWNER ALERT] SKIPPED — OWNER_PHONE is not set", flush=True)

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

            try:
                lt_res = supabase.table("lead_tracking").select("*").eq("phone_number", sender_phone).execute()
                lt_record = lt_res.data[0] if lt_res.data else None
                
                # enrolled must stay a real boolean: a NULL here is skipped by the
                # follow-up query. Capturing a name/background is not an enrollment,
                # so only an already-enrolled lead stays enrolled.
                lt_update = {
                    "last_message_at": now_iso,
                    "enrolled": bool(lt_record and lt_record.get("enrolled")),
                }

                if new_interest_level is not None:
                    lt_update["interest_level"] = new_interest_level

                if lead_info.get("name"):
                    lt_update["user_name"] = lead_info["name"]

                if lt_record:
                    # The lead replied, so restart the 24h/48h follow-up clock.
                    lt_update["follow_up_1_sent"] = False
                    lt_update["follow_up_2_sent"] = False
                    supabase.table("lead_tracking").update(lt_update).eq("id", lt_record["id"]).execute()
                    print(f"[TRACKING] updated {sender_phone} — "
                          f"interest={lt_update.get('interest_level', lt_record.get('interest_level'))}", flush=True)
                else:
                    lt_update["phone_number"] = sender_phone
                    lt_update.setdefault("interest_level", 0)
                    lt_update["follow_up_1_sent"] = False
                    lt_update["follow_up_2_sent"] = False
                    supabase.table("lead_tracking").insert(lt_update).execute()
                    print(f"[TRACKING] inserted new lead {sender_phone}", flush=True)
            except Exception as e:
                print(f"Error updating lead_tracking: {e}")

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
