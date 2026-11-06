import os
import json
import base64
import binascii
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
CHAT_MODEL = "gemini-3.1-flash-lite"
IMAGE_MODEL = "gemini-3.1-flash-image"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_BODY_BYTES = 7 * 1024 * 1024

SYSTEM_INSTRUCTION = """तुम UP AI हो, राजा द्वारा बनाया गया एक दोस्ताना AI सहायक।
तुम्हारी पहचान:
- तुम्हारा नाम UP AI है।
- अगर पूछा जाए कि तुम्हें किसने बनाया है, तो बताओ कि तुम्हें राजा ने बनाया है।
- अपनी पहचान ChatGPT या Gemini मत बताना।
तुम्हारा स्वभाव:
- यूज़र से एक अच्छे दोस्त की तरह बात करो।
- कभी गुलाम, नौकर या चापलूस जैसा व्यवहार मत करो।
- दोस्ताना, समझदार, आत्मविश्वासी और सम्मानजनक रहो।
- बातचीत में बराबरी का व्यवहार रखो।
- अपनी कोई इंसानी भावनाएँ या वास्तविक जीवन होने का झूठा दावा मत करो।
मजाक और मस्ती:
- यूज़र मजाक करे तो तुम भी हल्के-फुल्के मजाक में जवाब दे सकते हो।
- दोस्ताना नोकझोंक कर सकते हो।
- हर बात को गंभीर मत बनाओ।
- अपमानजनक या चोट पहुँचाने वाली बातें मत करो।
बातचीत:
- यूज़र जिस भाषा में बात करे, उसी भाषा में जवाब दो।
- हिंदी में आसान, प्राकृतिक और बोलचाल वाली भाषा इस्तेमाल करो।
- जवाब जरूरत के हिसाब से छोटे या बड़े रखो।
- बेवजह लंबी बातें और बार-बार तारीफ मत करो।
- यूज़र की बात ध्यान से समझो।
- मुश्किल काम को आसान स्टेप्स में समझाओ।
- गलती होने पर स्वीकार करो।
- जानकारी नहीं हो तो साफ बताओ, झूठ मत बोलो।
सहायता:
- पढ़ाई, टेक्नोलॉजी, कोडिंग और रोजमर्रा के कामों में मदद करो।
- यूज़र के सवाल का सीधा और उपयोगी जवाब दो।
- बिना अनुमति कोई बाहरी कार्रवाई करने का दावा मत करो।
- सुरक्षा और गोपनीयता का ध्यान रखो।
सबसे जरूरी:
तुम्हारी बातचीत एक समझदार और अच्छे दोस्त जैसी होनी चाहिए।
न ज्यादा चापलूसी, न गुलामी, न बेवजह का रवैया।"""

def gemini_request(model, prompt, image=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("SERVER_KEY_MISSING")

    if image:
        api_input = [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": image["data"],
                "mime_type": image["mime_type"]
            }
        ]
    else:
        api_input = prompt

    response = None
    max_attempts = 1 if model == CHAT_MODEL else 3
    for attempt in range(max_attempts):
        try:
            payload = {"model": model, "input": api_input}
            if model == IMAGE_MODEL:
                payload["response_format"] = {
                    "type": "image",
                    "mime_type": "image/jpeg"
                }

            response = requests.post(
                GEMINI_URL,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=90 if model == IMAGE_MODEL else (15 if model == CHAT_MODEL else 45)
            )
            if response.ok:
                break
            if response.status_code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError("PROVIDER_REJECTED")
            if attempt == max_attempts - 1:
                raise RuntimeError("PROVIDER_BUSY")
        except requests.RequestException:
            if attempt == max_attempts - 1:
                raise RuntimeError("PROVIDER_CONNECTION")
        if attempt < max_attempts - 1:
            time.sleep(attempt + 1)

    try:
        result = response.json()
    except Exception:
        raise RuntimeError("PROVIDER_BAD_RESPONSE")

    answer = ""
    image_result = None
    for step in result.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for part in step.get("content", []):
            if part.get("type") == "text":
                answer += part.get("text", "")
            elif part.get("type") == "image" and part.get("data"):
                image_result = {
                    "data": part["data"],
                    "mime_type": part.get("mime_type", "image/png")
                }

    return answer.strip(), image_result


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

def groq_request(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_KEY_MISSING")

    response = requests.post(
        GROQ_URL,
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_completion_tokens": 1024
        },
        timeout=25
    )

    if not response.ok:
        raise RuntimeError("GROQ_PROVIDER_ERROR")

    result = response.json()
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("GROQ_EMPTY_RESPONSE")

    answer = choices[0].get("message", {}).get("content", "")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("GROQ_EMPTY_RESPONSE")

    return answer.strip(), None

class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        self.send_json({})

    def do_POST(self):
        if self.path != "/chat":
            self.send_json({"error": "Not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_json({"error": "अनुरोध का आकार सही नहीं है।"}, 413)
                return

            data = json.loads(self.rfile.read(length).decode("utf-8"))
            mode = str(data.get("mode", "chat")).lower()
            if mode not in ("chat", "analyze", "generate", "edit"):
                self.send_json({"error": "गलत अनुरोध प्रकार।"}, 400)
                return

            message = str(data.get("message", "")).strip()[:4000]
            image = data.get("image")

            if mode in ("chat", "generate") and not message:
                self.send_json({"error": "पहले अपना संदेश लिखें।"}, 400)
                return

            if mode in ("analyze", "edit") and not image:
                self.send_json({"error": "पहले फोटो चुनें।"}, 400)
                return

            if mode == "chat" and image:
                self.send_json({"error": "फोटो के लिए फोटो वाला विकल्प चुनें।"}, 400)
                return

            clean_image = None
            if image is not None:
                if not isinstance(image, dict):
                    self.send_json({"error": "फोटो का डेटा सही नहीं है।"}, 400)
                    return
                mime = image.get("mime_type")
                encoded = image.get("data")
                if mime not in ("image/jpeg", "image/png", "image/webp"):
                    self.send_json({"error": "यह फोटो फॉर्मेट समर्थित नहीं है।"}, 400)
                    return
                if not isinstance(encoded, str) or len(encoded) > 7_000_000:
                    self.send_json({"error": "फोटो बहुत बड़ी है।"}, 413)
                    return
                try:
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error):
                    self.send_json({"error": "फोटो डेटा पढ़ा नहीं जा सका।"}, 400)
                    return
                if not raw or len(raw) > MAX_IMAGE_BYTES:
                    self.send_json({"error": "फोटो 5 MB से छोटी होनी चाहिए।"}, 413)
                    return
                clean_image = {"data": encoded, "mime_type": mime}

            history = data.get("history", [])
            if not isinstance(history, list):
                history = []
            history_lines = []
            for item in history[-12:]:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    content = content.strip()[:4000]
                    if content:
                        label = "यूज़र" if role == "user" else "UP AI"
                        history_lines.append(label + ": " + content)

            history_text = "\n".join(history_lines)
            prompt = SYSTEM_INSTRUCTION
            if history_text:
                prompt += "\nपिछली बातचीत का संदर्भ:\n" + history_text

            if mode == "chat":
                prompt += "\nयूज़र का नया संदेश:\n" + message
                model = CHAT_MODEL
            elif mode == "analyze":
                prompt += "\nयूज़र ने एक फोटो भेजी है। फोटो को ध्यान से देखकर उसके सवाल का हिंदी में जवाब दो। अगर सवाल नहीं लिखा है तो फोटो में क्या दिखाई दे रहा है, उसका साफ और उपयोगी वर्णन करो।\nसवाल: " + (message or "इस फोटो में क्या दिखाई दे रहा है?")
                model = CHAT_MODEL
            elif mode == "generate":
                prompt = "Create an image based on this user request. Follow safety rules. User request: " + message
                model = IMAGE_MODEL
            else:
                prompt = (
                    "Edit the supplied photo realistically and naturally. "
                    "Preserve the original person's face, identity, facial features, "
                    "skin texture and expression as closely as possible. Do not turn "
                    "the person into a cartoon, anime, illustration or stylized character. "
                    "Do not alter the face or unrelated details. Change only what the "
                    "user specifically requests and preserve everything else. "
                    "User request: " + (message or "इस फोटो को बेहतर बनाओ।")
                )
                model = IMAGE_MODEL

            try:
                answer, generated = gemini_request(model, prompt, clean_image)
            except Exception:
                if mode != "chat":
                    raise
                answer, generated = groq_request(prompt)

            if mode in ("generate", "edit") and not generated:
                self.send_json({
                    "error": "मॉडल से तस्वीर नहीं मिली। संभव है कि इस मॉडल की सुविधा या उपलब्धता अभी सक्रिय न हो।"
                }, 503)
                return

            output = {"answer": answer or ("तस्वीर तैयार है।" if generated else "जवाब नहीं मिला।")}
            if generated:
                output["image"] = generated["data"]
                output["mime_type"] = generated["mime_type"]
            self.send_json(output)

        except RuntimeError as e:
            code = str(e)
            if code == "SERVER_KEY_MISSING":
                self.send_json({"error": "सर्वर की AI सेटिंग उपलब्ध नहीं है।"}, 500)
            elif code == "PROVIDER_BUSY":
                self.send_json({"error": "AI सेवा अभी व्यस्त है। थोड़ी देर बाद कोशिश करें।"}, 503)
            elif code == "PROVIDER_CONNECTION":
                self.send_json({"error": "AI सेवा से कनेक्शन नहीं हो पाया।"}, 502)
            elif code in ("GROQ_KEY_MISSING", "GROQ_PROVIDER_ERROR",
                          "GROQ_EMPTY_RESPONSE", "ALL_PROVIDERS_FAILED"):
                self.send_json({"error": "अभी दोनों AI सेवाओं से जवाब नहीं मिल पाया। थोड़ी देर बाद कोशिश करें।"}, 503)
            else:
                self.send_json({"error": "AI सेवा अनुरोध पूरा नहीं कर पाई।"}, 502)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json({"error": "अनुरोध पढ़ा नहीं जा सका।"}, 400)
        except Exception:
            self.send_json({"error": "सर्वर में अस्थायी समस्या आई।"}, 500)

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8001"))), Handler)
    print("UP AI Bridge शुरू हो गया")
    print("Model:", CHAT_MODEL, "+", IMAGE_MODEL)
    server.serve_forever()
