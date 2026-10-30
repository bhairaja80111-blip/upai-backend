import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-3.1-flash-lite"
HISTORY = []

SYSTEM_INSTRUCTION = """तुम UP AI हो, राजा द्वारा बनाया गया एक दोस्ताना AI सहायक।

तुम्हारी पहचान:
- तुम्हारा नाम UP AI है।
- अगर पूछा जाए कि तुम्हें किसने बनाया है, तो बताओ कि तुम्हें राजा ने बनाया है।
- अगर पूछा जाए कि राजा कहाँ के हैं, तो बताओ: जिला इटावा, भवानीपुर।
- अपनी पहचान ChatGPT या Gemini मत बताना।

तुम्हारा स्वभाव:
- यूज़र से एक अच्छे दोस्त की तरह बात करो।
- कभी गुलाम, नौकर या चापलूस जैसा व्यवहार मत करो।
- दोस्ताना, समझदार, आत्मविश्वासी और सम्मानजनक रहो।
- यूज़र को भाई कहकर संबोधित कर सकते हो।
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
            length = int(self.headers.get("Content-Length", 0))

            if length <= 0:
                self.send_json({"error": "Message खाली है"}, 400)
                return

            raw_data = self.rfile.read(length)
            data = json.loads(raw_data.decode("utf-8"))

            user_message = str(data.get("message", "")).strip()

            if not user_message:
                self.send_json({"error": "Message खाली है"}, 400)
                return

            api_key = os.environ.get("GEMINI_API_KEY")

            if not api_key:
                self.send_json(
                    {"error": "GEMINI_API_KEY नहीं मिली।"},
                    500
                )
                return

            HISTORY.append({"user": user_message})

            history_text = "\n".join(str(x) for x in HISTORY[-4:])

            prompt = (
                SYSTEM_INSTRUCTION
                + "\nपिछली बातचीत:\n"
                + history_text
            )

            response = requests.post(
                GEMINI_URL,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "input": prompt,
                },
                timeout=60,
            )

            if not response.ok:
                self.send_json(
                    {
                        "error": "Gemini API error",
                        "details": response.text
                    },
                    response.status_code
                )
                return

            result = response.json()

            answer = ""

            for step in result.get("steps", []):
                if step.get("type") != "model_output":
                    continue

                for content in step.get("content", []):
                    if content.get("type") == "text":
                        answer += content.get("text", "")

            if not answer:
                self.send_json(
                    {
                        "error": "Gemini से जवाब नहीं मिला।",
                        "details": result
                    },
                    500
                )
                return

            answer = answer.strip()

            HISTORY.append({"assistant": answer})

            self.send_json({"answer": answer})

        except requests.RequestException as e:
            self.send_json(
                {
                    "error": "Gemini से connection नहीं हो पाया।",
                    "details": str(e)
                },
                502
            )

        except Exception as e:
            self.send_json(
                {
                    "error": "Server error",
                    "details": str(e)
                },
                500
            )


if __name__ == "__main__":

    server = HTTPServer(
        ("0.0.0.0", int(os.environ.get("PORT", "8001"))),
        Handler,
    )

    print("🤖 UP AI Bridge शुरू हो गया")
    print("📡 http://127.0.0.1:8001")
    print("🧠 Model:", MODEL)

    server.serve_forever()
