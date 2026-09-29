import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
MODEL = "gemini-3.1-flash-lite"
HISTORY = []

SYSTEM_INSTRUCTION = """
तुम UP AI हो।
तुम्हें राजा ने बनाया है।
अगर पूछा जाए कि तुम्हें किसने बनाया है, तो बताओ: मुझे राजा ने बनाया है।
अगर पूछा जाए कि राजा कहाँ के हैं, तो बताओ: जिला इटावा, भवानीपुर के राजा।

यूज़र जिस भाषा में पूछे, उसी भाषा में जवाब देने की कोशिश करो।
जवाब साफ, मददगार और आसान भाषा में दो।
"""


class Handler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type"
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "POST, OPTIONS"
        )
        self.send_header(
            "Content-Length",
            str(len(body))
        )
        self.end_headers()

        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_json({})

    def do_POST(self):

        if self.path != "/chat":
            self.send_json(
                {"error": "Not found"},
                404
            )
            return

        try:
            length = int(
                self.headers.get("Content-Length", 0)
            )

            raw_data = self.rfile.read(length)

            data = json.loads(
                raw_data.decode("utf-8")
            )

            user_message = str(
                data.get("message", "")
            ).strip()

            if not user_message:
                self.send_json(
                    {"error": "Message खाली है"},
                    400
                )
                return

            api_key = os.environ.get(
                "GEMINI_API_KEY"
            )

            if not api_key:
                self.send_json(
                    {
                        "error": (
                            "GEMINI_API_KEY नहीं मिली।"
                        )
                    },
                    500
                )
                return

            HISTORY.append({"user": user_message})
            history_text = "\n".join(str(x) for x in HISTORY[-10:])

            prompt = (
                SYSTEM_INSTRUCTION
                + "\nपिछली बातचीत:\n"
                + history_text
            )

            response = requests.post(
                GEMINI_URL,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "input": prompt
                },
                timeout=120
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

                for content in step.get(
                    "content", []
                ):
                    if content.get("type") == "text":
                        answer += content.get(
                            "text", ""
                        )

            if not answer:
                self.send_json(
                    {
                        "error": (
                            "Gemini से जवाब नहीं मिला।"
                        ),
                        "details": result
                    },
                    500
                )
                return

            answer = answer.strip()
            HISTORY.append({"assistant": answer})

            self.send_json(
                {
                    "answer": answer
                }
            )

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
        ("127.0.0.1", 8001),
        Handler
    )

    print("🤖 UP AI Bridge शुरू हो गया")
    print("📡 http://127.0.0.1:8001")
    print("🧠 Model: gemini-3.8-flash")

    server.serve_forever()
