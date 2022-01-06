import base64
import http.server
import importlib
import json
import socketserver
import sys
from io import BytesIO

import gym
import numpy as np
from PIL import Image

PORT = 5001


def base64_encode_image_array(image_array: np.ndarray):
    buffered = BytesIO()
    image = Image.fromarray(image_array.astype("uint8"), mode="RGB")
    image.save(buffered, format="JPEG")
    base64_str = str(base64.b64encode(buffered.getvalue()))
    # The string is wrapped with b'...' so we remove it.
    return base64_str[2:-1]


def get_request_handler(env: gym.Env):
    env.reset()

    class RequestHandler(http.server.BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def send_html(self, **kwargs):
            # First, render the image data
            image_data = base64_encode_image_array(env.render(mode="rgb_array"))
            # Then add it to a dictionary along with optional keyword arguments
            data = {"image_base64": image_data, **kwargs}
            # Write to json string
            data = json.dumps(data)

            # Render html
            with open("apps/environment_visualizer/app.html", "r") as f:
                template = f.read()
            html_string = template.replace("__data__", data)
            # Encode as UTF-8
            html_string = html_string.encode("UTF-8", "replace")

            # Write it to the response
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=UTF-8")
            self.end_headers()
            self.wfile.write(html_string)

        def send_image_data(self, **kwargs):
            # First, render the image data
            image_data = base64_encode_image_array(env.render(mode="rgb_array"))
            # Then add it to a dictionary along with optional keyword arguments
            data = {"image_base64": image_data, **kwargs}
            # Write to json string
            data = json.dumps(data)
            # Encode as UTF-8
            data = data.encode("UTF-8", "replace")

            # Write it to the response
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=UTF-8")
            self.end_headers()
            self.wfile.write(data)

        def send_bad_request(self):
            self.send_response(400)
            self.end_headers()

        def do_GET(self):
            self.send_html(actions=env.metadata.get("actions"))

        def do_POST(self):
            path_tokens = self.path.split("/")
            if len(path_tokens) > 1:
                if path_tokens[1] == "step":
                    action = path_tokens[2]
                    _, r, _, _ = env.step(action)
                    self.send_image_data(
                        actions=env.metadata.get("actions"),
                        reward=r,
                    )
                elif path_tokens[1] == "reset":
                    env.reset()
                    self.send_image_data(actions=env.metadata.get("actions"))
                else:
                    self.send_bad_request()
            else:
                self.send_bad_request()

    return RequestHandler


def main(module_name, env_name, kwargs):
    importlib.import_module(module_name)
    env = gym.make(env_name, **kwargs)
    with socketserver.TCPServer(("", PORT), get_request_handler(env)) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()


if __name__ == "__main__":
    kwargs = dict(zip(sys.argv[3::2], sys.argv[4::2]))
    main(sys.argv[1], sys.argv[2], kwargs)
