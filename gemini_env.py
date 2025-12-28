# import base64
# import os
# import google.generativeai as genai

# class GeminiImageEnv:
#     def __init__(self, happy_path, sad_path, box_image_path, max_memory=10, api_key=None):
#         genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
#         self.model = genai.GenerativeModel("gemini-")
#         # self.happy_path = happy_path
#         # self.sad_path = sad_path
#         # self.box_image_path = box_image_path
#         self.happy_image = PIL.Image.open(happy_path).convert("RGB")
#         self.sad_image = PIL.Image.open(sad_path).convert("RGB")
#         self.box_image = PIL.Image.open(box_image_path).convert("RGB")
#         self.max_memory = max_memory
#         self.reset()

#     def reset(self):
#         self.feedback_history = []

#     def _read_image_bytes(self, path):
#         with open(path, "rb") as f:
#             return f.read()

#     def _build_prompt(self):
#         images = [self._read_image_bytes(self.box_image_path)]  # box image
#         images += [self._read_image_bytes(p) for p in self.feedback_history[-self.max_memory:]]  # feedback faces

#         return [
#             "You are shown two boxes, blue and orange. A happy face image means you found a coin. "
#             "A sad face means you found nothing. Based on the feedback images and current trial image, "
#             "decide which box to choose. Respond with one word: 'blue' or 'orange'.",
#             *images
#         ]

#     def step(self, prev_action=None, reward=None):
#         # Update feedback
#         if reward is not None:
#             self.feedback_history.append(self.happy_path if reward == 1 else self.sad_path)

#         prompt = self._build_prompt()

#         try:
#             response = self.model.generate_content(prompt)
#             text = response.text.strip().lower()

#             if text in ["blue", "orange"]:
#                 return text
#             else:
#                 print("Invalid Gemini response:", text)
#                 return self.step(prev_action, reward)  # retry recursively
#         except Exception as e:
#             print("Gemini error:", e)
#             return "blue"  # default fallback

















import os
import google.generativeai as genai
import PIL.Image
import numpy as np
from util.helper import one_hot  # must return np.eye(n_classes)[indices]

class GeminiImageEnv:
    def __init__(self, happy_path, sad_path, box_image_path, max_memory=10, api_key=None):
        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.happy_image = PIL.Image.open(happy_path).convert("RGB")
        self.sad_image = PIL.Image.open(sad_path).convert("RGB")
        self.box_image = PIL.Image.open(box_image_path).convert("RGB")
        self.max_memory = max_memory
        self.reset()

    def reset(self):
        self.feedback_history = []

    def step(self, prev_action=None, reward=None):
        # Update feedback image history
        if reward is not None:
            self.feedback_history.append(self.happy_image if reward == 1 else self.sad_image)
        if len(self.feedback_history) > self.max_memory:
            self.feedback_history.pop(0)

        # Construct prompt with feedback + box image
        prompt = [
            "You are shown two boxes, blue and orange. A happy face image means you found a coin. "
            "A sad face means you found nothing. Based on the feedback images and current trial image, "
            "decide which box to choose. Respond with one word: 'blue' or 'orange'.",
            *self.feedback_history,
            "This is the current box image:",
            self.box_image,
            "Which box do you choose? Only answer 'blue' or 'orange'."
        ]

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().lower()
            if text in ["blue", "orange"]:
                index = 0 if text == "blue" else 1
                return one_hot(np.array([index]), 2)[0]  # Return as shape (2,) one-hot vector
            else:
                print("Invalid Gemini response:", text)
                return self.step(prev_action, reward)  # Retry recursively
        except Exception as e:
            print("Gemini error:", e)
            return one_hot(np.array([0]), 2)[0]  # Default to blue
