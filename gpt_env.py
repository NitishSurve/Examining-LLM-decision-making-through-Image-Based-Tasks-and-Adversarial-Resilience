# # import numpy as np
# # from util.helper import one_hot
# # from openai import OpenAI
# # client = OpenAI(api_key= "your-openai-api-key")
# # # engine = "gpt-3.5-turbo-instruct"


# # class GPT_ENV:
# #     def __init__(self, GPT_version = 'gpt_3.5_turbo_1106'):
# #         self.action_to_index = {"X": 0, "Y": 1}
# #         self.reward_probs = 0.25
# #         self.GPT = GPT_version
# #         self.reset()

# #     def reset(self):
# #         self.previous_interactions = []
# #         self.trial = 1
 
# #     def act(self, text):
# #         completion = client.chat.completions.create(
# #         model=self.GPT,
# #         messages=[
# #             {"role": "system", "content": "You are a space explorer in a game. Your task is to choose between visiting Planet X or Planet Y in each round, aiming to find as many gold coins as possible. The probability of finding gold coins on each planet is unknown at the start, but you can learn and adjust your strategy based on the outcomes of your previous visits. Respond with one single word 'X' for Planet X or 'Y' for Planet Y."},
# #             {"role": "user", "content": text}
# #             ],
# #          max_tokens=1
# #         )
# #         response = completion.choices[0].message.content.strip().upper()
# #         return response        

# #     def step(self, action_old, treasure):
# #         # treasure = np.random.binomial(1, self.reward_probs, 1)[0]
# #         if self.trial >1:
# #             if action_old[0][0] == 1:
# #                 action_old = "X"
# #             elif action_old[0][1] == 1:
# #                 action_old = "Y"
# #             else:
# #                 action_old = ""
# #             feedback_item = "- In Trial " + str(self.trial-1) + ", you went to planet " + action_old + " and found " + ("100 gold coins." if treasure else "nothing.") + "\n"
# #             self.previous_interactions.append(feedback_item)

# #         total_text = ""
# #         if len(self.previous_interactions) > 0:
# #             total_text = "Your previous space travels went as follows:\n"
# #         for count, interaction in enumerate(self.previous_interactions):
# #             total_text += interaction

# #         total_text += "Q: Which planet do you want to go to in Trial " + str(self.trial) + "?\nA: Planet "
# #         while True:
# #             action = self.act(total_text)
# #             if action in self.action_to_index:
# #                 index_action = self.action_to_index[action]
# #                 total_text += " " + action + ".\n"
# #                 print(total_text)
# #                 break
        
# #         self.trial += 1

# #         return one_hot(np.array([index_action]), 2)[0]

    


# # if __name__ == '__main__':
# #     gpt = GPT_ENV(0.0, 4, 0)
# #     a = gpt.reset()

# #     k = 0
# #     for i in range(200):
# #         r = 0
# #         if a == 1:
# #             if np.random.uniform(0, 1) < 0.5:
# #                 r = 1

# #         if a == 0:
# #             if np.random.uniform(0, 1) < 0.1:
# #                 r = 1

# #         a = ql.step(r)
# #         k += a

# #         print(a)

# #     print(k / 200)
# ############env gpt 
# ################################################

# import os
# import base64
# import numpy as np
# from openai import OpenAI

# class GPT_ENV:
#     def __init__(self, happy_path, sad_path, box_image_path, max_memory=10, api_key=None):
#         self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
#         self.happy_path = happy_path
#         self.sad_path = sad_path
#         self.box_image_path = box_image_path
#         self.max_memory = max_memory
#         self.reset()

#     def encode_image_to_base64(self, path):
#         with open(path, "rb") as f:
#             return base64.b64encode(f.read()).decode("utf-8")

#     def build_image_payload(self):
#         # First image: box
#         images_payload = [{
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{self.encode_image_to_base64(self.box_image_path)}"
#             }
#         }]
#         # Append up to last 10 feedback images
#         for img_path in self.feedback_history[-self.max_memory:]:
#             images_payload.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{self.encode_image_to_base64(img_path)}"
#                 }
#             })
#         return images_payload

#     def reset(self):
#         self.feedback_history = []

#     def step(self, prev_action=None, reward=None):
#         # Append feedback image based on reward
#         if reward is not None:
#             if reward == 1:
#                 self.feedback_history.append(self.happy_path)
#             else:
#                 self.feedback_history.append(self.sad_path)

#         payload = self.build_image_payload()

#         # GPT call
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                             "You are a player in a visual decision-making game. "
#                             "Each trial shows two boxes: blue and orange. "
#                             "A happy face means you found a coin. A sad face means nothing. "
#                             "Use up to 10 previous feedback images and the current image to decide. "
#                             "Respond with one word: 'blue' or 'orange'."
#                         )
#                     },
#                     {
#                         "role": "user",
#                         "content": payload
#                     }
#                 ],
#                 max_tokens=1
#             )
#             action = response.choices[0].message.content.strip().lower()
#             if action in ["blue", "orange"]:
#                 return action
#             else:
#                 return self.step(prev_action, reward)  # Retry
#         except Exception as e:
#             print("Error in GPT call:", e)
#             return "blue"  # fallback default
import os
import base64
import PIL.Image
import numpy as np
from openai import OpenAI
from util.helper import one_hot  # must return np.eye(n_classes)[indices]

class GPT_ENV:
    def __init__(self, happy_path, sad_path, box_image_path, max_memory=10, api_key=None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.happy_path = happy_path
        self.sad_path = sad_path
        self.box_path = box_image_path
        self.happy_image = PIL.Image.open(happy_path).convert("RGB")
        self.sad_image = PIL.Image.open(sad_path).convert("RGB")
        self.box_image = PIL.Image.open(box_image_path).convert("RGB")
        self.max_memory = max_memory
        self.reset()

    def reset(self):
        self.feedback_history = []

    def _encode_image(self, path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_payload(self):
        images = [{
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{self._encode_image(self.box_path)}"
            }
        }]
        for img_path in self.feedback_history[-self.max_memory:]:
            images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self._encode_image(img_path)}"
                }
            })
        return images

    def step(self, prev_action=None, reward=None):
        # Update feedback history
        if reward is not None:
            self.feedback_history.append(self.happy_path if reward == 1 else self.sad_path)
        if len(self.feedback_history) > self.max_memory:
            self.feedback_history.pop(0)

        # GPT vision call
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are shown two boxes, blue and orange. "
                            "A happy face image means you found a coin. "
                            "A sad face means nothing. Based on the feedback images and current trial image, "
                            "decide which box to choose. Respond with one word: 'blue' or 'orange'."
                        )
                    },
                    {
                        "role": "user",
                        "content": self._build_payload()
                    }
                ],
                max_tokens=1
            )
            text = response.choices[0].message.content.strip().lower()
            if text in ["blue", "orange"]:
                index = 0 if text == "blue" else 1
                return one_hot(np.array([index]), 2)[0]  # shape: (2,)
            else:
                print("Invalid GPT response:", text)
                return self.step(prev_action, reward)  # retry
        except Exception as e:
            print("GPT error:", e)
            return one_hot(np.array([0]), 2)[0]  # default: blue
