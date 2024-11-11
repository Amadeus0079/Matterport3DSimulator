import openai
import base64
import requests

# optional; defaults to `os.environ['OPENAI_API_KEY']`
openai.api_key = 'sk-I1xNML4qSn5V4JAFoZUm3rZfqfJGyTviI82PFbbFyvXP9vew'

# all client options can be configured just like the `OpenAI` instantiation counterpart
openai.base_url = "https://api.chatanywhere.tech"
openai.default_headers = {"x-foo": "true"}

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_gpt_response(instruction, scene, choices):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
                            I will offer you the instruction, the description of scene and the viewpoint choice with heading angles. \
                            The heading_degree indicates the relevant degree of viewpoint, negative for left and positive for right. \
                            You can also give me the specific angle if you want to turn left or turn right, range from -45 to 45.\
                            Please return the output in the format as 'choice number#angle#reason'"
            },
            {
                "role": "user",
                "content": f"Instruction: {instruction} \
                            Scene: {scene} \
                            Choices: {choices}",
            },
        ],
    )
    return completion.choices[0].message.content

def get_gpt_response_his(progress, scene, choices):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
                            I will offer you the degree of the progress of instruction, the description of scene, the viewpoint choice with heading angles. \
                            The heading_degree indicates the relevant degree of viewpoint, negative for left and positive for right. \
                            You can also give me the specific angle if you want to turn left or turn right, range from -45 to 45.\
                            Please return the output in the format as 'choice number#angle#reason'"
            },
            {
                "role": "user",
                "content": f"Instruction: {instruction} \
                            Scene: {scene} \
                            Choices: {choices}",
            },
        ],
    )
    return completion.choices[0].message.content

def get_gpt_angle(instruction, scene):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
                            I will offer you the instruction and the description of scene. \
                            Now we don't have viewpoint in our view, please tell me the angle we should turn, \
                            negative for left and positive for right, range from -45 to 45.\
                            Please return the output in the format as 'angle#reason']"
            },
            {
                "role": "user",
                "content": f"Instruction: {instruction} \
                            Scene: {scene}",
            },
        ],
    )
    return completion.choices[0].message.content

def img_gpt(instruction, img_path):
    base64_image = encode_image(img_path)
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
                            I will offer you the instruction and the picture of scene. You can see several numbers in the picture and you need \
                            to decide which number should I go.\
                            Please return the output in the format as 'number#reason'"
            },
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"Instruction: {instruction}"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
            }
        ],
    )
    return completion.choices[0].message.content

def get_history_sum(instruction, trac):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a trajectory summary expert. Your work is to help me determine the progress \
                            of the implementation of the instruction \
                            according to the scene and navigation decision trajectory list I give you. \
                            Return the instruction we have done and the instruction we haven't done"
            },
            {
                "role": "user",
                "content": f"Instruction: {instruction}\
                            I have only move {len(trac)} steps. \
                            Tractory include scene and decision reason: {trac}",
            },
        ],
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    # instruction = "Go down the path on the left towards the large dining room. \
    #                 Enter and go down the red carpet on the right side of the room, stopping at the door leading into another room."
    
    # scene = "The image depicts an indoor scene with a staircase in the middle of the room. \
    #         There is a doorway on the left side of the room, leading to a hallway with a staircase on the right side of the room. \
    #         There is a chair placed at the end of the hallway, which can be used as a seat or a place to rest your feet. \
    #         There is also a clock located at the end of the hallway, which can be used as a clock or a timer. \
    #         There are several pieces of furniture scattered throughout the room, including a sofa, a chair, and a table. \
    #         There is also a lamp placed at the end of the hallway, which can be used as a lamp or a place to rest your feet."
    
    # choices = {1: {'heading_degree': -5.878878714866082 }, 2: {'heading_degree': -2.718334608055504}, 3: {'heading_degree': 12.613345344982044}, 4: {'heading_degree': 55.41975194432243}}
    # output = get_gpt_response(instruction, scene, choices)
    # output = get_gpt_angle(instruction, scene)
    # infos = output.split('#')
    # print(output)

    instruction = 'Go down the path on the left towards the large dining room. \
                    Enter and go down the red carpet on the right side of the room, stopping at the door leading into another room.'
    img_path = '/root/Matterport3DSimulator/output/images/irgb.jpg'
    print(img_gpt(instruction, img_path))