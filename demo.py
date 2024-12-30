import cv2
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import re
import os
import json
from tqdm import tqdm

modes = ["point1", "point2"]
MODE = modes[1]

output_dir = "/home/minghao/code/vlm/molmo/demo_output3"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

images_dir = "/home/minghao/code/vlm/molmo/data/see_images_origin/val"
# Load instructions from the JSON file
with open(os.path.join(images_dir, 'instruction.json'), 'r') as f:
    instructions = json.load(f)

# Get all image paths and corresponding instructions
images = []
for file_name in os.listdir(images_dir):
    if file_name.endswith('.jpg'):
        image_id = re.search(r'val_\d+_id_(\d+)\.jpg', file_name).group(1)
        if image_id in instructions:
            image_path = os.path.join(images_dir, file_name)
            instruction = instructions[image_id]
            images.append((image_path, instruction))

github_model_dir = '/home/minghao/warehouse/molmo/model/Molmo-7B-D-0924'
molmo_7b = '/data/minghao/molmo/hf/Molmo-7B-D-0924'
molmo_1b = '/data/minghao/molmo/hf/MolmoE-1B-0924'
model_dir = molmo_1b

# load the processor
processor = AutoProcessor.from_pretrained(
    model_dir, # 'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    model_dir, # 'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)


def molmo_model(image, text):
    # process the image and text
    # Open the image from the file path
    inputs = processor.process(
        images=[image],
        text=text
    )

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print the generated text
    return generated_text

def molmo_to_xy(generated_text, pattern = r'<point x="([\d.]+)" y="([\d.]+)" alt="[^"]+">[^<]+</point>'):
    # Use regular expression to find the coordinates in the generated_text
    pattern = r'<point x="([\d.]+)" y="([\d.]+)" alt="[^"]+">[^<]+</point>'
    match = re.search(pattern, generated_text)

    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        # print(f"Coordinates extracted: x={x}, y={y}")
        return x, y
    else:
        # print(f"No coordinates found in {generated_text}.")
        return None

def new_path(output_dir, image_path):
    # Extract the file name from the image path
    file_name = os.path.basename(image_path)

    # Remove the file extension to get the base name
    base_name = os.path.splitext(file_name)[0]
    
    new_image_path = os.path.join(output_dir, "{}_with_point.jpg".format(base_name))
    return new_image_path

def draw_point(image_path, x, y, text=None):
    # Load the image
    image_w_point = cv2.imread(image_path)

    # Convert ratio coordinates to absolute integer values
    height, width, _ = image_w_point.shape
    x_abs = int(x * width / 100)
    y_abs = int(y * height / 100)

    # Draw a circle at the specified coordinates
    if MODE == "point1":
        color = (0, 0, 255)
    elif MODE == "point2":
        color = (0, 255, 0)
    cv2.circle(image_w_point, (x_abs, y_abs), radius=10, color=color, thickness=-1)

    # Define the new output image path
    # new_image_path = os.path.join(output_dir, "{}_with_point.jpg".format(file_name))
    new_image_path = new_path(output_dir, image_path)

    if MODE == "point1":
        if text is not None:
            # Draw the text at the left up corner
            cv2.putText(image_w_point, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # Save the modified image
    cv2.imwrite(new_image_path, image_w_point)

    return new_image_path


for output_image_path, instruction in tqdm(images, desc="Processing images"):
    image = Image.open(output_image_path)
    # text = "Point out where the hand is holding the object."
    text = f"Point out where to grasp to {instruction} as a robot with a grasper."
    text = f"Point out next point to move to {instruction} as a robot with a grasper."
    generated_text = molmo_model(image, text)

    coor = molmo_to_xy(generated_text)
    if coor:
        x, y = coor
        new_image_path = draw_point(output_image_path, x, y, text)
        # print(f"Image with point saved at {new_image_path}")
    else:
        print(f"No coordinates found in {generated_text}.")
