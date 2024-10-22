import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import time
from tqdm import tqdm

def generate_synthetic_data(output_dir, text_file, image_dir, image_files):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read sentences from text file
    with open(text_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]

    # Iterate through each sentence to generate an image
    for i, sentence in enumerate(tqdm(sentences)):
        # Randomly select an image
        selected_image = Image.open(os.path.join(image_dir, random.choice(image_files)))
        
        width, height = selected_image.size

        if height > width:
            continue

        if height < 100 or height * width > 2200000:#6000000
            continue
        
        area = height * width
        if 1560800 < area <= 2200000:
            selected_image = selected_image.resize((1920, 1080))
            width, height = 1920, 1080
            font_size = (75, 100)
        elif 660000 < area <= 1560800:
            selected_image = selected_image.resize((1280, 720))
            width, height = 1280, 720
            font_size = (50, 70)
        elif 315000 < area <= 660000:
            selected_image = selected_image.resize((854, 480))
            width, height = 854, 480
            font_size = (35, 48)
        elif area <= 315000:
            selected_image = selected_image.resize((640, 360))
            width, height = 640, 360
            font_size = (27, 36)
        else:
            selected_image = selected_image.resize((640, 360))
            width, height = 640, 360
            font_size = (27, 36)

        # Convert to RGB if the image is grayscale
        selected_image = selected_image.convert("RGB")
        
        # Create a blank image with the same size as the selected image
        new_image = selected_image.copy()
        draw = ImageDraw.Draw(new_image)

        # Function to create text
        def create_text(text_height, sentence, text_real_width, text_real_height, text_color, selected_font):
            text_position = ((width - text_real_width) // 2, text_height)
            region_right = text_position[0] + text_real_width
            draw.text(text_position, sentence, font=selected_font, fill=text_color)
                
            region_left = text_position[0]
            region_top = text_position[1]
            region_bottom = text_position[1] + text_real_height

            # Find maximum brightness within the region
            max_brightness = 0
            for x in range(region_left, region_right + 1):
                for y in range(region_top, region_bottom + 1):
                    try:
                        pixel_brightness = sum(selected_image.getpixel((x, y))) // 3  # Calculate brightness (average of RGB)
                    except:
                        continue
                    max_brightness = max(max_brightness, pixel_brightness)

            # Check if maximum brightness exceeds threshold
            brightness_threshold = 150  # Adjust this threshold as needed
            if max_brightness > brightness_threshold:
                border_thickness = 1
                for dx in range(-border_thickness, border_thickness + 1):
                    for dy in range(-border_thickness, border_thickness + 1):
                        draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0, 0, 0))
                draw.text(text_position, sentence, font=selected_font, fill=text_color)
            else:
                border_thickness = 1
                if random.randint(0, 3) == 2:
                    thick = random.randint(1, 3)
                    for dx in range(-border_thickness, border_thickness + thick):
                        for dy in range(-border_thickness, border_thickness + thick):
                            draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0, 0, 0))
                draw.text(text_position, sentence, font=selected_font, fill=text_color)

        gt_string = ""
        random.seed()
        num = 0
        out_range = 1

        text_color = (random.randint(250, 255), random.randint(250, 255), random.randint(250, 255))
        font_dir = "fit_video_chinese_font"
        font_files = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
        while out_range == 1:
            sentence = sentence.replace("\n", "")
            size = random.randint(font_size[0], font_size[1])
            selected_font = ImageFont.truetype(os.path.join(font_dir, random.choice(font_files)), size)
            text_bbox = draw.textbbox((0, 0), sentence, font=selected_font)
            text_real_width = text_bbox[2] - text_bbox[0]
            text_real_height = text_bbox[3] - text_bbox[1]
            line1 = height - int(text_real_height * 3)
            line2 = height - (text_real_height * 2)
            if width - text_real_width >= 0:
                create_text(random.randint(line1, line2), sentence, text_real_width, text_real_height, text_color, selected_font)
                out_range = 0
            else:
                num += 1
                with open("/mnt/sda1/htchang/DL/HW3/train_data/mistake_filtered.txt", "a", encoding="utf-8") as fw:
                    fw.write("第{}張有一行跳過".format(i) + "\n")
            if num == 10:
                break
        if num == 10:
            continue
        gt_string = gt_string + sentence + " "

        output_path = os.path.join(output_dir, "image_{}.jpg".format(i + 1))
        with open("/mnt/sda1/htchang/DL/HW3/train_data/out_filtered.txt", "a", encoding="utf-8") as f:
            f.write(output_path + " " + gt_string + "\n")

        alpha_value = 255  # You can adjust this value
        Image.blend(selected_image, new_image, alpha=alpha_value / 255.0).save(output_path)

if __name__ == "__main__":
    output_directory = '/mnt/sda1/htchang/DL/HW3/train_data/image_filtered'
    text_file_path = "/mnt/sda1/htchang/DL/HW3/online_txt/meanings_processed.txt"
    image_directory = "/mnt/sda1/htchang/DL/HW3/imagenet/only_image"
    image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

    generate_synthetic_data(output_directory, text_file_path, image_directory, image_files)
