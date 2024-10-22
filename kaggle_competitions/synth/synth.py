import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageEnhance
import time
from tqdm import tqdm

def generate_synthetic_data(output_dir, num_images, text_file, image_dir,image_file):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read sentences from text file
    with open(text_file, 'r', encoding='utf-8') as file:
        sentences = [line.strip() for line in file.readlines()]

    # Get list of image files from the image directory
    # image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # Get list of font files from the font directory
    for i in tqdm(range(num_images)):
        # print(i )
        
        # Randomly select an image and font
        selected_image = Image.open(os.path.join(image_dir, random.choice(image_file)))
        
        width, height = selected_image.size

        if (height<100 or height*width>2200000):
            continue
        
        area= height*width
        if 1560800<area<=2200000:
            selected_image=selected_image.resize((1920,1080))
            width, height =1920,1080
            font_size=(75,100)
        elif 660000<area<=1560800:
            selected_image=selected_image.resize((1280,720))
            width, height =1280,720
            font_size=(50,70)
        elif 315000<area<=660000:
            selected_image=selected_image.resize((854,480))
            width, height =854,480
            font_size=(35,48)
        elif area<=315000:
            selected_image=selected_image.resize((640,360))
            width, height =640,360
            font_size=(27,36)
        else:
            selected_image=selected_image.resize((640,360))
            width, height =640,360
            font_size=(27,36)



        # 如果是灰度圖像，將其轉換為 RGB 格式
        selected_image = selected_image.convert("RGB")
        
        # Create a blank image with the same size as the selected image
        new_image = selected_image.copy()
        draw = ImageDraw.Draw(new_image)



        # Calculate text position horizontal
        def create_text(text_height,sentence,text_real_width,text_real_height,text_color,selected_font):

            text_position = ((width-text_real_width)//2, text_height )
            region_right =text_position[0] +text_real_width
            draw.text(text_position, sentence, font=selected_font, fill=text_color)
                
            region_left = text_position[0]
            region_top = text_position[1]
            region_bottom =  text_position[1] +text_real_height

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
                        draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0,0,0))
                draw.text(text_position, sentence, font=selected_font, fill=text_color)
            else:
                border_thickness = 1
                if random.randint(0,3)==2:
                    thick=random.randint(1,3)
                    for dx in range(-border_thickness, border_thickness + thick):
                        for dy in range(-border_thickness, border_thickness + thick):
                            draw.text((text_position[0] + dx, text_position[1] + dy), sentence, font=selected_font, fill=(0,0,0))
                draw.text(text_position, sentence, font=selected_font, fill=text_color)
# end of def()


        gt_string=""
        random.seed()
        num=0
        out_range=1

        text_color=(random.randint(250,255),random.randint(250,255),random.randint(250,255))
        font_dir="fit_video_chinese_font"
        font_files = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f))]
        while out_range==1:
            sentence=random.choice(sentences)
            sentence=sentence.replace("\n","")
            size=random.randint(font_size[0],font_size[1])
            selected_font = ImageFont.truetype(os.path.join(font_dir, random.choice(font_files)), size)
            text_bbox = draw.textbbox((0,0), sentence, font=selected_font)
            text_real_width = text_bbox[2] - text_bbox[0]
            text_real_height = text_bbox[3] - text_bbox[1]
            # print(text_real_height,text_real_width)
            #text_real_width, text_real_height = draw.textsize(sentence, font=selected_font)
            line1=height-int(text_real_height*3)
            line2=height-(text_real_height*2)
            if width-text_real_width>=0:
                
                create_text(random.randint(line1,line2),sentence,text_real_width,text_real_height,text_color,selected_font)
                out_range=0
            else:
                num=num+1
                # print("第{}張有一行跳過".format(i))
                with open ("/mnt/sda1/htchang/DL/HW3/train_data/mistake.txt","a",encoding="utf-8") as fw:
                    fw.write("第{}張有一行跳過".format(i)+"\n")
            if num ==10:
                # print(num)       
                break
        if num ==10:
            continue
        gt_string=gt_string+sentence+" "

       
            

        output_path = os.path.join("image_{}.jpg".format(i + 1))
        with open ("/mnt/sda1/htchang/DL/HW3/train_data/out.txt","a",encoding="utf-8") as f:
            f.write(output_path+" "+gt_string+"\n")
        

        #Image.blend(selected_image, new_image, alpha=0.5).save(output_path)
        alpha_value = 255  # You can adjust this value
        Image.blend(selected_image, new_image, alpha=alpha_value/255.0).save(
            os.path.join(output_dir, "image_{}.jpg".format(i + 1))
        )



if __name__ == "__main__":
    output_directory = '/mnt/sda1/htchang/DL/HW3/train_data/images'
    #output_directory = "/nfs/RS2416RP/Corpora/Workspace/spec/synth/test/"
    number_of_images = 5000000
    text_file_path = "/mnt/sda1/htchang/DL/HW3/online_txt/all_txt.txt"
    #image_directory = ""
    image_directory = "/mnt/sda1/htchang/DL/HW3/imagenet/only_image"
    image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
    
    #text_color = (255, 255, 255)  # RGB color

    generate_synthetic_data(output_directory, number_of_images, text_file_path, image_directory,image_files)
