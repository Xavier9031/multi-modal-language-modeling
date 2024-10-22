import pandas as pd
from tinyllava.eval.run_tiny_llava import load_pretrained_model, TextPreprocess, ImagePreprocess, Message, KeywordsStoppingCriteria
import torch
import os
from tqdm import tqdm
from PIL import Image

model_path = "/mnt/sda1/htchang/DL/HW3/checkpoints/custom-finetune-TinyLLaVA-Phi-2-SigLIP-3.1B-lora_A2"
prompt = "<image>\n請幫我截取出圖片中下方的語音字幕"
image_dir = "/mnt/sda1/htchang/DL/HW3/kaggle_competitions/test"
conv_mode = "phi"  # or llama, gemma, etc

files = os.listdir(image_dir)

# Load model and preprocessors once
model, tokenizer, image_processor, context_len = load_pretrained_model(model_path, force_download=True)
text_processor = TextPreprocess(tokenizer, conv_mode)
data_args = model.config
image_processor = ImagePreprocess(image_processor, data_args)
model.cuda()

results = []

for file in tqdm(files):
    image_file = os.path.join(image_dir, file)
    # print(f"Processing {image_file}")

    # Prepare input text
    qs = prompt
    qs = "<image>\n" + qs
    msg = Message()
    msg.add_message(qs)
    result = text_processor(msg.messages, mode='eval')
    input_ids = result['input_ids']
    input_ids = input_ids.unsqueeze(0).cuda()
    
    # Load and process image
    image = Image.open(image_file).convert("RGB")
    images_tensor = image_processor(image)
    images_tensor = images_tensor.unsqueeze(0).half().cuda()

    stop_str = text_processor.template.separator.apply()[1]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=100,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    results.append({"id": file, "text": outputs})

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(results)
csv_output_path = "/mnt/sda1/htchang/DL/HW3/inference/results_A2.csv"
df.to_csv(csv_output_path, index=False)
