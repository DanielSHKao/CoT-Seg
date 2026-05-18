import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)



from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import yaml
import pdb
import cv2
from PIL import Image as PILImage
import re
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import matplotlib.pyplot as plt



from utils.refer import REFER
from openai import AzureOpenAI

from playwright.sync_api import sync_playwright
from markdownify import markdownify as md
import requests
from ddgs import DDGS

from pathlib import Path

import base64
from mimetypes import guess_type

with open("templates/cot-seg_new.txt", 'r', encoding='utf-8') as f:
    FIRST_GENERAL_PROMPT = f.read()
with open("templates/cot-refine_new.txt", 'r', encoding='utf-8') as f:
    REF_GENERAL_PROMPT = f.read()
FIRST_TASK_SPEC_PROMPT = f"""Please consider the following prompt "<USR_Q>" and follow the instructions."""
REF_TASK_SPEC_PROMPT = f"""Now you are given two images. Consider the user query "<USR_Q>" and follow the instruction to justify the correctness of the segmentation. Output the meta-queries for refinement if and only if needed."""

MIN_CLUSTER_SIZE = 20
APPLY_CLUSTERING = True

import requests
from ddgs import DDGS

GOOGLE_API_KEY = ""
SEARCH_ENGINE_ID = ""

def clean_clusters_scipy(mask: np.ndarray, min_size: int = 20, connectivity: int = 2) -> np.ndarray:
    """
    Remove small outlier clusters using scipy only.
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be 2D")
    
    binary = mask.astype(bool)
    
    # Label connected components
    labeled, num_features = ndimage.label(binary, structure=ndimage.generate_binary_structure(2, connectivity))
    
    # Count size of each label
    sizes = np.bincount(labeled.ravel())
    
    # Create mask of components that are large enough
    mask_sizes = sizes > min_size
    mask_sizes[0] = 0  # background (label 0) is never kept as object
    
    # Keep only large components
    cleaned = mask_sizes[labeled]
    
    return cleaned.astype(mask.dtype)

class RAG_Agent:
    def __init__(self, engine="ddgs", output_dir="./rag_files/", k=1):  # engine = ddgs | google
        os.makedirs(output_dir, exist_ok=True)
        self.ddgs = DDGS()
        self.searched_urls = []
        self.engine = engine
        self.output_dir = output_dir
        self.k = k
        self.num_reattempt = 50

    def search_urls(self, query):
        return self.search_urls_ddgs(query) if self.engine == 'ddgs' else self.search_urls_google(query)

    def search_urls_google(self, query):
        page = 1
        start = (page - 1) * 10 + 1
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&start={start}&sort=date"
        data = requests.get(url).json()
        search_items = data.get("items")
        urls = []

        for i, search_item in enumerate(search_items, start=1):
            link = search_item.get("link")
            urls.append(link)
        self.searched_urls = urls
        return self.searched_urls

    def search_urls_ddgs(self, query):
        attempt = 0
        while True:
            print("Scrape Attempt...")
            print(query)
            results = self.ddgs.text(query, max_results=self.k)
            print("full results", results)
            self.searched_urls = [r['href'] for r in results]
            if len(self.searched_urls) > 0 and len(self.searched_urls[0]) > 0:
                break
            attempt += 1
            if attempt > self.num_reattempt:
                print("Cannot find urls")
                break
        return self.searched_urls

    def scrape_to_markdown(self, name):
        output_dir = self.output_dir
        urls = self.searched_urls
        files = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            for id, url in enumerate(urls):
                output_fn = f"{name}_{id}.md"
                ofile = output_dir + output_fn
                page = browser.new_page()

                page.goto(url)
                page.wait_for_load_state("networkidle")

                rendered_html = page.content()

                page.close()

                markdown_content = md(rendered_html)

                with open(ofile, "w", encoding="utf-8") as file:
                    file.write(markdown_content)

                files.append(ofile)

                print(f"Saved {ofile!r}")
                if id + 1 >= self.k:
                    break
            browser.close()
        return files

    def scrape_query(self, query, name='test'):
        self.search_urls(query)
        files = self.scrape_to_markdown(name)
        return files



def inference_segzero_logits(predictor, processor, reasoning_model, image_path, prompt):
    QUESTION_TEMPLATE = \
        "Please find \"{Question}\" with bboxs and points." \
        "Compare the difference between object(s) and find the most closely matched object(s)." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"

    image = PILImage.open(image_path)
    image = image.convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size

    messages = []
    message = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(
                    Question=prompt.lower().strip("."),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                )
            }
        ]
    }]
    messages.append(message)

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    # pdb.set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    # pdb.set_trace()
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # pdb.set_trace()
    bboxes, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
    # pdb.set_trace()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=np.float16)
        predictor.set_image(image)
        for bbox, point in zip(bboxes, points):
            masks, scores, _ = predictor.predict(
                point_coords=[point],
                point_labels=[1],
                box=bbox,
                return_logits=True
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask = masks[0]
            mask_all = mask_all + relu(mask)
    return mask_all


def inference_segzero(predictor, processor, reasoning_model, image_path, prompt):
    QUESTION_TEMPLATE = \
        "Please find \"{Question}\" with bboxs and points." \
        "Compare the difference between object(s) and find the most closely matched object(s)." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"

    image = PILImage.open(image_path)
    image = image.convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size

    messages = []
    message = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(
                    Question=prompt.lower().strip("."),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                )
            }
        ]
    }]
    messages.append(message)

    # Preparation for inference
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

    # pdb.set_trace()
    image_inputs, video_inputs = process_vision_info(messages)
    # pdb.set_trace()
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=2048, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # pdb.set_trace()
    bboxes, points, think = extract_bbox_points_think(output_text[0], x_factor, y_factor)
    # pdb.set_trace()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=bool)
        predictor.set_image(image)
        for bbox, point in zip(bboxes, points):
            masks, scores, _ = predictor.predict(
                point_coords=[point],
                point_labels=[1],
                box=bbox
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask = masks[0].astype(bool)
            mask_all = np.logical_or(mask_all, mask)
    return mask_all.copy()





def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="../Seg-Zero/pretrained_models/VisionReasoner-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="./pretrained_models/sam_cp/sam2.1_hq_hiera_large.pt")
    parser.add_argument("--prompt", type=str, default="Segment the G7 host country flag from 2025.")
    parser.add_argument("--image_path", type=str, default="./demo_results/demo_images/g7.png")
    parser.add_argument("--output_path", type=str, default="./demo_results/outputs")
    parser.add_argument("--reseg_rounds", type=int, default=2)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--use_rag", type=bool, default=False)
    
    return parser.parse_args()


def extract_bbox_points_think(output_text, x_factor, y_factor):
    pred_bboxes = []
    pred_points = []
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            pred_bboxes = [[
                int(item['bbox_2d'][0] * x_factor + 0.5),
                int(item['bbox_2d'][1] * y_factor + 0.5),
                int(item['bbox_2d'][2] * x_factor + 0.5),
                int(item['bbox_2d'][3] * y_factor + 0.5)
            ] for item in data]
            pred_points = [[
                int(item['point_2d'][0] * x_factor + 0.5),
                int(item['point_2d'][1] * y_factor + 0.5)
            ] for item in data]
        except:
            try:
                data = [json.loads(line) for line in json_match.group(1).split('\n')][0]
                pred_bboxes = [[
                    int(item['bbox_2d'][0] * x_factor + 0.5),
                    int(item['bbox_2d'][1] * y_factor + 0.5),
                    int(item['bbox_2d'][2] * x_factor + 0.5),
                    int(item['bbox_2d'][3] * y_factor + 0.5)
                ] for item in data]
                pred_points = [[
                    int(item['point_2d'][0] * x_factor + 0.5),
                    int(item['point_2d'][1] * y_factor + 0.5)
                ] for item in data]
            except:
                pass

    think_pattern = r'<think>([^<]+)</think>'
    think_match = re.search(think_pattern, output_text)
    think_text = ""
    if think_match:
        think_text = think_match.group(1)

    return pred_bboxes, pred_points, think_text


def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def rag_summary(client, file_path, image_urls, query):
    with open(file_path, 'r', encoding='utf-8') as f:
        scraped = f.read()

    llm_prompt = f"You will serve as an agent for language-based image segmentation model, you are to extract information from a web scrape result relevant to a user query which will be served as context to guide segmentation. Please consider the following webscrape result, there may be irrelevant results or information \"{scraped}\", the user query is {query}."

    image_controls = [{
        "type": "image_url",
        "image_url": {
            "url": image_url
        }
    } for image_url in image_urls]
    text_control = [{
        "type": "text",
        "text": llm_prompt
    }]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers question as simple as possible."},
            {"role": "user", "content": text_control + image_controls}
        ],
        temperature=0
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def prompt_openai(client, image_urls, user_query, method='cot-seg', summary="", context=""):
    if method == "cot-seg-bbox-points" or method == "cot-seg-text":
        llm_prompt = FIRST_TASK_SPEC_PROMPT.replace("<USR_Q>", user_query)

    if method == "cot-seg-text":
        llm_prompt = FIRST_GENERAL_PROMPT + llm_prompt
    elif method == "cot-compare":
        llm_prompt = "You will serve as an agent for language-based image segmentation model. You need to decide which segmentation results is the best. You are given three images, the original image, labeled segzero result and labeled hq result, and you are also given the user query '" + user_query + "' and the summary of the scene and task '" + summary + "'. A segmentation is better if it segments all the object(s) wanted by the user query and it does not include extra objects, please strike a balance between these two aspects, for example, one result segments most of the object while the other one segments almost everything in the image, in this case the first segmentation is better. Please also note that we need segmentation mask and not outline or edge, having an outline of an object does not satisfy the segmentation requirements. Please output an integer indicating which result is better, 0 - segzero is better, 1 - hq result is better in the strict format of <answer>integer</answer> and provide a series of chain of thought question and answer that led to that result."
    elif method == "cot-compare-old":
        llm_prompt = "You will serve as an agent for language-based image segmentation model. You need to decide which segmentation results is the best. You are given three images, the original image, labeled segzero result, and labeled previous result, and you are also given the user query '" + user_query + "' and the summary of the scene and task '" + summary + "'. A segmentation is better if it segments all the object(s) wanted by the user query and it does not include extra objects, please strike a balance between these two aspects, for example, one result segments most of the object while the other one segments almost everything in the image, in this case the first segmentation is better. Please also note that we need segmentation mask and not outline or edge, having an outline of an object does not satisfy the segmentation requirements. Please output an integer indicating which result is better, 0 - current segzero is better, 1 - previous result is better in the strict format of <answer>integer</answer> and provide a series of chain of thought question and answer that led to that result."
    elif method == "cot-reseg":
        llm_prompt = REF_GENERAL_PROMPT + REF_TASK_SPEC_PROMPT.replace("<USR_Q>", user_query)


    if len(context) > 0:
        llm_prompt += rf" Here is also some addition context from a web scrape, it may or may not contain relevant information, <context>{context}<\context>"

    
    image_controls = [{
        "type": "image_url",
        "image_url": {
            "url": image_url
        }
    } for image_url in image_urls]
    text_control = [{
        "type": "text",
        "text": llm_prompt
    }]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers question as simple as possible."},
            {"role": "user", "content": text_control + image_controls}
        ],
        temperature=0.0
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def parse_meta_query(x, method='cot-seg'):
    if method == 'cot-seg':
        meta_query = x.split('<prompt>')[-1]
        meta_query = meta_query.split('</prompt>')[0]
        labels = x.split("<label>")[-1]
        labels = labels.split("</label>")[0]
        summary = x.split("<summary>")[-1]
        summary = summary.split("</summary>")[0]
        meta_query.replace("\"", "")
        meta_query.replace("\'", "")
        return {"meta_query": meta_query, "labels": labels, "summary": summary}
    else:
        # correctness
        correctness = x.split('<correctness>')[-1]
        correctness = correctness.split("</correctness>")[0]
        if "true" in correctness.lower():
            return {"pos_meta": "", "neg_meta": "", "plabels": [], "nlabels": []}
        meta_query = x.split('Meta-queries')[-1]
        pos_meta = meta_query.split('<positive>')[-1]
        pos_meta = pos_meta.split('</positive>')[0]
        pos_meta = pos_meta.lower().strip()
        if 'none' in pos_meta:
            pos_meta = ""

        neg_meta = meta_query.split("<negative>")[-1]
        neg_meta = neg_meta.split("</negative>")[0].lower().strip()
        if 'none' in neg_meta:
            neg_meta = ""
        neg_meta = neg_meta.replace("please remove", "please segment")

        # parse labels
        pos_labels = x.split("<plabel>")[-1].split('</plabel>')[0].split('.')
        neg_labels = x.split("<nlabel>")[-1].split('</nlabel>')[0].split('.')
        pos_labels = [i.strip() for i in pos_labels]
        neg_labels = [i.strip() for i in neg_labels]

        print("Positive:", pos_meta, "Negative", neg_meta, "P-Labels:", pos_labels, "N-Labels:", neg_labels)
        return {"pos_meta": pos_meta, "neg_meta": neg_meta, 'plabels': pos_labels, 'nlabels': neg_labels}


def parse_bb_points(output_text):
    json_match = re.search(r'```json[\r\n]*(.|[\r\n]*?)*```', output_text, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(0).replace("```json", "").replace("```", ""))
        summary = data.pop(0)["summary"]
        pred_bboxes = [[
            int(item['bbox_2d'][0]),
            int(item['bbox_2d'][1]),
            int(item['bbox_2d'][2]),
            int(item['bbox_2d'][3])
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0]),
            int(item['point_2d'][1])
        ] for item in data]

    return pred_bboxes, pred_points, summary





def relu(x, thres=0.):
    return x * (x > thres)




def visualize(score, save_path=None):
    score = (score - score.min()) / (score.max() - score.min())
    if save_path is not None:
        score = (score * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(score, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, heatmap)


def parse_comparision(txt):
    ans = txt.split("<answer>")[-1]
    ans = ans.split("</answer>")[0]
    return ans


def apply_mask_to_image(image, mask):
    """
    Apply a binary mask to an image, setting non-selected areas to a blank background.

    Parameters:
    - image: numpy array of shape (H, W, 3), representing the image.
    - mask: numpy array of shape (H, W), representing the binary mask.

    Returns:
    - masked_image: numpy array of shape (H, W, 3), with non-selected areas set to [0, 0, 0].
    """
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Initialize the masked image with a blank background
    masked_image = np.ones_like(image) * 255

    # Apply the mask: copy the image where mask is True
    masked_image[mask] = image[mask]

    return masked_image


def add_label(path, image, label):
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (255, 0, 0)

    # Calculate text size
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    # Automatic padding: 20px margin + text height
    padding = 20 + text_size[1]
    total_height = image.shape[0] + padding

    # Create extended canvas
    result_image = np.ones((total_height, image.shape[1], 3), dtype=np.uint8) * 255
    result_image[0:image.shape[0], 0:image.shape[1]] = image

    # Center text horizontally, position vertically with nice spacing
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = image.shape[0] + (padding * 2 // 3)  # Position in the lower 2/3 of padding area
    cv2.putText(result_image, label, (text_x, text_y), font, font_scale, color, thickness)
    cv2.imwrite(path, result_image)


def save_masks(data_dir, save_mask, image_np, img_id, name, color='red'):
    new_save_path = data_dir
    # Save the two results and compare
    save_path = "{}/{}_cot_seg_mask_{}.png".format(
        new_save_path, img_id, name
    )
    cv2.imwrite(save_path, save_mask * 255)

    masked_image = apply_mask_to_image(image_np, save_mask)
    masked_path = "{}/{}_cot_seg_result_{}.jpg".format(
        new_save_path, img_id, name
    )
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(masked_path, masked_image)

    save_path = "{}/{}_cot_seg_img_{}.jpg".format(
        new_save_path, img_id, name
    )
    save_img = image_np.copy() * 0.4
    if color == 'red':
        save_img[save_mask] += (save_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.6)[save_mask]
    else:
        save_img[save_mask] += (save_mask[:, :, None].astype(np.uint8) * np.array([0, 0, 255]) * 0.6)[save_mask]
    save_img = save_img.astype(np.uint8)
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img)


def save_compare_mask(data_dir, save_mask, image_np, img_id, name):
    save_path = "{}/{}_cot_seg_img_{}.jpg".format(
        data_dir, img_id, name
    )
    save_img = image_np.copy() * 0.4
    save_img[save_mask] += (save_mask[:, :, None].astype(np.uint8) * np.array([0, 255, 0]) * 0.6)[save_mask]
    save_img = save_img.astype(np.uint8)
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, save_img)


def load_data(data_dir):
    file_ids = [img_id.split('.')[0] for img_id in os.listdir(os.path.join(data_dir, "Image"))]
    cam_ids = [img_id for img_id in file_ids if img_id.split('-')[1] == "CAM"]
    return cam_ids

class moe_inference_pipeline:
    def __init__(self, save_dir, reasoning_model_path, checkpoint_path, reseg_rounds, mask_threshold):
        '''
        Loads the GPT client, reasoning and segmentation agents.

        '''
        self.save_dir = save_dir
        self.reseg_rounds = reseg_rounds
        self.mask_threshold = mask_threshold
        
        with open('config/openai.yaml', 'r') as file:
            openai_docs = yaml.safe_load(file)
        self.client = AzureOpenAI(
                api_key=openai_docs["api_key"],
                api_version=openai_docs["api_version"],
                azure_endpoint=openai_docs["azure_endpoint"]
        )

        self.reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self.reasoning_model.eval()


        # default processer
        self.processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")

        # Sam HQ2
        checkpoint = checkpoint_path
        model_cfg = "configs/sam2.1/sam2.1_hq_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        print("Finished Init")

    def save_image_grid(self, img_id, max_cols=4, figsize=(20, 12), dpi=150):
        """Save a grid of all images for a given img_id."""
        output_dir = self.save_dir
        # Collect all image files with this img_id
        image_files = []
        for ext in ('.jpg', '.jpeg', '.png'):
            for filepath in Path(output_dir).glob(f"{img_id}_*{ext}"):
                # Skip the grid file itself if it already exists
                if "grid" in filepath.name:
                    continue
                # Create a readable label from filename
                name = filepath.stem.replace(f"{img_id}_cot_seg_", "")
                label = name.replace("_", " ").replace("result", "result:").replace("mask", "mask:").replace("img",
                                                                                                             "overlay:")
                image_files.append((str(filepath), label))

        if not image_files:
            print(f"No images found for {img_id} in {output_dir}")
            return

        n_images = len(image_files)
        n_cols = min(max_cols, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (img_path, label) in enumerate(image_files):
            ax = axes[idx]
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(label, fontsize=8)
            ax.axis('off')

        # Turn off unused subplots
        for idx in range(len(image_files), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        grid_path = os.path.join(output_dir, f"{img_id}_grid.png")
        plt.savefig(grid_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved image grid to {grid_path}")
    def inference(self, prompt, image_path, img_id, cot_text=None, scraped=None):
        img_id = img_id.split('.')[0]
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_url = local_image_to_data_url(image_path)
        save_dir = self.save_dir
        if scraped == None:
            context = ""
        else:
            context = rag_summary(self.client, scraped, [image_url], prompt)
            print("RAG Summary:", context)
        try:
            cot_text = prompt_openai(self.client, [image_url], prompt, method='cot-seg-text', context=context)
        except Exception as e:
            print("Error:", e)
            return


        cot_path_text = os.path.join(save_dir, f"{img_id}_text_0.txt")

        with open(cot_path_text, 'w', encoding='utf-8') as file:
            file.write(cot_text)
        rs = parse_meta_query(cot_text, method='cot-seg')
        prompt = rs['meta_query']
        summary = rs['summary']

        '''
            SEG-ZERO RESULTS
        '''

        seg_zero_result = inference_segzero(self.predictor, self.processor, self.reasoning_model, image_path, prompt)


        # saving intermediate results

        save_masks(save_dir, seg_zero_result, image_np, img_id, "segzero_0", 'red')
        image_url = local_image_to_data_url(image_path)
        temp_img = cv2.imread("{}/{}_cot_seg_result_{}.jpg".format(save_dir, img_id, 'segzero_0'))
        add_label("{}/{}_cot_seg_compare_mask_{}.jpg".format(
            save_dir, img_id, 'segzero_0'
        ), temp_img, 'segzero result')
        segzero_url = local_image_to_data_url("{}/{}_cot_seg_compare_mask_{}.jpg".format(
            save_dir, img_id, 'segzero_0'
        ))

        print(summary)
        image_url = local_image_to_data_url(image_path)

        best_result = seg_zero_result.copy()
        # save compare_mask
        save_compare_mask(save_dir, best_result, image_np, img_id, "best_0")
        # save previous url
        best_image = apply_mask_to_image(image_np, best_result)
        best_path = "{}/{}_cot_seg_best_mask_{}.jpg".format(
            save_dir, img_id, "best_0"
        )
        best_image = cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR)
        add_label(
            best_path, best_image, "previous result"
        )
        prev_url = local_image_to_data_url(best_path)

        pred_mask_list = [best_result]

        
        # refinement process
        masked_path = "{}/{}_cot_seg_result_{}.jpg".format(
            save_dir, img_id, "segzero_0" 
        )
        for i in range(self.reseg_rounds):
            pred_mask = pred_mask_list[-1]
            seg_image_url = local_image_to_data_url(masked_path)
            cot_path = os.path.join(save_dir, f"{img_id}_{i}.txt")
            image_url = local_image_to_data_url(image_path)
            try:
                cot_answer = prompt_openai(self.client, [image_url, seg_image_url], prompt,
                                            method='cot-reseg')
            except Exception as e:
                print(e)
                break
            with open(cot_path, 'w', encoding='utf-8') as file:
                file.write(cot_answer)

            meta_queries = parse_meta_query(cot_answer, method='cot-reseg')
            positive_prompt = meta_queries["pos_meta"]
            negative_prompt = meta_queries["neg_meta"]

            if len(positive_prompt) == 0 and len(negative_prompt) == 0:
                print("No need for further refinement.")
                break

            pos_strength = 2.
            neg_strength = 2.
            ### Seg Zero
            pred_mask_segzero = pred_mask.copy()
            positive_score = np.zeros(pred_mask_segzero.shape)
            negative_score = np.zeros(pred_mask_segzero.shape)
            ### Positive Prompt
            if len(positive_prompt) > 0:
                positive_score = inference_segzero_logits(self.predictor, self.processor, self.reasoning_model,
                                                            image_path, positive_prompt)
                positive_score = positive_score / (positive_score.max() + 1e-6)
                pred_mask_segzero = pred_mask_segzero + positive_score * pos_strength

            ### Negative Prompt
            if len(negative_prompt) > 0:
                negative_score = inference_segzero_logits(self.predictor, self.processor, self.reasoning_model,
                                                            masked_path, negative_prompt)
                negative_score = negative_score / (negative_score.max() + 1e-6)
                pred_mask_segzero = pred_mask_segzero - negative_score * neg_strength
                if APPLY_CLUSTERING:
                    pred_mask_segzero = clean_clusters_scipy((pred_mask_segzero > self.mask_threshold).astype(np.uint8), min_size=MIN_CLUSTER_SIZE)
            ### Saving results
            heat_vis_path = "{}/{}_cot_seg_score_segzero_{}.jpg".format(
                save_dir, img_id, i + 1
            )
            pos_heat_vis_path = "{}/{}_cot_seg_pos_score_segzero_{}.jpg".format(
                save_dir, img_id, i + 1
            )
            neg_heat_vis_path = "{}/{}_cot_seg_neg_score_segzero_{}.jpg".format(
                save_dir, img_id, i + 1
            )
            visualize(pred_mask_segzero, heat_vis_path)
            visualize(positive_score, pos_heat_vis_path)
            visualize(negative_score, neg_heat_vis_path)

            save_mask = pred_mask_segzero > self.mask_threshold
            save_path = "{}/{}_cot_seg_mask_segzero_{}.png".format(
                save_dir, img_id, i + 1
            )
            cv2.imwrite(save_path, save_mask * 255)

            masked_image = apply_mask_to_image(image_np, save_mask)
            masked_path_segzero = "{}/{}_cot_seg_result_segzero_{}.jpg".format(
                save_dir, img_id, i + 1
            )
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(masked_path_segzero, masked_image)

            save_path = "{}/{}_cot_seg_img_segzero_{}.jpg".format(
                save_dir, img_id, i + 1
            )
            save_img = image_np.copy() * 0.4
            save_img[save_mask] += (save_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.6)[
                save_mask]
            save_img = save_img.astype(np.uint8)
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)

            ### Comparison
            temp_img = cv2.imread("{}/{}_cot_seg_result_{}.jpg".format(save_dir, img_id, f'segzero_{i + 1}'))
            add_label("{}/{}_cot_seg_compare_mask_{}.jpg".format(
                save_dir, img_id, f'segzero_{i + 1}'
            ), temp_img, 'segzero result')

            segzero_url = local_image_to_data_url("{}/{}_cot_seg_compare_mask_{}.jpg".format(
                save_dir, img_id, f'segzero_{i + 1}'
            ))

            compare_path = os.path.join(save_dir, f"{img_id}_compare_{i + 1}.txt")
            image_url = local_image_to_data_url(image_path)
            try:
                comparision_result = prompt_openai(self.client, [image_url, prev_url, segzero_url], prompt,
                                                    method='cot-compare-old', summary=summary)
            except:
                print("ERROR! Not Enough FUNDS!")
                break
            with open(compare_path, 'w', encoding='utf-8') as file:
                file.write(comparision_result)

            comparision_result = parse_comparision(comparision_result)
            if comparision_result == '0':
                best_pred_mask = pred_mask_segzero
                masked_path = "{}/{}_cot_seg_result_segzero_{}.jpg".format(
                    save_dir, img_id, i + 1
                )
            else:
                break

            best_image = apply_mask_to_image(image_np, best_pred_mask > self.mask_threshold)
            best_path = "{}/{}_cot_seg_best_mask_{}.jpg".format(
                save_dir, img_id, f"best_{i + 1}"
            )
            best_image = cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR)
            add_label(
                best_path, best_image, "previous result"
            )
            prev_url = local_image_to_data_url(best_path)

            ### Take best result
            pred_mask_list.append((best_pred_mask > self.mask_threshold).astype(int))

            save_compare_mask(save_dir, best_pred_mask > self.mask_threshold, image_np, img_id, f"best_{i + 1}")
        print("------------ Finished ------------")
        self.save_image_grid(img_id)

def main(args):
    args = parse_args(args)
    pl = moe_inference_pipeline(args.output_path, args.reasoning_model_path, args.segmentation_model_path, args.reseg_rounds, args.mask_threshold)
    img_id = args.image_path.split("/")[-1]
    prompt = args.prompt
    img_path = args.image_path

    if args.use_rag:
        agent = RAG_Agent(output_dir=args.output_path)
        agent.search_urls(prompt)
        if len(agent.searched_urls) == 0:
            pl.inference(prompt, img_path, img_id, None, None)
        else:
            files = agent.scrape_to_markdown("contexts")
            print("Retrieved to: ",files)
            # take top result
            pl.inference(prompt, img_path, img_id, None, files[0])
    else:
        pl.inference(prompt, img_path, img_id, None, None)


if __name__ == "__main__":
    main(sys.argv[1:])
    
    





