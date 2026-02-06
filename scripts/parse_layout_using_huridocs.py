# To run the document layout analysis docker:
# docker run --rm --name pdf-document-layout-analysis --gpus '"device=0"' -p 5060:5060 --entrypoint ./start.sh huridocs/pdf-document-layout-analysis:v0.0.24
# To run without GPU:
# docker run --rm --name pdf-document-layout-analysis -p 5060:5060 --entrypoint ./start.sh huridocs/pdf-document-layout-analysis:v0.0.24

import logging
logging.basicConfig(level=logging.INFO)

import os
import json
import time
import argparse
from typing import List, Dict, Tuple
import requests

import numpy as np
import tqdm
import pdf2image
import matplotlib.pyplot as plt
import layoutparser as lp

def convert_pdf2img(filename: str, scale: float=1.0) -> List:
    images = pdf2image.convert_from_path(filename)
    images = [img.resize(size=(int(img.size[0]*scale), int(img.size[1]*scale))) for img in images]
    return images

def save_images_and_update_layout(
    images: List, layout: List[Dict], image_folder: str, 
    base_path: str, show_segments: bool=False
):
    for block_id, block in enumerate(tqdm.tqdm(layout)):
        page_id = block['page_number']
        block_type = block['type']
        directory, filename = f"images/{image_folder}/clippings/", f"block{block_id}_page{page_id}_{block_type}.png",
        if not os.path.exists(os.path.join(base_path, directory)):
            os.makedirs(os.path.join(base_path, directory))
        
        # Save segment image
        image = images[page_id-1]
        image_address = os.path.join(directory, filename)
        scale = image.size[0]/block['page_width']        
        segment_image = image.crop((
            block['left']*scale - 10, block['top']*scale - 10,
            (block['left']+block['width'])*scale + 10, (block['top']+block['height'])*scale + 10,
        ))
        segment_image.save(os.path.join(base_path, image_address))

        if show_segments:
            plt.imshow(segment_image)
            plt.axis('off')
            plt.show()

        # Update layout
        block['block'] = block_id 
        block['image_address'] = image_address
        if 'text' in block:
            block['text'] = {'text': block['text'], 'source': 'DLA'}

    return layout

def save_layout_images(
    images: List, layout: List[Dict], image_folder: str, base_path: str, show_segments: bool=False
):
    page_layouts = {}  # Dictionary to store layouts by page number

    for block in layout:
        page_num = block['page_number']
        if page_num not in page_layouts:
            page_layouts[page_num] = []
        page_layouts[page_num].append(block)
    
    for page_num, blocks in page_layouts.items():
        image = images[page_num - 1]  
        lp_blocks = []
        
        for block in blocks:
            scale = image.size[0] / block['page_width']
            
            lp_block = lp.TextBlock(
                block=lp.Rectangle(
                    x_1=block['left'] * scale,
                    y_1=block['top'] * scale,
                    x_2=(block['left'] + block['width']) * scale,
                    y_2=(block['top'] + block['height']) * scale
                ),
                type=block['type']
            )
            lp_blocks.append(lp_block)
        
        image = lp.draw_box(
            image,
            lp_blocks,
            box_width=3,
            show_element_type=True,
            show_element_id=True,
        )
        
        directory = os.path.join(base_path, f"images/{image_folder}/layouts/")
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        filename = f"page{page_num}.png"
        image.save(os.path.join(directory, filename))
        
        if show_segments:
            plt.imshow(image)
            plt.axis('off')
            plt.show()


def ocr_pdf(file_path: str, url: str='http://localhost:5060/ocr', output_path: str='temp.pdf') -> str:
    """
    Run OCR on the PDF file and return the file with OCR.

    Args:
        file_path (str): Full path to the PDF file.
        url (str): The URL to send the POST request to (default: http://localhost:5060/ocr).

    Returns:
        pdf_file_path (str): Full path to the PDF file with OCR.
    """
    with open(file_path, 'rb') as f:
        files = {
            'file': (
                file_path.split('/')[-1],  # filename
                f,                         # file object
                'application/pdf'          # content type
            )   
        }
        data = {
            "extraction_format": "markdown"
        }
        response = requests.post(url+"/ocr", files=files, data=data)

        if response.status_code != 200:
            raise Exception(f"Response Status Code: {response.status_code} - {response.text}")
        
    logging.info(f"File OCR ran successfully (status code: {response.status_code})")

    # Save the file to output_path
    with open(output_path, 'wb') as f:
        f.write(response.content)

    return os.path.abspath(output_path)


def pdf_layout_analysis(file_path: str, url: str='http://localhost:5060', ocr: bool=True) -> List[Dict]:
    """
    Get the PDF document layout from DLA server using a POST request.

    Args:
        file_path (str): Full path to the PDF file.
        url (str): The URL to send the POST request to (default: http://localhost:5060).

    Returns:
        requests.Response: The response object from the server.
    """
    with open(file_path, 'rb') as f:
        files = {
            'file': (
                file_path.split('/')[-1],  # filename
                f,                         # file object
                'application/pdf'          # content type
            )
        }
        data = {
            "extraction_format": "markdown"
        }
        response = requests.post(url, files=files, data=data)

    if response.status_code != 200:
        raise Exception(f"Response Status Code: {response.status_code} - {response.text}")
    
    logging.info(f"DLA ran successfully (status code: {response.status_code})")
    response_json = response.json()

    # If the file doesn not have OCR, run OCR
    if all(block['text'] == '' for block in response_json) and ocr:
        logging.info("File does not have OCR, running OCR...")
        new_file_path = ocr_pdf(file_path, url, "temp.pdf")
        logging.info("OCR ran successfully")
        response_json = pdf_layout_analysis(new_file_path, url, ocr=False)

    # Block types have spaces, replace them with underscores
    for block in response_json:
        block['type'] = block['type'].replace(' ', '_')

    return response_json

def get_args():
    parser = argparse.ArgumentParser(description='Prase a pdf document with layout processing.')
    parser.add_argument('filename', type=str, help='Name of the PDF file to process')
    parser.add_argument('output', type=str, help='Name of the output JSON file')
    parser.add_argument('base_path', type=str, help='Name of the base folder (to save images/)')
    args = parser.parse_args()

    if not args.filename.lower().endswith(('pdf', '.pdf')):
        raise ValueError("Filename is not of type PDF.")

    if not args.output.lower().endswith(('json', '.json')):
        raise ValueError("Output is not of type JSON.")

    return args

def main():
    args = get_args()
    start_time = time.time()

    image_name = os.path.splitext(os.path.basename(args.filename))[0]

    # Get the document layout from DLA server 
    logging.info("Getting document layout from DLA server...")
    layout = pdf_layout_analysis(args.filename)

    logging.info("Converting PDF to images...")
    pdf_images = convert_pdf2img(args.filename)
    logging.info(f"Number of pages: {len(pdf_images)}")

    logging.info("Saving layout images...")
    save_layout_images(pdf_images, layout, image_name, args.base_path)

    logging.info("Saving images and updating layout...")
    layout = save_images_and_update_layout(pdf_images, layout, image_name, args.base_path)


    with open(args.output, 'w') as json_file:
        json.dump(layout, json_file, indent=4)

    end_time = time.time()
    logging.info(f"Time taken: {end_time - start_time} seconds")
    logging.info(f"Time taken per page: {(end_time - start_time) / len(pdf_images)} seconds")
    logging.info(f"Time taken per layout item: {(end_time - start_time) / len(layout)} seconds")

    return

if __name__ == "__main__":
    main()
