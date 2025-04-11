import os
import asyncio
from typing import Any, Dict
from PIL import Image

from loguru import logger as eval_logger
from collections import defaultdict

# Load your existing references
from hadrian_vllm.model_caller import call_model, get_base64_image
from hadrian_vllm.evaluation import evaluate_answer, normalize_gdt, EASY_EVALUATION_MODE
from hadrian_vllm.result_processor import extract_answer
# Easy mode maybe later?
# import hadrian_vllm.evaluation as eval_mod
# eval_mod.EASY_EVALUATION_MODE = True


def gdt_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """
    The function that returns the image input for the model.
    This is analogous to 'mme_doc_to_visual' in the MME example.
    For you, doc["image_paths"] is a list of local PNGs.
    """
    # For a single-image scenario, we might do:
    #   return [open_image_as_PIL(doc["image_paths"][0])]
    # but many VLLM API calls want base64 or just a local path.
    # In the new version of lmms_eval, doc_to_visual is used if the model is
    # a huggingface-type that expects a PIL image or something.
    #
    # If your model is an OpenAI endpoint that doesn't accept images in a standard "visual" input,
    # you'll handle it in doc_to_text or in the model call.
    # For now let's just pass back the local paths:
    seen = {}
    out = []
    for image_path in doc['image_paths']:
        if image_path not in seen:
            # Expects list of RGB I think
            # try:
            #     image_base64 = get_base64_image(image_path)
            # except:
            #     image_base64 = get_base64_image("/data2/Users/clark/hadrian_vllm/" + image_path)
            # out += [{ "type": "image_url",
            #             "image_url": {
            #                 "url": f"data:image/png;base64,{image_base64}",
            #                 "detail": "high",
            #             }
            #         }]

            if isinstance(image_path, str):
                if "/data2/Users/clark/hadrian_vllm/" not in image_path:
                    image_path = "/data2/Users/clark/hadrian_vllm/" + image_path
                img = Image.open(image_path).convert("RGB")
            elif hasattr(image_path, "convert"):
                img = image_path.convert("RGB")
            else:
                raise ValueError("Unsupported type in image_paths: {}".format(type(image_path)))
            out.append(img)

            seen[image_path]=True
    return out



def gdt_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """
    The function that returns the textual portion for the model.
    If you have multi-turn data in doc["prompt"], you might store it or flatten it.
    """
    # If your prompt is a single big string, just do:
    if isinstance(doc["messages"],str):
        return doc["messages"]
    else:
        # TODO format chat better
        "\n".join([ f"{d['role']}: ```{d['content']}```" for d in doc["messages"]])

def gdt_doc_to_target(doc):
    """
    The function that returns the ground truth for the doc, e.g. doc['ground_truth'].
    This is used in 'doc_to_target' in the config. For a typical generation-based approach,
    the doc_to_target can just be a string or a short text you want the model to produce.

    But you might keep it minimal, just returning the GT text.
    """
    return doc["ground_truth"]

def gdt_process_results(doc, results):
    """
    Like 'mme_process_results' in the MME example.
    'results' is typically a list with one string: the model's generated output.
    We'll pass that to `evaluate_answer()` along with doc['ground_truth'].
    Then we return a dict with a custom key that references the aggregator.
    """

    prediction = results[0]  # single string from the model
    ground_truth = doc["ground_truth"]
    print('proecss results', results, ground_truth, evaluate_answer(prediction, ground_truth))

    # Evaluate
    try:
        score = evaluate_answer(extract_answer(prediction), ground_truth)
    except Exception as e:
        print(e)
        score = 0.0

    # We'll store it under a key you define, e.g. 'gdt_correctness'
    # so the aggregator can pick it up
    return {
        "gdt_correctness": {
            "element_id": doc["element_id"],
            "assemlby_id": doc["assemlby_id"],
            "score": score
        }
    }

def gdt_aggregate_results(results):
    """
    Aggregation function that sees all the "gdt_correctness" dicts from each doc.
    We'll just sum them and compute an average.
    """
    total_score = 0.0
    n = len(results)
    by_asem= defaultdict(lambda: {'n':0,'correct':0})
    for r in results:
        total_score += r["score"]  # each r is e.g. {"element_id": "...", "score": float}
        by_asem[r['assemlby_id']]['n'] +=1
        by_asem[r['assemlby_id']]['correct'] += r['score']
    if n == 0:
        return 0.0
    print({k:v['correct']/max(1,v['n']) for k,v in by_asem.items()})
    return (total_score / n) * 100.0
