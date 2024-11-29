from PIL import Image
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from FlexEditMetric.utils import DetectionResult, get_boxes, refine_masks, load_image
from FlexEditMetric.utils.utils import preprocess_caption, DetectionResult, BoundingBox
from transformers import GroundingDinoProcessor
from transformers import GroundingDinoForObjectDetection 
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# get pos tagger
from flair.data import Sentence
from flair.models import SequenceTagger

# load metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn.functional import mse_loss

class GroundedSAM:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load tagger
        self.tagger = SequenceTagger.load("flair/pos-english")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.psnr_metric_model = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric_model = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_metric_model = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(self.device)
        self.mse_loss = mse_loss
        
        self.processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        # Load the model and generate the grounding
        self.model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    
    def preprocess_caption(self, prompt: str) -> str:
        prompt_sent = Sentence(prompt)
        self.tagger.predict(prompt_sent)
        # tags = [token.text if 'NN' in token.tag or 'JJ' in token.tag else None for token in prompt_sent.tokens]
        objs = []
        # extract objects (NN) and adjectives (JJ) from the prompt
        for token_i in range(len(prompt_sent.tokens)-1, -1, -1):
            token = prompt_sent.tokens[token_i]
            if 'NN' in token.tag:
                if token_i-1 >= 0 and 'JJ' in prompt_sent.tokens[token_i-1].tag:
                    objs.append(f"{prompt_sent.tokens[token_i-1].text} {token.text}")
                else:
                    objs.append(token.text)
        processed_prompt = '. '.join(objs) if len(objs) > 0 else self.preprocess_caption_II(prompt)
        return processed_prompt

    
    def preprocess_caption_II(self, caption: str) -> str:
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."
    
    def dinogrounding(self, image: Image.Image, prompt: str, threshold: float = 0.3):
        # Preprocess the prompt
        processed_prompt = self.preprocess_caption(prompt)
        input = self.processor(images=image, text=processed_prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**input)
        # postprocess model outputs
        width, height = image.size
        postprocessed_output = self.processor.image_processor.post_process_object_detection(output,
                                                                            target_sizes=[(height, width)],
                                                                            threshold=0.3)
        return postprocessed_output[0]
    
    def modify_result_dict(self, results):
        ret = []
        results['boxes'] = results['boxes'].numpy().astype(int)
        for result_i in range(len(results['scores'])):
            ret.append(DetectionResult(score=results['scores'][result_i],
                                    label=results['labels'][result_i],
                                    box=BoundingBox(xmin=results['boxes'][result_i][0],
                                                    ymin=results['boxes'][result_i][1],
                                                    xmax=results['boxes'][result_i][2],
                                                    ymax=results['boxes'][result_i][3])))
            
        return ret
    
    def segment(self, image: Image.Image, detection_results: List[Dict[str, Any]], polygon_refinement=False):
        dino_segmentation = self.modify_result_dict(detection_results)
        seg_results = segment(image, dino_segmentation, polygon_refinement)
        return seg_results
    
    def compute_metrics(self, source: Image.Image, edit: Image.Image, prompt: str) -> Dict[str, float]:
        source_detection = self.dinogrounding(source, prompt)
        edit_detection = self.dinogrounding(edit, prompt)
        source_segmentation = self.segment(source, source_detection)
        edit_segmentation = self.segment(edit, edit_detection)
        segmentations = source_segmentation + edit_segmentation

        joint_segmentation = np.zeros((source.size[1], source.size[0]), dtype=np.uint8)

        for segmentation in segmentations:
            joint_segmentation += segmentation.mask
        joint_segmentation = joint_segmentation.clip(0, 1)
        unedited_a = pil_to_tensor(source)*torch.from_numpy(1-joint_segmentation)
        unedited_b = pil_to_tensor(edit)*torch.from_numpy(1-joint_segmentation)

        unedited_a_pil = to_pil_image(unedited_a)
        unedited_b_pil = to_pil_image(unedited_b)
        unedited_a_sam = get_sam_embeddings(unedited_a_pil)
        unedited_b_sam = get_sam_embeddings(unedited_b_pil)
        score_sam = self.mse_loss(unedited_a_sam, unedited_b_sam).cpu().item()*100

        unedited_a = unedited_a.unsqueeze(0).to(dtype=torch.float, device=self.device)/255.0
        unedited_b = unedited_b.unsqueeze(0).to(dtype=torch.float, device=self.device)/255.0

        score_psnr = self.psnr_metric_model(unedited_a, unedited_b).cpu().item()
        score_ssim = self.ssim_metric_model(unedited_a, unedited_b).cpu().item()
        score_lpips = self.lpips_metric_model(unedited_a * 2 - 1, unedited_b * 2 - 1).cpu().item()

        return {"PSNR": score_psnr, "SSIM": score_ssim, "LPIPS": score_lpips, 'SAM': score_sam}

def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def get_sam_embeddings(image: Image.Image,
                       segmenter_id: Optional[str] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)
    inputs = processor(images=image)
    pixel_values = torch.stack([torch.tensor(pixel_values, dtype=torch.float32, device=device) for pixel_values in inputs['pixel_values']])
    return segmentator.get_image_embeddings(pixel_values)


def segment(
    image: Image.Image,
    detection_results: List[Dict[str, Any]],
    polygon_refinement: bool = False,
    segmenter_id: Optional[str] = None
) -> List[DetectionResult]:
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    # boxes = [box for box in detection_results]
    if len(boxes[0]) == 0:
        return []

    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results

def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
) -> Tuple[np.ndarray, List[DetectionResult]]:
    if isinstance(image, str):
        image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections
     