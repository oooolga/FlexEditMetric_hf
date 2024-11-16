import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .task_prompts import TaskPrompts
prompt_getter = TaskPrompts()

class VisMin(Dataset):
    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str):
        super(VisMin, self).__init__()
        self.model_type = model_type
        self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if self.model_type == "clip" else self.get_item_mllm

    def load_dataset(self, img_root_dir: str, annotation_file: str):
        annotation_file = os.path.expanduser(annotation_file)
        img_root_dir = os.path.expanduser(img_root_dir)
        print(f"Loading annotations from {annotation_file} and images from {img_root_dir}")
        data = pd.read_csv(annotation_file)
        import pdb; pdb.set_trace()
        data["image_path"] = data["image_path"].apply(
            lambda x: os.path.join(img_root_dir, "original_images", x.lstrip("/"))
        )
        data["edited_image_path"] = data["edited_image_path"].apply(
            lambda x: os.path.join(img_root_dir, "edited_images", x.lstrip("/"))
        )
        return [
            {
                "image_0": row["image_path"],
                "caption_0": row["caption"],
                "image_1": row["edited_image_path"],
                "caption_1": row["edited_caption"],
                "id": row["edit_id"],
            }
            for index, row in data.iterrows()
        ]

    def get_item_clip(self, index):
        sample = self.dataset[index]
        image0, image1 = sample["image_0"], sample["image_1"]
        caption0, caption1 = sample["caption_0"], sample["caption_1"]
        return {"image_0": image0, "image_1": image1, "caption_0": caption0, "caption_1": caption1}

    def get_item_mllm(self, index):
        sample = self.dataset[index]
        captions = [sample["caption_0"], sample["caption_1"]]
        images = [sample["image_0"], sample["image_1"]]
        text_task_prompt = prompt_getter.get_text_task_prompt(captions)

        item_all_combinations = []
        for i in range(2):
            image_task_prompt = prompt_getter.get_image_task_prompt(captions[i])
            item_all_combinations.append(
                {
                    "type": "text",
                    "text": text_task_prompt,
                    "image": images[i],
                    "label": ITG_LABEL_DICT["text"][i],
                }
            )
            item_all_combinations.append(
                {
                    "type": "image",
                    "text": image_task_prompt,
                    "image": images,
                    "label": ITG_LABEL_DICT["image"][i],
                }
            )
        return {"items": item_all_combinations, "id": sample["id"]}

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.dataset)

    def collate_clip(self, batch):
        return torch.utils.data.default_collate(batch)

    def collate_mllm(self, batch):
        all_combinations = []
        ids = []
        for item in batch:
            all_combinations.append(item["items"])  # Append each item's combinations as an inner list
            ids.append(item["id"])  # Collect all ids

        return {
            "items": all_combinations,  # This should now be a 2D list
            "id": ids,  # This is a list of ids corresponding to each original sample
        }

    def collate(self, batch):
        return self.collate_clip(batch) if self.model_type == "clip" else self.collate_mllm(batch)