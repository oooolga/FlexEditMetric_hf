import json
import shutil
import os
import pandas as pd

coco_image_root = "/network/scratch/x/xuolga/Datasets/coco/2017/train2017"
vismin_scratch_root = "/network/scratch/r/rabiul.awal/vismin/"
csv_file = os.path.join(vismin_scratch_root, "train.csv")

def get_coco_image_path(image_url):
    return os.path.join(coco_image_root, image_url.split("/")[-1])

def get_vismin_editing_set():
    # load the annotation file
    annotation_df = pd.read_csv(csv_file)
    vismin_editing_set = []
    for index, row in annotation_df.iterrows():
        item = {
            "edited_image_id": row['image_id'],
            "edited_file_name": os.path.join(vismin_scratch_root, row['file_name']),
            "edited_caption": row['caption'],
            "edited_bounding_boxes": row['bounding_boxes'],
            "category": row['category']
        }
        
        if pd.notna(row['source_image_id']):
            # Find the source image row
            source_row = annotation_df[annotation_df['image_id'] == row['source_image_id']].iloc[0]
            
            # Add source image information to the item
            item.update({
                "source_image_id": source_row['image_id'],
                "source_file_name": get_coco_image_path(source_row['file_name']),
                "source_caption": source_row['caption'],
            })
            vismin_editing_set.append(item)

    return vismin_editing_set

if __name__ == "__main__":
    vismin_editing_set = get_vismin_editing_set()
    print(len(vismin_editing_set))