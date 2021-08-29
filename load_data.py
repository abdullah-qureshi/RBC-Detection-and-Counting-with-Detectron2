from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "/train/_annotations.coco.json", "/train")
register_coco_instances("my_dataset_val", {}, "/valid/_annotations.coco.json", "/valid")
register_coco_instances("my_dataset_test", {}, "/test/_annotations.coco.json", "/test")

#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

import random
from detectron2.utils.visualizer import Visualizer

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])
