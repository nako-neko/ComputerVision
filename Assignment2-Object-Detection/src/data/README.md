# Dataset README

## Dataset Description

This dataset is a 2D image dataset for object detection in autonomous driving, which is created by HUAWEI Company.
The dataset contains 4K labeled training images, 500 labeled val images, and 1.5K test images with later released annotations.

There are only 4 categories in this dataset.
```python
CAR_CLASSES = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
```

## The Dataset Organization
```python
- annotations
    - instance_train.json # annotation for train data
    - instance_val.json   # annotation for val data
    - instance_test.json  # annotation for test data. It will be used to evaluate your model outputs and not be released
- train
- val
- test   # It will be released later.
```

## The Annotation Format
```shell
"annotations": {
    "image_name": <str>  # The image name for this annotation.
    "category_id": <int>  # The category id.
    "bbox": <list>  # Coordinate of boundingbox [x, y, w, h].
}

"categories": {
    "name": <str>  # Unique category name.
    "id": <int>   # Unique category id.
    "supercategory": <str>  # The supercategory for this category.
}
```
