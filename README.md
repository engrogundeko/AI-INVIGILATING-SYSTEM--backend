# **AI INVIGILATING SYSTEM**

A class for storing and manipulating inference results.

Attributes:
        orig_img (numpy.ndarray): Original image as a numpy array.
        orig_shape (tuple): Original image shape in (height, width) format.
        boxes (Boxes, optional): Object containing detection bounding boxes.
        masks (Masks, optional): Object containing detection masks.
        probs (Probs, optional): Object containing class probabilities for classification tasks.
        keypoints (Keypoints, optional): Object containing detected keypoints for each object.
        speed (dict): Dictionary of preprocess, inference, and postprocess speeds (ms/image).
        names (dict): Dictionary of class names.
        path (str): Path to the image file.

* [ ] Methods:
  update(boxes=None, masks=None, probs=None, obb=None): Updates object attributes with new detection results.
  cpu(): Returns a copy of the Results object with all tensors on CPU memory.
  numpy(): Returns a copy of the Results object with all tensors as numpy arrays.
  cuda(): Returns a copy of the Results object with all tensors on GPU memory.
  to(*args, **kwargs): Returns a copy of the Results object with tensors on a specified device and dtype.
  new(): Returns a new Results object with the same image, path, and names.
  plot(...): Plots detection results on an input image, returning an annotated image.
  show(): Show annotated results to screen.
  save(filename): Save annotated results to file.
  verbose(): Returns a log string for each task, detailing detections and classifications.
  save_txt(txt_file, save_conf=False): Saves detection results to a text file.
  save_crop(save_dir, file_name=Path("im.jpg")): Saves cropped detection images.
  tojson(normalize=False): Converts detection results to JSON format.
