# 🧢 Cap Vision

## Overview

Hi I'm Nana and I love to dabble in the arts of Computer Vision, AI and Robotics.
This is a fun computer vision project where I trained a **YOLO11s** model to recognize the five caps I own.

To create the dataset, I took over **200 images** of my caps under different lighting conditions and from different angles. The images were then annotated using **Label Studio** before being used to train and evaluate the model.

The goal of this project was to brush up on my computer vision skills while building something personal and practical.

---

## Project Structure

### `split_image.py`

Randomly splits the dataset into training, validation, and test sets for YOLO training.

### `traintest.py`

Uses the configuration in `data.yaml` to train and evaluate the YOLO11s model.

### `yolo_deploy.py`

Deploys the trained YOLO model and runs inference through an application interface.

The application supports:

- 📷 Images
- 🎥 Videos
- 📹 Live webcam recordings

and can identify which cap is being worn.

---

## Dataset Creation

1. Collected 200+ images of my caps.
2. Captured images under different lighting conditions and viewing angles.
3. Annotated the dataset using Label Studio.
4. Split the dataset into training, validation, and test sets.
5. Trained a YOLO11s model.
6. Deployed the trained model for real-time inference.

---

## Tech Stack

- Python
- YOLO11s (Ultralytics)
- OpenCV
- Label Studio

---

## Results

The model successfully learns to distinguish between the five caps and can perform inference on images, videos, and webcam feeds.

---

## Inspiration

This project was inspired by the following tutorial:

👉 https://youtu.be/r0RspiLG260?si=1NFyTBY16xGGJRb0

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Nadjei74/Cap_Vision.git
cd Cap_Vision
```

Train the model:

```bash
python traintest.py
```

Run inference:

```bash
python yolo_deploy.py
```

---

## Future Improvements

- Collect more training images
- Add more cap classes
- Improve performance under difficult lighting conditions
- Package the application for easier deployment

---

⭐ Feel free to fork the project and use it as a starting point for your own custom YOLO object detection projects.
``
