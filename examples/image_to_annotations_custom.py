# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Image to annotations script for drawn character animation."""

import sys
import requests
import cv2
import json
import numpy as np
from skimage import measure
from scipy import ndimage
from pathlib import Path
import yaml
import logging
import os

os.environ["PYOPENGL_PLATFORM"] = "osmesa"


def fill_skeleton(skeleton_json_loc):
    """
    Loads skeleton data from a JSON file and returns a list of joints.
    Each joint is a dict with keys: loc, name, parent.
    """
    with open(skeleton_json_loc, "r") as f:
        data = json.load(f)
    skeleton = []
    for joint in data["skeleton"]:
        skeleton.append({
            "loc": joint["loc"],
            "name": joint["name"],
            "parent": joint["parent"]
        })
    return skeleton


def image_to_annotations(
    img_fn: str,
    out_dir: str,
    skeleton_json_loc: str
) -> None:
    """
    Given the RGB image located at img_fn, runs detection, segmentation,
    and pose estimation for drawn character within it.
    Crops the image and saves texture, mask, and character config files
    necessary for animation. Writes to out_dir.

    Params:
        img_fn: path to RGB image
        out_dir: directory where outputs will be saved
        skeleton_json_loc: path to the skeleton json file
    """
    outdir = Path(out_dir)
    outdir.mkdir(exist_ok=True)

    img = cv2.imread(img_fn)
    cv2.imwrite(str(outdir / "image.png"), img)

    if len(img.shape) != 3:
        msg = f"image must have 3 channels (rgb). Found {len(img.shape)}"
        logging.critical(msg)
        assert False, msg

    if np.max(img.shape) > 1000:
        scale = 1000 / np.max(img.shape)
        img = cv2.resize(
            img,
            (
                round(scale * img.shape[1]),
                round(scale * img.shape[0])
            )
        )

    img_b = cv2.imencode(".png", img)[1].tobytes()
    request_data = {"data": img_b}
    resp = requests.post(
        "http://localhost:8080/predictions/drawn_humanoid_detector",
        files=request_data,
        verify=False
    )
    if resp is None or resp.status_code >= 300:
        raise Exception(
            "Failed to get bounding box, please check if the "
            "'docker_torchserve' is running and healthy, resp: {resp}"
        )

    detection_results = json.loads(resp.content)

    if (isinstance(detection_results, dict) and
            "code" in detection_results.keys() and
            detection_results["code"] == 404):
        assert False, (
            "Error performing detection. Check that "
            "drawn_humanoid_detector.mar was properly downloaded. "
            f"Response: {detection_results}"
        )

    detection_results.sort(key=lambda x: x["score"], reverse=True)

    if len(detection_results) == 0:
        msg = "Could not detect any drawn humanoids in the image. Aborting"
        logging.critical(msg)
        assert False, msg

    msg = (
        f"Detected {len(detection_results)} humanoids in image. "
        f"Using detection with highest score {detection_results[0]['score']}."
    )
    logging.info(msg)

    bbox = np.array(detection_results[0]["bbox"])
    l, t, r, b = [round(x) for x in bbox]

    with open(str(outdir / "bounding_box.yaml"), "w") as f:
        yaml.dump({
            "left": l,
            "top": t,
            "right": r,
            "bottom": b
        }, f)

    cropped = img[t:b, l:r]
    mask = segment(cropped)

    data_file = {"data": cv2.imencode(".png", cropped)[1].tobytes()}
    resp = requests.post(
        "http://localhost:8080/predictions/drawn_humanoid_pose_estimator",
        files=data_file,
        verify=False
    )
    if resp is None or resp.status_code >= 300:
        raise Exception(
            "Failed to get skeletons, please check if the "
            "'docker_torchserve' is running and healthy, resp: {resp}"
        )

    pose_results = json.loads(resp.content)

    if (isinstance(pose_results, dict) and
            "code" in pose_results.keys() and
            pose_results["code"] == 404):
        assert False, (
            "Error performing pose estimation. Check that "
            "drawn_humanoid_pose_estimator.mar was properly downloaded. "
            f"Response: {pose_results}"
        )

    if len(pose_results) == 0:
        msg = (
            (
                (
                    "Could not detect any skeletons within the character "
                    "bounding box. Expected exactly 1. Aborting."
                )
            )
        )
        logging.critical(msg)
        assert False, msg

    if 1 < len(pose_results):
        msg = (
            f"Detected {len(pose_results)} skeletons with the character "
            "bounding box. Expected exactly 1. Aborting."
        )
        logging.critical(msg)
        assert False, msg

    skeleton = fill_skeleton(skeleton_json_loc)

    char_cfg = {
        "skeleton": skeleton,
        "height": cropped.shape[0],
        "width": cropped.shape[1]
    }

    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(str(outdir / "texture.png"), cropped)
    cv2.imwrite(str(outdir / "mask.png"), mask)

    with open(str(outdir / "char_cfg.yaml"), "w") as f:
        yaml.dump(char_cfg, f)

    joint_overlay = cropped.copy()
    for joint in skeleton:
        x, y = joint["loc"]
        name = joint["name"]
        cv2.circle(joint_overlay, (int(x), int(y)), 5, (0, 0, 0), 5)
        cv2.putText(
            joint_overlay, name, (int(x), int(y + 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2
        )
    cv2.imwrite(str(outdir / "joint_overlay.png"), joint_overlay)


def segment(img: np.ndarray):
    """Threshold and segment the character from the background."""
    img = np.min(img, axis=2)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        115,
        8
    )
    img = cv2.bitwise_not(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, iterations=2)

    mask = np.zeros([img.shape[0] + 2, img.shape[1] + 2], np.uint8)
    mask[1:-1, 1:-1] = img.copy()

    im_floodfill = np.full(img.shape, 255, np.uint8)
    h, w = img.shape[:2]
    for x in range(0, w - 1, 10):
        cv2.floodFill(im_floodfill, mask, (x, 0), 0)
        cv2.floodFill(im_floodfill, mask, (x, h - 1), 0)
    for y in range(0, h - 1, 10):
        cv2.floodFill(im_floodfill, mask, (0, y), 0)
        cv2.floodFill(im_floodfill, mask, (w - 1, y), 0)

    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    mask2 = cv2.bitwise_not(im_floodfill)
    mask = None
    biggest = 0

    contours = measure.find_contours(mask2, 0.0)
    for c in contours:
        x = np.zeros(mask2.T.shape, np.uint8)
        cv2.fillPoly(x, [np.int32(c)], 1)
        size = len(np.where(x == 1)[0])
        if size > biggest:
            mask = x
            biggest = size

    if mask is None:
        msg = "Found no contours within image"
        logging.critical(msg)
        assert False, msg

    mask = ndimage.binary_fill_holes(mask).astype(int)
    mask = 255 * mask.astype(np.uint8)

    return mask.T


if __name__ == "__main__":
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f"{log_dir}/log.txt", level=logging.DEBUG)

    img_fn = sys.argv[1]
    out_dir = sys.argv[2]
    skeleton_json_loc = sys.argv[3]
    image_to_annotations(img_fn, out_dir, skeleton_json_loc)
