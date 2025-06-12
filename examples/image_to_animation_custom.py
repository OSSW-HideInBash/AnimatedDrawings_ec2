# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from image_to_annotations_custom import image_to_annotations
from annotations_to_animation import annotations_to_animation
from pathlib import Path
import logging
import sys
from pkg_resources import resource_filename


def image_to_animation(
    img_fn: str,
    char_anno_dir: str,
    motion_cfg_fn: str,
    retarget_cfg_fn: str,
    skeleton_json_loc: str,
):
    """
    Given the image located at img_fn, create annotation
    files needed for animation.
    skeleton_json_loc: used for custom skeleton structure.
    Then create animation from those annotations and motion cfg
    and retarget cfg.
    """
    # Step 1: 이미지에서 annotation 파일 생성
    image_to_annotations(img_fn, char_anno_dir, skeleton_json_loc)

    # Step 2: annotation 파일로부터 애니메이션 생성
    annotations_to_animation(char_anno_dir, motion_cfg_fn, retarget_cfg_fn)


# 사용할 수 있는 모션 리스트
motion_list = [
    "config/motion/dab.yaml",
    "config/motion/jesse_dance.yaml",
    "config/motion/jumping.yaml",
    "config/motion/jumping_jacks.yaml",
    "config/motion/wave_hello.yaml",
    "config/motion/zombie.yaml",
]


if __name__ == "__main__":
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f"{log_dir}/log.txt", level=logging.DEBUG)

    # 필수 인자
    img_fn = sys.argv[1]  # 이미지 파일 경로
    char_anno_dir = sys.argv[2]  # 생성될 annotation 디렉토리
    skeleton_json_loc = sys.argv[4]  # 사용자 정의 스켈레톤 JSON 경로

    # 모션 선택 (옵션)
    if len(sys.argv) > 4:
        try:
            motion_idx = int(sys.argv[4])
            motion_cfg_fn = resource_filename(__name__, motion_list[motion_idx])
        except (ValueError, IndexError):
            motion_cfg_fn = resource_filename(
                __name__, "config/motion/dab.yaml"
            )
    else:
        motion_cfg_fn = resource_filename(__name__, "config/motion/dab.yaml")

    # Change the retargetting /config/bvh yaml file according to the dance
    # motion
    dance_motion_retarget = sys.argv[3] 
    if dance_motion_retarget == "1":
        retarget_cfg_fn = resource_filename(
            __name__, "config/retarget/mixamo_fff.yaml"
        )
    elif dance_motion_retarget == "3":
        retarget_cfg_fn = resource_filename(
            __name__, "config/retarget/cmu1_pfp.yaml"
        )
    else:
        retarget_cfg_fn = resource_filename(
            __name__, "config/retarget/fair1_ppf.yaml"
        )

    # 전체 파이프라인 실행
    image_to_animation(
        img_fn,
        char_anno_dir,
        motion_cfg_fn,
        retarget_cfg_fn,
        skeleton_json_loc,
    )