import cv2
import numpy as np


def extract_dark_regions(image_path, output_path):
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 仅做自适应阈值提取，不做任何过滤，不做形态学操作
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    # 保存结果
    cv2.imwrite(output_path, mask)


if __name__ == "__main__":
    input_path = "./picture/his/4_crop.jpg"     # 改成你的输入图片
    output_path = "./picture/his/binary_4.png"   # 改成你的输出图片

    extract_dark_regions(input_path, output_path)