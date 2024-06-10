import cv2
import random
from scipy.spatial import ConvexHull
import numpy as np
import torch


def org_data(points: "torch.Tensor", labels: "np.ndarray"):
    k = np.max(labels) + 1
    rst = []
    for i in range(k):
        ids = np.argwhere(labels == i)
        ids = ids[:, 0]
        o = points[ids]
        rst.append(o)
    return rst


class Drawer:
    def __init__(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        window_x: int = 400,
        edge: int = 10,
    ) -> None:

        self.window_x = window_x
        self.win_w = int(window_x + 2 * edge)
        self.win_h = int(window_x / (max_x - min_x) * (max_y - min_y) + 2 * edge)

        self.f = window_x / (max_x - min_x)
        self.min_x = min_x
        self.min_y = min_y
        self.bias_x = -self.f * min_x + edge
        self.bias_y = -self.f * min_y + edge

        pass

    def draw(self, pos: "list[torch.Tensor]"):
        canvas = np.ones((self.win_h, self.win_w, 3), dtype=np.uint8)
        canvas = canvas * 255

        for i in range(len(pos)):
            c = pos[i]
            center = torch.mean(c, dim=0)
            c_x, c_y = int(self.f * center[0] + self.bias_x), int(
                self.f * center[1] + self.bias_y
            )
            color = (
                int(c_x / self.win_w * 255),
                int(c_y / self.win_h * 255),
                int((c_x + c_y) / (self.win_w + self.win_h) * 255),
            )
            try:
                hull = ConvexHull(c[:, :2])
                box = c[hull.vertices]
            except:
                box = c
            for o in c:
                x, y = int(self.f * o[0] + self.bias_x), int(
                    self.f * o[1] + self.bias_y
                )
                canvas = cv2.circle(canvas, (x, y), 2, color, 2)
            for i in range(len(box) - 1):
                x1, y1 = int(self.f * box[i, 0] + self.bias_x), int(
                    self.f * box[i, 1] + self.bias_y
                )
                x2, y2 = int(self.f * box[i + 1, 0] + self.bias_x), int(
                    self.f * box[i + 1, 1] + self.bias_y
                )
                canvas = cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

            x1, y1 = int(self.f * box[len(box) - 1, 0] + self.bias_x), int(
                self.f * box[len(box) - 1, 1] + self.bias_y
            )
            x2, y2 = int(self.f * box[0, 0] + self.bias_x), int(
                self.f * box[0, 1] + self.bias_y
            )
            canvas = cv2.line(canvas, (x1, y1), (x2, y2), color, 2)
        return canvas
