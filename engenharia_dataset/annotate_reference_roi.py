import os
import cv2
import numpy as np


DATASET_DIR = "dataset_inicial"
OUTPUT_DIR = "dataset_aumentado/fontes/reference_masks"
WINDOW_NAME = "Reference ROI Annotator"


class PolygonAnnotator:
    def __init__(self, image, split, file_name):
        self.image = image
        self.split = split
        self.file_name = file_name
        self.points = []
        self.closed = False

    def reset(self):
        self.points = []
        self.closed = False

    def handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.closed:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.points) >= 3:
            self.closed = True

    def build_display(self):
        image_panel = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        instructions = [
            f"{self.split}: {self.file_name}",
            "Left click = add polygon point",
            "Right click = close polygon",
            "s = save polygon",
            "r = reset polygon",
            "n = next image (skip)",
            "q = quit",
            "Ctrl+C or close window = quit",
        ]

        panel_width = 420
        side_panel = np.zeros((self.image.shape[0], panel_width, 3), dtype=np.uint8)

        y = 20
        for line in instructions:
            cv2.putText(
                side_panel,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y += 20

        if len(self.points) >= 1:
            for point in self.points:
                cv2.circle(image_panel, point, 3, (0, 255, 255), -1)

        if len(self.points) >= 2:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(image_panel, [pts], self.closed, (255, 0, 255), 2)

        if self.closed and len(self.points) >= 3:
            overlay = image_panel.copy()
            cv2.fillPoly(overlay, [np.array(self.points, dtype=np.int32)], (255, 0, 255))
            image_panel = cv2.addWeighted(image_panel, 0.7, overlay, 0.3, 0)

        return cv2.hconcat([image_panel, side_panel])

    def build_mask(self):
        if len(self.points) < 3 or not self.closed:
            return None

        mask = np.zeros_like(self.image, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.points, dtype=np.int32)], 255)
        return mask


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def annotate_split(split):
    image_dir = os.path.join(DATASET_DIR, split, "image")
    output_dir = os.path.join(OUTPUT_DIR, split)

    if not os.path.isdir(image_dir):
        return False

    ensure_dir(output_dir)

    files = sorted(os.listdir(image_dir))

    for file_name in files:
        image_path = os.path.join(image_dir, file_name)
        mask_path = os.path.join(output_dir, file_name)

        if os.path.exists(mask_path):
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        annotator = PolygonAnnotator(image, split, file_name)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, annotator.handle_mouse)

        while True:
            display = annotator.build_display()
            cv2.imshow(WINDOW_NAME, display)

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return True

            key = cv2.waitKey(50) & 0xFF

            if key == 255:
                continue

            if key == ord("q"):
                return True

            if key == ord("n"):
                print(f"Skipped: {split}/{file_name}")
                break

            if key == ord("r"):
                annotator.reset()
                continue

            if key == ord("s"):
                mask = annotator.build_mask()
                if mask is None:
                    print(f"Polygon not ready: {split}/{file_name}")
                    continue

                cv2.imwrite(mask_path, mask)
                print(f"Saved polygon ROI: {mask_path}")
                break

    return False


def main():
    try:
        should_quit = False

        for split in ["train", "val", "test"]:
            should_quit = annotate_split(split)
            if should_quit:
                break
    except KeyboardInterrupt:
        print("\nAnnotation interrupted by user.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
