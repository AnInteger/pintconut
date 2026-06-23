"""Ground-truth bead annotator — center + edge mode (validation only).

This tool does ZERO detection. It only records where YOU click:
  click 1 = bead center, click 2 = a point on the bead's circular edge.
It draws the resulting circle back so you can confirm it hugs the bead.
The ground truth is entirely your judgment.

Usage:
    python tests/validation/gt_annotate.py <image> <out.txt>

Controls:
    Left-click  : 1st click = center (red dot); move mouse = preview circle;
                  2nd click = lock radius (records cx cy r, draws green circle)
    u           : undo last bead
    c           : clear all
    s           : save and quit
    q / ESC     : quit (without saving)

Output format: one line per bead, "cx cy r" in pixel coordinates.
"""
import sys
import cv2

if len(sys.argv) != 3:
    sys.exit("usage: python tests/validation/gt_annotate.py <image> <out.txt>")

img_path, out_path = sys.argv[1], sys.argv[2]
img = cv2.imread(img_path)
if img is None:
    sys.exit(f"cannot read {img_path} (is it a real PNG/JPG? HEIC must be converted first)")

WIN = "gt_annotate: click=center, move=preview, click=edge | u=undo c=clear s=save q=quit"
beads: list[tuple[int, int, int]] = []   # (cx, cy, r)
center: list[tuple[int, int]] = []       # at most one pending center
mouse = [0, 0]                            # last mouse position (for preview)


def draw():
    disp = img.copy()
    for (cx, cy, r) in beads:
        cv2.circle(disp, (cx, cy), r, (0, 255, 0), 2)         # green = saved
        cv2.circle(disp, (cx, cy), 2, (0, 255, 0), -1)
    if center:
        cx, cy = center[0]
        cv2.circle(disp, (cx, cy), 3, (0, 0, 255), -1)        # red = pending center
        r = int(((cx - mouse[0]) ** 2 + (cy - mouse[1]) ** 2) ** 0.5)
        cv2.circle(disp, (cx, cy), r, (0, 255, 255), 1)       # yellow preview
        cv2.line(disp, (cx, cy), (mouse[0], mouse[1]), (0, 255, 255), 1)
        cv2.putText(disp, f"r={r}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(disp, f"{len(beads)} beads saved", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.imshow(WIN, disp)


def on_mouse(event, x, y, flags, param):
    mouse[0], mouse[1] = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        if not center:
            center.append((x, y))
        else:
            cx, cy = center[0]
            r = int(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5)
            if r > 0:
                beads.append((cx, cy, r))
            center.clear()
    draw()


cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, on_mouse)
draw()
print("annotating... click=center, move=preview, click=edge | u=undo c=clear s=save q=quit")

while True:
    k = cv2.waitKey(30) & 0xFF
    if k == 255:
        continue
    if k == ord("u") and beads:
        beads.pop()
        center.clear()
        draw()
    elif k == ord("c"):
        beads.clear()
        center.clear()
        draw()
    elif k in (ord("s"), ord("q"), 27):
        break

cv2.destroyAllWindows()
if beads:
    with open(out_path, "w") as f:
        for (cx, cy, r) in beads:
            f.write(f"{cx} {cy} {r}\n")
    print(f"saved {len(beads)} beads -> {out_path}")
else:
    print("no beads; nothing saved.")
