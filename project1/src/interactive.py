from pathlib import Path

from numpy.core.numeric import True_
from graphcut import GraphCut
from argparse import ArgumentParser
from PIL import Image, ImageDraw
import numpy as np
import PySimpleGUI as sg
import io

BBOX_SEL = "Please select a bounding box: "
MASK_SEL = "Please draw mask points: "

patches = []
bg = None
N = 0
bg_file = None
cur_op = BBOX_SEL
cur_bbox_stat = ""
mask = None
translate = None
radius = 5
bg_start_point, bg_end_point = None, None

def draw_frame(target, img, start=None, end=None, color=None, mask=None):
    if mask is not None:
        buffer = Image.fromarray(np.maximum(img, mask))
    else:
        buffer = img.copy()
    draw = ImageDraw.Draw(buffer)
    if color is None:
        color = (255, 255, 255)
    if start is not None and end is not None:
        draw.rectangle((start, end), outline=color, width=3)
    b = io.BytesIO()
    buffer.save(b, 'PNG')
    target.erase()
    target.DrawImage(data=b.getvalue(), location=(0, 0))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-input', help="Source image folder to open", default=None, type=str)
    args = parser.parse_args()

    if args.input is None:
        folder = sg.popup_get_folder("Source image folder to open", default_path="")
    else:
        folder = args.input

    if not folder:
        sg.popup_cancel("Cancelling")
        exit(0)

    def is_num(fname):
        try:
            int(fname.split('.')[0])
        except:
            return False
        return True

    input_dir = Path(folder)
    fnames = [file.name for file in input_dir.iterdir() if is_num(file.name)]
    fnames = sorted(fnames)
    N = len(fnames)
    if N == 0:
        sg.popup("No patch file in folder")
        exit(0)

    bg_file = str(input_dir / 'bg.jpg')

    bg = Image.open(bg_file).convert("RGB")
    bg_h = bg.height
    bg_w = bg.width
    patch_h = -1
    patch_w = -1


    for f in fnames:
        tmp = Image.open(str(input_dir / f)).convert("RGB")
        patch_h = max(patch_h, tmp.height)
        patch_w = max(patch_w, tmp.width)
        patches.append(tmp)
    
    w_patch_L_layout = [
        [sg.Text("List of available patches", size=(40, 3), font='25')],
        [sg.Listbox(values=fnames, change_submits=True, size=(60, 30), key='listbox', font='20')]
    ]
    w_patch_R_layout = [
        [sg.Text(cur_op, size=(40, 3), key='cur_op', font='25'), sg.Button('Clear Bbox', disabled=True, size=(30, 3), key='clear_bbox', font='20')],
        [sg.Text(cur_bbox_stat, size=(40, 3), key='cur_bbox_stat', font='25')],
        [sg.Graph((patch_w, patch_h), (0, patch_h), (patch_w, 0), key='img_disp', change_submits=True, drag_submits=True), 
         sg.Column([
             [sg.Text('Radius', size=(20, 3), font='20')],
             [sg.Slider(range=(1, 50), default_value=5, resolution=1, orientation='vertical', key='radius', change_submits=True)],
         ])]
    ]

    layout = [
        [sg.Column(w_patch_L_layout), sg.Column(w_patch_R_layout)],
        [sg.Column([[
            sg.Graph((bg_w, bg_h), (0, bg_h), (bg_w, 0), key='bg_disp', change_submits=True, drag_submits=False),
            sg.Column(
                [[sg.Button('Go!', size=(30, 3), key='proceed', font='20', disabled=True)],
                [sg.Button('Save Output', size=(30, 3), key='save', font='20')]]
            ),
            sg.Graph((bg_w, bg_h), (0, bg_h), (bg_w, 0), key='res_disp', change_submits=False, drag_submits=False)
        ]])]
    ]

    window = sg.Window('Patch Selector', layout, return_keyboard_events=True, location=(0, 0))
    window.Finalize()

    i = -1

    dragging = False
    start_point, end_point = None, None

    draw_frame(window['bg_disp'], bg)

    def clear_bbox():
        global start_point, end_point, cur_op, cur_bbox_stat, mask, translate, bg_start_point, bg_end_point
        start_point, end_point = None, None
        cur_op = BBOX_SEL
        cur_bbox_stat = ""
        window['clear_bbox'].update(disabled=True)
        draw_frame(window['img_disp'], patches[i])
        mask = np.zeros_like(patches[i])
        translate = None
        bg_start_point, bg_end_point = None, None

    def draw_bbox():
        global cur_bbox_stat, cur_op
        cur_bbox_stat = f'bbox: {start_point} - {end_point}'
        cur_op = MASK_SEL
        window['clear_bbox'].update(disabled=False)
        draw_frame(window['img_disp'], patches[i], start_point, end_point, mask=mask)

    def bbox_ready():
        global start_point, end_point
        return start_point is not None and end_point is not None

    blender = GraphCut(np.array(bg), (bg.height, bg.width))
    blender.insert_patch(0, 0, bg.width, bg.height)

    while True:
        draw_frame(window["res_disp"], Image.fromarray(blender.output))

        event, values = window.read()
        # print(event, values)
        if event == sg.WIN_CLOSED:
            break

        old_i = i
        if event == 'listbox':            # something from the listbox
            f = values["listbox"][0]            # selected filename
            i = fnames.index(f)                 # update running index
        if old_i != i:
            clear_bbox()

        if event == "save":
            path = sg.popup_get_file("Save Output As", save_as=True, default_extension=".png", file_types=(("Jpeg", "*.jpg"), ("PNG", "*.png")))
            if path is not None:
                Image.fromarray(blender.output).save(path)

        if i < 0:
            continue

        # handle bbox selection
        if event == "img_disp":
            x, y = values["img_disp"]

            if bbox_ready():
                im = Image.fromarray(mask)
                draw = ImageDraw.Draw(im)
                draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill=(255, 255, 255))
                mask = np.array(im)
                draw_frame(window['img_disp'], patches[i], start_point, end_point, mask=mask)
            elif not dragging:
                start_point = (x, y)
                dragging = True
        elif event.endswith('+UP'):
            dragging = False
            if end_point is None:
                end_point = values["img_disp"]
                x1, y1 = start_point
                x2, y2 = end_point
                start_point = min(x1, x2), min(y1, y2)
                end_point = max(x1, x2), max(y1, y2)
            if start_point[0] == end_point[0] and start_point[1] == end_point[1]:
                start_point, end_point = None, None     
        elif event == "bg_disp":
            if bbox_ready():
                translate = values["bg_disp"]
                x, y = translate
                w, h = end_point[0] - start_point[0], end_point[1] - start_point[1]
                bg_start_point = translate
                bg_end_point = (x + w, y + h)
        elif event == "clear_bbox":
            clear_bbox()
        elif event == "proceed":
            assert bbox_ready() and np.count_nonzero(mask) > 0 and translate is not None
            x1, y1 = start_point
            x2, y2 = end_point
            patch = np.array(patches[i])[y1:y2, x1:x2]
            pmask = np.any(mask[y1:y2, x1:x2] > 0, axis=-1)
            blender.set_patch(patch)
            blender.insert_patch(translate[0], translate[1], patch.shape[1], patch.shape[0], mask=pmask)
            bg = Image.fromarray(blender.output)
        elif event == "radius":
            radius = int(values["radius"])

        if bbox_ready():
            draw_bbox()

        draw_frame(window['bg_disp'], bg, bg_start_point, bg_end_point)
        
        window['proceed'].update(disabled=np.count_nonzero(mask) == 0 or not bbox_ready() or translate is None)
        window['cur_op'].update(cur_op)
        window['cur_bbox_stat'].update(cur_bbox_stat)

    window.close()