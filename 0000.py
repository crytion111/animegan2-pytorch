from PIL import Image
import torch
import gradio as gr


TESTdevice = "cpu"

index = 1

model2 = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device=TESTdevice,
    progress=False
)


model1 = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v1",
    device=TESTdevice)

face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint',
    size=512, device=TESTdevice, side_by_side=False
)


def inference(img, ver):
    global index

    img.save("input/" + str(index) + ".jpg")
    index += 1

    if ver == 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)':
        out = face2paint(model2, img)
    else:
        out = face2paint(model1, img)
    return out


title = "äººåå¨æ¼«åv2"

description = "ä¼ å¥çäººäººå,å°å¶ä½å¨æ¼«åçäººå"

article = "<p style='text-align: center'><a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>ä»£ç æ¥æºanimegan2-pytorch</a></p></p>"

examples = [
    ['IU.png', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['1.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['2.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['3.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['4.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['5.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['6.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['7.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['8.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['9.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['10.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['11.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['12.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['13.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['14.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['15.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
    ['16.jpg', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
]


web = gr.Interface(inference, [
    gr.inputs.Image(type="pil", label="è¿éæ¯çäººå¤´å"),
    gr.inputs.Radio(['version 1 (ðº é£æ ¼å, ð» ç¨³å¥æ§)', 'version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)'],
                    type="value", default='version 2 (ðº ç¨³å¥æ§,ð» é£æ ¼å)', label='æ¨¡åçæ¬')
],
    gr.outputs.Image(type="pil"),
    title=title,
    description=description,
    # article=article,
    examples=examples,
    allow_flagging=False,
    allow_screenshot=False)

web.launch(
    share=True
)
