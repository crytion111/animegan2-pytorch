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

    if ver == 'version 2 (🔺 稳健性,🔻 风格化)':
        out = face2paint(model2, img)
    else:
        out = face2paint(model1, img)
    return out


title = "人像动漫化v2"

description = "传入真人人像,将制作动漫化的人像"

article = "<p style='text-align: center'><a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>代码来源animegan2-pytorch</a></p></p>"

examples = [
    ['IU.png', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['1.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['2.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['3.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['4.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['5.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['6.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['7.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['8.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['9.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['10.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['11.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['12.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['13.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['14.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['15.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
    ['16.jpg', 'version 2 (🔺 稳健性,🔻 风格化)'],
]


web = gr.Interface(inference, [
    gr.inputs.Image(type="pil", label="这里是真人头像"),
    gr.inputs.Radio(['version 1 (🔺 风格化, 🔻 稳健性)', 'version 2 (🔺 稳健性,🔻 风格化)'],
                    type="value", default='version 2 (🔺 稳健性,🔻 风格化)', label='模型版本')
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
