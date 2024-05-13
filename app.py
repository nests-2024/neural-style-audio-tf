import gradio as gr

import model

from utils import read_audio_spectum, spectrum_to_audio


example_audios = [
    ["outs/examples/johntejada-1.wav", "outs/examples/beat-box-2.wav"],
    ["outs/examples/imperial.mp3", "outs/examples/usa.mp3"]
]


def do_transfer(content_path, style_path):
    content_spectrum, sr = read_audio_spectum(content_path)
    style_spectrum, _ = read_audio_spectum(style_path)

    gen_spectrum = model.run(
        content_spectrum,
        style_spectrum,
        num_filters=4096,
        alpha=1e-2,
        max_iterations=128
    )
    gen_wav = spectrum_to_audio(gen_spectrum)

    return sr, gen_wav

demo = gr.Interface(
    title="Audio Style Transfer",
    description="Combine style and content from two different audio files",

    fn=do_transfer,
    inputs=[
        gr.Audio(type="filepath", source="upload", label="Content"),
        gr.Audio(type="filepath", source="upload", label="Style")
    ],
    outputs=[
        gr.Audio(label="Output"),
    ],

    examples=example_audios,
    cache_examples=True,

    allow_flagging="never",
    analytics_enabled=None
)

demo.launch(show_api=False, server_name="0.0.0.0")
