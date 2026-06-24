import tensorflow as tf

from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)


processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


class BlipCaptionSummaryLayer(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        processor,
        model,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.processor = processor
        self.model = model

    def call(
        self,
        image_path,
        task
    ):

        return tf.py_function(
            self.process_image,
            [image_path, task],
            tf.string
        )

    def process_image(
        self,
        image_path,
        task
    ):

        image = Image.open(
            image_path.numpy().decode(
                "utf-8"
            )
        ).convert("RGB")

        if task.numpy().decode("utf-8") == "caption":

            prompt = "This is a picture of"

        else:

            prompt = (
                "This is a detailed photo showing"
            )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )

        output = self.model.generate(
            **inputs
        )

        return self.processor.decode(
            output[0],
            skip_special_tokens=True
        )


def generate_text(
        image_path,
        task
):

    layer = BlipCaptionSummaryLayer(
        processor,
        model
    )

    return layer(
        image_path,
        task
    )
