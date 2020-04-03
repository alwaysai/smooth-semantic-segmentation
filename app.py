import cv2 as cv
import edgeiq
import time
import os
import json
import numpy as np

"""
Use semantic segmentation to determine a class for each pixel of an image.
This particular example app uses semantic segmentation to cut a person out
of a frame and either blur the background or replace the background with an
image.

This app also has an option of smoothing our the edges of the segmentation,
which can be done by toggling 'smooth' to true in the config.json file.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""

# Static keys for extracting data from config.json
CONFIG_FILE = "config.json"
SEGMENTER = "segmenter"
MODEL_ID = "model_id"
BACKGROUND_IMAGES = "background_images"
IMAGE = "image"
TARGETS = "target_labels"
BLUR = "blur"
USE_BACKGROUND_IMAGE = "use_background_image"
BLUR_LEVEL = "blur_level"

def load_json(filepath):
    # check that the file exsits and return the loaded json data
    if os.path.exists(filepath) == False:
        raise Exception('File at {} does not exist'.format(filepath))

    with open(filepath) as data:
        return json.load(data)


def overlay_image(foreground_image, background_image, foreground_mask):
    background_mask = cv.cvtColor(
            255 - cv.cvtColor(foreground_mask, cv.COLOR_BGR2GRAY),
            cv.COLOR_GRAY2BGR)

    masked_fg = (foreground_image * (1 / 255.0)) * (foreground_mask * (1 / 255.0))
    masked_bg = (background_image * (1 / 255.0)) * (background_mask * (1 / 255.0))

    return np.uint8(cv.addWeighted(masked_fg, 255.0, masked_bg, 255.0, 0.0))


def main():
    # load the configuration data from config.json
    config = load_json(CONFIG_FILE)
    labels_to_mask = config.get(TARGETS)
    model_id = config.get(MODEL_ID)
    background_image = config.get(BACKGROUND_IMAGES) + config.get(IMAGE)
    blur = config.get(BLUR)
    use_background_image = config.get(USE_BACKGROUND_IMAGE)
    blur_level = config.get(BLUR_LEVEL)

    semantic_segmentation = edgeiq.SemanticSegmentation(model_id)
    semantic_segmentation.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(semantic_segmentation.engine))
    print("Accelerator: {}\n".format(semantic_segmentation.accelerator))
    print("Model:\n{}\n".format(semantic_segmentation.model_id))
    print("Labels:\n{}\n".format(semantic_segmentation.labels))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:

            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:
                # read in the video stream
                frame = video_stream.read()

                results = semantic_segmentation.segment_image(frame)

                # Generate text to display on streamer
                text = ["Model: {}".format(semantic_segmentation.model_id)]
                text.append("Inference time: {:1.3f} s".format(results.duration))


                # build the color mask, making all colors the same except for background
                semantic_segmentation.colors = [ (0,0,0) for i in semantic_segmentation.colors]

                # iterate over all the desired items to identify, labeling those white
                for label in labels_to_mask:
                    index = semantic_segmentation.labels.index(label)
                    semantic_segmentation.colors[index] = (255,255,255)

                # build the color mask
                mask = semantic_segmentation.build_image_mask(results.class_map)

                # Enlarge the mask
                dilatation_size = 15
                # Options: cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE
                dilatation_type = cv.MORPH_CROSS
                element = cv.getStructuringElement(
                        dilatation_type,
                        (2*dilatation_size + 1, 2*dilatation_size+1),
                        (dilatation_size, dilatation_size))
                mask = cv.dilate(mask, element)

                # apply smoothing to the mask
                mask = cv.blur(mask, (blur_level, blur_level))

                # the background defaults to just the original frame
                background = frame

                if use_background_image:
                    # read in the image
                    img = cv.imread(background_image)

                    # get 2D the dimensions of the frame (need to reverse for compatibility with cv2)
                    shape = frame.shape[:2]

                    # resize the image
                    background = cv.resize(img, (shape[1], shape[0]), interpolation=cv.INTER_NEAREST)

                if blur:
                    # blur the background
                    background = cv.blur(background, (blur_level, blur_level))

                frame = overlay_image(frame, background, mask)

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
