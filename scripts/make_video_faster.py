import cv2
import os
import argparse
from natsort import natsorted

def stitch_images_to_video(image_folder, output_video_path, fps=30, loop=False):
    """
    Takes an image directory and outputs a video file using cv2

    image_folder : path to image directory
    output_video_path : output video file path (must be .mp4)
    fps : frames per second
    loop : whether to loop the images
    """
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(images)  # Natural sort order
    if loop:
        images += images[::-1][1:]

    # Read the first image to get the shape
    img = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = img.shape

    # Define the codec using VideoWriter_fourcc and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'X264'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        out.write(img)

    # Release the VideoWriter
    out.release()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a video from images.')
    parser.add_argument('image_folder', type=str, help='Image folder path.')
    parser.add_argument('output_video_path', type=str, help='Output video path.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second.')
    parser.add_argument('--loop', action='store_true', help='Enable loop/patrol cycle.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    stitch_images_to_video(args.image_folder, args.output_video_path, args.fps, args.loop)

if __name__ == '__main__':
    main()
