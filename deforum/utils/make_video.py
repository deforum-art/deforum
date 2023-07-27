import imageio.v2 as iio
import cv2
import os
import glob
import argparse
from natsort import natsorted

def make_video(image_folder, video_name, fps, loop):
    images = sorted(glob.glob(os.path.join(image_folder, '*')))
    # Add in reverse if loop is True
    if loop:
        images += sorted(glob.glob(os.path.join(image_folder, '*')), reverse=True)[1:]
    
    with iio.get_writer(video_name, fps=fps, format='mp4', codec='libx264', mode='I') as writer:
        for img_file in images:
            img = iio.imread(img_file)
            writer.append_data(img)

def make_video_cv2(image_folder, output_video_path, fps=30, loop=False):
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
    parser.add_argument('video_name', type=str, help='Output video path.')
    parser.add_argument('--backend', type=str, default='imageio', choices=['imageio', 'cv2'], help='Backend to use for creating video.')
    parser.add_argument('--loop', action='store_true', help='Enable loop/patrol cycle.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second.')

    return parser.parse_args()

def main():
    args = parse_arguments()

    if args.backend == 'imageio':
        make_video(args.image_folder, args.video_name, args.fps, args.loop)
    elif args.backend == 'cv2':
        make_video_cv2(args.image_folder, args.video_name, args.fps, args.loop)

if __name__ == "__main__":
    main()
