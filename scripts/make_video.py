import imageio.v2 as iio
import os
import glob
import argparse

def make_video(image_folder, video_name, fps, loop):
    images = sorted(glob.glob(os.path.join(image_folder, '*')))
    # Add in reverse if loop is True
    if loop:
        images += sorted(glob.glob(os.path.join(image_folder, '*')), reverse=True)[1:]
    
    with iio.get_writer(video_name, fps=fps, format='mp4', codec='libx264', mode='I') as writer:
        for img_file in images:
            img = iio.imread(img_file)
            writer.append_data(img)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a video from images.')
    parser.add_argument('image_folder', type=str, help='Image folder path.')
    parser.add_argument('video_name', type=str, help='Output video path.')
    # Use action='store_true' to make --loop a flag
    parser.add_argument('--loop', action='store_true', help='Enable loop/patrol cycle.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    make_video(args.image_folder, args.video_name, args.fps, args.loop)

if __name__ == "__main__":
    main()