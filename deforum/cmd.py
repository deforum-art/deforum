
import json
import os, sys

print(os.getcwd())

#sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])
sys.path.extend([os.path.join(os.getcwd(), "deforum", "exttools")])

import secrets
import time

from deforum.general_utils import substitute_placeholders
from deforum.main import Deforum


def keyframeExamples():
    return '''{
    "0": "https://deforum.github.io/a1/Gi1.png",
    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",
    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",
    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",
    "max_f-20": "https://deforum.github.io/a1/Gi1.png"
}'''

def DeforumAnimPrompts():
    return r"""{
    "0": "tiny cute swamp bunny, highly detailed, intricate, ultra hd, sharp photo, crepuscular rays, in focus, by tomasz alen kopera",
    "30": "anthropomorphic clean cat, surrounded by fractals, epic angle and pose, symmetrical, 3d, depth of field, ruan jia and fenghua zhong",
    "60": "a beautiful coconut --neg photo, realistic",
    "90": "a beautiful durian, trending on Artstation"
}
    """



def main():
    print("Deforum Generator Test")



    deforum = Deforum()

    setattr(deforum.loop_args, "init_images", "")


    animation_prompts = DeforumAnimPrompts()

    deforum.root.animation_prompts = json.loads(animation_prompts)

    deforum.animation_prompts = deforum.root.animation_prompts
    deforum.args.timestring = time.strftime('%Y%m%d%H%M%S')
    current_arg_list = [deforum.args, deforum.anim_args, deforum.video_args, deforum.parseq_args]
    full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
    deforum.root.raw_batch_name = deforum.args.batch_name
    deforum.args.batch_name = substitute_placeholders(deforum.args.batch_name, current_arg_list, full_base_folder_path)
    deforum.args.outdir = os.path.join(full_base_folder_path, str(deforum.args.batch_name))

    if deforum.args.seed == -1 or deforum.args.seed == "-1":
        setattr(deforum.args, "seed", secrets.randbelow(999999999999999999))
        setattr(deforum.root, "raw_seed", int(deforum.args.seed))
        setattr(deforum.args, "seed_internal", 0)
    else:
        deforum.args.seed = int(deforum.args.seed)

    success = deforum()


if __name__ == "__main__":
    main()
