#Thread safe global storage object
from engine import singleton
gs = singleton

import copy, yaml

#SHA256 check to determine if it's a known model
from engine.model_check import return_model_version

class SDGenerator():
    def __init__(self,
                 parent
                 ):
        self.parent = parent
        self.defaults = yaml.load(gs.sd_infer.defaults)
        self.args = copy.deepcopy(self.defaults)
        gs.loadedmodel = None
    def reset_extras(self):
        self.pre_run_extras = None
        self.mid_run_extras = None
        self.post_run_extras = None

    def model_check(self):
        path = f"{gs.system.models}/{self.args.model}"
        version = return_model_version()
        #Check if desired model is loaded,
        #then check SHA of selected model
        #select config based on result
        #and load the model

    def model_load(self):

    def prep_txt2img(self):

        """
        Create starting noise, determine if we are
        running hires


        """

        self.prep_common()

    def prep_img2img(self):

        """
        Prepare latent image, force txt2img if None is found

        """
        #Debug print to test init_image type
        #We should be able to handle Tensor, PIL Image, QT Image, NumPy array, path, url, and byteslike objects
        #print(type(self.args.init_image))

        self.prep_common()
    def prep_common(self):
    def inference(self):

    def pipeline(self, args=None):
        if self.pre_run_extras is not None:
            for i in self.pre_run_extras:
                i(self.args)
        """
        Main entry point for Stable Diffusion inference
        can be called without any arguement.

        Please create a SimpleNameSpace object to interact with its params.

        Available parameters:
        job = "txt2img", ["txt2img", "img2img"]*
        width = 512
        height = 512
        cfg_scale = 7.5
        denoising_strength = 7.5
        init_image = None [PIL Image, path, or url (bytes array, numpy array) ]

        *If img2img is called without an init image, falls back to txt2img
        """

        if self.args != None:
            self.args = copy.deepcopy(args)
        else:
            self.args = copy.deepcopy(self.defaults)
        if self.args.model != gs.loadedmodel:
            self.model_check()
        getattr(self, f"prep_{self.args.job}")
        if self.mid_run_extras is not None:
            for i in self.mid_run_extras:
                i(self.args)

        images = self.inference()

        if self.post_run_extras is not None:
            for i in self.post_run_extras:
                i(self.args)

        if self.args.reset_extras == True:
            self.reset_extras()

        return images




