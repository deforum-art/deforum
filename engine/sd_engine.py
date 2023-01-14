#Thread safe global storage object
from engine import singleton
gs = singleton

import copy, yaml

#SHA256 check to determine if it's a known model
from engine.model_check import return_model_version

class SDGenerator():
    def __init__(self,
                 args
                 ):
        self.defaults = yaml.load(gs.sd_infer.defaults)
        self.args = args
        gs.loadedmodel = None

    def model_check(self):
        path = f"{gs.system.models}/{self.args.model}"
        version = return_model_version()
        #Check if desired model is loaded,
        #then check SHA of selected model
        #select config based on result
        #and load the model

    def model_load(self):

    def prep_txt2img(self):

    def prep_img2img(self):

    def inference(self):

    def pipeline(self, args=None):

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
        images = self.inference()
        return images




