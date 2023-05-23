from deforum.exttools.depth import DepthModel


def main():

    print("Depth Test")

    model = DepthModel("models", "cuda", True, keep_in_vram=True,
                                 depth_algorithm="Zoe", Width=512, Height=512,
                                 midas_weight=0.8)

    print(model)
