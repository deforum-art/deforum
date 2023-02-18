import hashlib


def return_model_version(model):
    with open(model, 'rb') as file:
        # Read the contents of the file
        file_contents = file.read()

        # Calculate the SHA-256 hash
        sha256_hash = hashlib.sha256(file_contents).hexdigest()
        if sha256_hash == 'd635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824':
            version = '2.0 512'
            config = '512-base-ema.yaml'
        elif sha256_hash == '2a208a7ded5d42dcb0c0ec908b23c631002091e06afe7e76d16cd11079f8d4e3':
            version = '2.0 Inpaint'
            config = '512-inpainting-ema.yaml'
        elif sha256_hash == 'bfcaf0755797b0c30eb00a3787e8b423eb1f5decd8de76c4d824ac2dd27e139f':
            version = '2.0 768'
            config = '768-v-ema.yaml'
        elif sha256_hash == 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556':
            version = '1.4'
            config = 'sd-v1-4.yaml'
        elif sha256_hash == 'c6bbc15e3224e6973459ba78de4998b80b50112b0ae5b5c67113d56b4e366b19':
            version = '1.5 Inpaint'
            config = 'sd-v1-5-inpainting.yaml'
        elif sha256_hash == 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516':
            version = '1.5 EMA Only'
            config = 'v1-5-pruned-emaonly.yaml'
        elif sha256_hash == '88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36':
            version = '2.1 512'
            config = 'v2-1_512-ema-pruned.yaml'
        elif sha256_hash == 'ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0':
            version = '2.1 768'
            config = 'v2-1_768-ema-pruned.yaml'
        else:
            version = 'unknown'
            config = None
        # Print the hash
        return config, version
