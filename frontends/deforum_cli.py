import argparse
import yaml

try:
    with open("../configs/cli-modules.yaml") as f:
        modules = yaml.safe_load(f)
        f.close()
except:
    with open("configs/cli-modules.yaml") as f:
        modules = yaml.safe_load(f)
        f.close()

for module in modules:
    try:
        __import__(module)
    except Exception as e:
        print(f"{module} could not be imported: {e}")

class DreamCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--width", type=int, default=512)
        self.parser.add_argument("--height", type=int, default=512)
        self.args = self.parser.parse_args()
        print("Welcome to DeforumCLI")
        print("use 'dream' to infer with the engine")
        print("use 'help' to get a list of all commands")

    def dream(self):
        print("Running dream method with args:", self.args)
    def help(self):
        print("Possible arguements:")
        for key, value in self.args.__dict__.items():
            print(key, value)


if __name__ == "__main__":
    cli = DreamCLI()
    while True:
        user_input = input("Enter command: ")
        if user_input == "dream":
            cli.dream()
        elif user_input == "help":
            cli.help()
        else:
            print("Invalid command.")
