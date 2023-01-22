import argparse

class DreamCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--arg1", type=str, default="default value")
        self.parser.add_argument("--arg2", type=int, default=0)
        self.args = self.parser.parse_args()
        print("Welcome to DeforumCLI")
        print("use 'dream' to infer with the engine")
        print("use 'help' to get a list of all commands")

    def dream(self):
        print("Running dream method with args:", self.args)
    def help(self):
        print("Possible arguements:")

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
