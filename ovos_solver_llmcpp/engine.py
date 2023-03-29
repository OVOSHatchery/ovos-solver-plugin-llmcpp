import subprocess
import sys
from enum import Enum


# from https://github.com/timweri/alpaca.cpp-bot/blob/master/alpaca_cpp_interface/interface.py
# MIT licensed
class LLMcppInterface:
    class State(Enum):
        ACTIVE = 0
        TERMINATED = 1

    def __init__(self, alpaca_exec_path, model_path):
        # TODO - expose kwargs
        #   -s SEED, --seed SEED  RNG seed (default: -1)
        #   -t N, --threads N     number of threads to use during computation (default: 4)
        #   -n N, --n_predict N   number of tokens to predict (default: 128)
        #   --top_k N             top-k sampling (default: 40)
        #   --top_p N             top-p sampling (default: 0.9)
        #   --repeat_last_n N     last n tokens to consider for penalize (default: 64)
        #   --repeat_penalty N    penalize repeat sequence of tokens (default: 1.3)
        #   -c N, --ctx_size N    size of the prompt context (default: 2048)
        #   --temp N              temperature (default: 0.1)
        #   -b N, --batch_size N  batch size for prompt processing (default: 8)

        self.alpaca_exec_path = alpaca_exec_path
        self.model_path = model_path

        self.start()

    def restart(self):
        self.terminate()
        self.start()

    def start(self):
        self.cli_process = subprocess.Popen(
            [self.alpaca_exec_path, '-m', self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr
        )
        # Signal whether the alpaca.cpp process is active
        self.state = LLMcppInterface.State.ACTIVE

        # Signal whether alpaca.cpp is ready for another prompt
        self.ready_for_prompt = True
        self._initial_flush_readline()

    def ask(self, query, single_line=True):
        self.write(query)
        ans = self.read()
        if ans.startswith(query):
            ans = ans[len(query):]
        if single_line:
            ans = ans.split("\n")[0].strip()
        return ans

    # Flush the initial prints before first user prompt
    def _initial_flush_readline(self):
        # Shouldn't be reading if process is killed
        if self.state != LLMcppInterface.State.ACTIVE:
            return

        # Flush 1 empty line
        self.cli_process.stdout.readline()

        # Flush "> "
        self.cli_process.stdout.read(2)

    # Read the alpaca.cpp generated text
    # Blocks until alpaca.cpp finishes
    def read(self):
        # Shouldn't be reading if process is killed or alpaca.cpp is waiting for
        # user input
        if self.state != LLMcppInterface.State.ACTIVE or self.ready_for_prompt:
            return

        output = ''

        # Used to detect user input prompt
        prev_new_line = True

        while True:
            # Read output of alpaca.cpp char by char till we see "\n> "
            byte_array = bytearray()
            while True:
                try:
                    byte_array += self.cli_process.stdout.read(1)
                    new_char = byte_array.decode('utf-8')
                    break
                except UnicodeDecodeError:
                    pass

            # User input prompt detection
            if prev_new_line and new_char == ">":
                # Check if the next char is " "
                next_char = self.cli_process.stdout.read(1).decode('utf-8')
                if next_char == " ":
                    break

                output += new_char + next_char
            else:
                output += new_char

            if new_char == '\n':
                prev_new_line = True
            else:
                prev_new_line = False

        # Now we wait for user input
        self.ready_for_prompt = True

        return output

    # Enters the next
    def write(self, prompt):
        # Shouldn't be writing if process is killed or alpaca.cpp is not
        # ready for user input
        if self.state != LLMcppInterface.State.ACTIVE or not self.ready_for_prompt:
            return False

        prompt = prompt.strip()
        # print(f"Wrote \"{prompt}\" to chat")
        self.cli_process.stdin.write((prompt + "\n").encode('utf-8'))
        self.cli_process.stdin.flush()

        # Now we wait for the model to generate text
        self.ready_for_prompt = False

        return True

    def terminate(self):
        if self.state != LLMcppInterface.State.ACTIVE:
            return

        self.state = LLMcppInterface.State.TERMINATED

        self.cli_process.stdin.close()
        self.cli_process.terminate()
        self.cli_process.wait(timeout=2)

    def __del__(self):
        self.terminate()
