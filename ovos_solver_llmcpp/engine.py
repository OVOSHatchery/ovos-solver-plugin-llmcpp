import subprocess
from enum import Enum

from ovos_utils.log import LOG


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
        self.state = LLMcppInterface.State.TERMINATED
        self.alpaca_exec_path = alpaca_exec_path
        self.model_path = model_path
        self.backend = self.get_backend_from_binary()
        if self.backend == "llama.cpp":
            raise ValueError("llama.cpp is unsupported, use ovos-solver-plugin-llamacpp instead")
        if self.backend not in ["alpaca.cpp", "gpt4all.cpp"]:
            LOG.warning("unrecognized binary, may be unsupported or hang forever")
        LOG.info(f"LLM engine: {self.backend}")
        if self.backend not in ["bloomz.cpp"]:
            self._start()

    def get_help_text(self):
        p = subprocess.Popen(
            [self.alpaca_exec_path, '--help'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        helptext = b""
        while True:
            l = p.stderr.readline()
            helptext += l
            if not l:
                break
        return helptext.decode("utf-8")

    def get_backend_from_binary(self):
        help = self.get_help_text()

        gpt4all_txt = "model path (default: gpt4all-lora-quantized.bin)"
        llama_txt = "model path (default: models/llama-7B/ggml-model.bin)"
        alpaca_txt = "model path (default: ggml-alpaca-7b-q4.bin)"
        bloomz_txt = "model path (default: models/ggml-model-bloomz-7b1-f16-q4_0.bin)"

        if llama_txt in help:
            return "llama.cpp"
        if gpt4all_txt in help:
            return "gpt4all.cpp"
        if alpaca_txt in help:
            return "alpaca.cpp"
        if bloomz_txt in help:
            return "bloomz.cpp"
        return "unknown"

    def restart(self):
        self.terminate()
        self._start()

    def _start(self):
        self.cli_process = subprocess.Popen(
            [self.alpaca_exec_path, '-m', self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Signal whether the alpaca.cpp process is active
        self.state = LLMcppInterface.State.ACTIVE

        # Signal whether alpaca.cpp is ready for another prompt
        self.ready_for_prompt = True
        self._initial_flush_readline()

    def _ask_bloomz(self, query):
        # bloomz does not support interactive mode
        self.cli_process = subprocess.Popen(
            [self.alpaca_exec_path, '-m', self.model_path, "-p", query],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        ans = b""
        flag = False
        while True:
            l = self.cli_process.stdout.readline()
            if b"sampling parameters:" in l:
                flag = True
            elif flag:
                ans += l
            if not l:
                break
        ans = ans.decode("utf-8").split("</s> [end of text]")[0].strip()
        if ans.startswith(query):
            ans = ans[len(query):]
        return ans.strip() or "?"

    def ask(self, query, single_line=True):
        if self.backend == "bloomz.cpp":
            return self._ask_bloomz(query)

        self._write(query)
        ans = self._read()
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
    def _read(self):
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
    def _write(self, prompt):
        # Shouldn't be writing if process is killed or alpaca.cpp is not
        # ready for user input
        if self.state != LLMcppInterface.State.ACTIVE or not self.ready_for_prompt:
            return False

        prompt = prompt.strip()
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
