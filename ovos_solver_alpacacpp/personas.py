from ovos_solver_llamacpp.personas import OVOSLLama
from ovos_utils import camel_case_split


class OVOSAlpaca(OVOSLLama):
    start_instruction = """Below is an start_instruction that describes a task. Write a response that appropriately completes the request.\n\n"""
    antiprompt = "## Instruction:\n\n"
    prompt = "### Response:\n\n"

    def _apply_text_hacks(self, ans):
        if ans.strip():
            # handle when llama continues with a made up user question
            if self.antiprompt:
                ans = ans.split(self.antiprompt)[0]

            # HACK: there seems to be a bug where output starts with a unrelated word???
            # sometimes followed by whitespace sometimes not
            wds = ans.split()
            # handle no whitespace case
            t = camel_case_split(wds[0]).split(" ")
            if len(t) == 2:
                wds[0] = t[-1]
            # handle whitespace case
            elif len(wds) > 1 and wds[1][0].isupper():
                wds[0] = ""
            ans = " ".join([w for w in wds if w])

            bad_ends = [self.prompt, self.antiprompt]
            for b in bad_ends:
                if ans.endswith(b):
                    ans = ans[:-1 * len(b)]

            # with alpaca somethings answers start with "#"
            while ans[0] in ["#"]:
                ans = ans[1:]

        return ans or "I don't known"
