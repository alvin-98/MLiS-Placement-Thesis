from guidance.chat import ChatTemplate
from guidance.chat import UnsupportedRoleException

class QwenChatTemplate(ChatTemplate):

    def get_role_start(self, role_name):
        # adjust tokens if you use a derivative model with different tags
        if role_name == "system":
            return "<|im_start|>system\n"
        elif role_name == "user":
            return "<|im_start|>user\n"
        elif role_name == "assistant":
            return "<|im_start|>assistant\n"
        else:
            raise UnsupportedRoleException(role_name, self)

    def get_role_end(self, role_name=None):
        # same token for every role in Qwen‑3
        return "<|im_end|>\n"