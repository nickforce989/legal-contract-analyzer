# SPDX-License-Identifier: Apache-2.0

from frontend_gradio import demo


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
