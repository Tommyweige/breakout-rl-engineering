"""Serve frozen Laya on the local NVIDIA GPU for the browser demo."""

import argparse

from breakout_rl.laya_server import LayaDecisionService, create_laya_server
from breakout_rl.laya_vision_agent import load_laya_agent, LAYA_MODEL_REVISION, LAYA_CODE_REVISION


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The local Decision Model demo requires CUDA")
    agent = load_laya_agent("cuda")
    if any(parameter.device.type != "cuda" for parameter in agent.model.parameters()):
        raise RuntimeError("Laya model parameters are not all on CUDA")
    runtime = {"device": str(agent.device), "gpu": torch.cuda.get_device_name(),
               "modelRevision": LAYA_MODEL_REVISION, "codeRevision": LAYA_CODE_REVISION}
    server = create_laya_server(LayaDecisionService(agent, runtime))
    print(f"Laya ready on {runtime['gpu']} at http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
