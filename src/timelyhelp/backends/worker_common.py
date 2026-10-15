"""AF_UNIX RPC isolates incompatible native Python/NumPy/CUDA environments."""

from __future__ import annotations
import argparse
import json
import socket
import traceback
from pathlib import Path


def foreign_physics_contact(contact):
    """Classify a Gazebo native Contact using its scoped Entity names.

    Collision messages are privileged evaluation labels. This function never
    supplies entity names or IDs to the observation or assistance model.
    Ground support and contacts between robot links are not task collisions.
    """
    names = (contact.collision1.name, contact.collision2.name)
    if not all(names):
        raise ValueError("Native physics contact omitted scoped collision names")
    models = tuple(name.split("::", 1)[0] for name in names)
    if "robot" not in models:
        raise ValueError("Robot contact sensor emitted a contact without the robot")
    return any(model not in ("robot", "ground") for model in models)


def serve(engine_class):
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    engine = engine_class(Path(args.cache))
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(args.socket)
    channel = connection.makefile("rw")
    channel.write(json.dumps({"ready": True}) + "\n")
    channel.flush()
    try:
        for line in channel:
            try:
                request = json.loads(line)
                if request["command"] == "close":
                    break
                result = getattr(engine, request["command"])(request.get("data", {}))
                response = {"result": result}
            except Exception:
                response = {"error": traceback.format_exc()}
            channel.write(json.dumps(response, allow_nan=False) + "\n")
            channel.flush()
    finally:
        engine.close()
        channel.close()
        connection.close()
