"""One CPU model process, isolated from ZeroGPU's forked CUDA workers."""
from typing import Any

import atexit
import multiprocessing
import os


def serve(connection: Any) -> None:
    """Process CPU-model commands until the parent closes the worker connection.

    Args:
        connection (Any): Connection used by this operation.

    Returns:
        None: This operation returns no value.
    """
    import torch
    from ielts_judge.aes.Inference import load_bundle
    from ielts_judge.panda.Corpus import Encoder

    torch.set_num_threads(2)
    aes = encoder = None
    while True:
        operation, payload = connection.recv()
        if operation == "close":
            break
        try:
            if operation == "load_aes":
                aes = load_bundle(payload, device="cpu")
                result = {"device": str(aes.device)}
            elif operation == "load_encoder":
                encoder = Encoder(payload, device="cpu")
                result = {"device": str(encoder.model.device)}
            elif operation == "aes":
                result = aes.predict_essay(payload)
            elif operation == "encode":
                result = encoder.encode(payload)
            else:
                raise ValueError("Unknown CPU operation")
            connection.send((True, result))
        except Exception as error:
            connection.send((False, type(error).__name__))
        finally:
            # Do not keep the last user's text while waiting for another request.
            payload = result = None
    connection.close()


class AESProxy:
    """Forward AES predictions to the separately owned CPU model process."""
    def __init__(self, worker: Any) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            worker (Any): Worker used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.worker = worker

    def predict_essay(self, essay: str) -> Any:
        """Return essay-only AES estimates with the model's input-length metadata.

        Args:
            essay (str): English essay text supplied for evaluation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return self.worker.exchange("aes", essay)


class EncoderProxy:
    """Forward embedding requests while retaining the accepted encoder profile."""
    def __init__(self, worker: Any, profile: Any) -> None:
        """Initialize this component's settings and collaborators.

        Args:
            worker (Any): Worker used by this operation.
            profile (Any): Profile used by this operation.

        Returns:
            None: This operation returns no value.
        """
        self.worker, self.profile = worker, profile

    def encode(self, texts: Any) -> Any:
        """Encode.

        Args:
            texts (Any): Texts used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        return self.worker.exchange("encode", texts)


class CPUWorker:
    """Own one spawned CPU model process and its synchronous command connection."""

    def __init__(self) -> None:
        """Initialize this component's settings and collaborators.

        Returns:
            None: This operation returns no value.
        """
        self.owner_pid = os.getpid()
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child,), daemon=True)
        self.process.start()
        child.close()
        atexit.register(self.close)

    def exchange(self, operation: str, payload: Any) -> Any:
        """Send one synchronous CPU command and receive its matching result; the scoring service serializes callers.

        Args:
            operation (str): Operation used by this operation.
            payload (Any): Payload used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if not self.process.is_alive():
            raise RuntimeError("CPU model process is unavailable; restart the runtime")
        self.connection.send((operation, payload))
        if not self.connection.poll(120):
            self.close()
            raise TimeoutError("CPU model process deadline expired")
        success, result = self.connection.recv()
        if not success:
            raise RuntimeError("CPU model operation failed: " + result)
        return result

    def load_aes(self, path: Any) -> Any:
        """Load the accepted AES bundle in the child process and return its CPU proxy.

        Args:
            path (Any): Filesystem location to read, validate or write.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if self.exchange("load_aes", str(path))["device"] != "cpu":
            raise ValueError("AES must remain on CPU")
        return AESProxy(self)

    def load_encoder(self, profile: Any) -> Any:
        """Load the pinned embedding encoder in the child process and return its proxy.

        Args:
            profile (Any): Profile used by this operation.

        Returns:
            Any: The computed result for the supplied inputs.
        """
        if self.exchange("load_encoder", profile)["device"] != "cpu":
            raise ValueError("Retrieval encoder must remain on CPU")
        return EncoderProxy(self, profile)

    def close(self) -> None:
        """Release this component's owned resources and complete its shutdown protocol.

        Returns:
            None: This operation returns no value.
        """
        if os.getpid() != self.owner_pid:
            return
        if self.process.is_alive():
            try:
                self.connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
        self.connection.close()
