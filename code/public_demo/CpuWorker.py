"""One CPU model process, isolated from ZeroGPU's forked CUDA workers."""
from typing import Any

import atexit
import multiprocessing
import os


def serve(connection: Any) -> None:
    """Process CPU model commands until the parent requests shutdown.

    Args:
        connection (Any): Child end of the duplex pipe for commands and replies.
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
        """Keep the worker that owns the loaded AES model.

        Args:
            worker (Any): CPUWorker with an initialized AES model.
        """
        self.worker = worker

    def predict_essay(self, essay: str) -> Any:
        """Forward an essay to the CPU worker for AES prediction.

        Args:
            essay (str): Essay text to score.

        Returns:
            Any: AES scores and model input-length metadata.
        """
        return self.worker.exchange("aes", essay)


class EncoderProxy:
    """Forward embedding requests while retaining the accepted encoder profile."""
    def __init__(self, worker: Any, profile: Any) -> None:
        """Bind a worker-owned encoder to its embedding profile.

        Args:
            worker (Any): CPUWorker with an initialized encoder.
            profile (Any): Pinned profile used when loading the encoder.
        """
        self.worker, self.profile = worker, profile

    def encode(self, texts: Any) -> Any:
        """Request a batch of embeddings from the worker.

        Args:
            texts (Any): Ordered list of input texts.

        Returns:
            Any: (Embeddings, original token lengths), aligned with the input texts.
        """
        return self.worker.exchange("encode", texts)


class CPUWorker:
    """Own one spawned CPU model process and its synchronous command connection."""

    def __init__(self) -> None:
        """Spawn the CPU process, open its pipe and register cleanup at exit."""
        self.owner_pid = os.getpid()
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child,), daemon=True)
        self.process.start()
        child.close()
        atexit.register(self.close)

    def exchange(self, operation: str, payload: Any) -> Any:
        """Run one synchronous CPU command. Callers must serialize access.

        Args:
            operation (str): One of load_aes, load_encoder, aes or encode.
            payload (Any): Bundle path, encoder profile, essay or list of texts, respectively.

        Returns:
            Any: Load acknowledgment, AES prediction or encoder output for the operation.

        Raises:
            RuntimeError: The worker is unavailable or the command failed.
            TimeoutError: No reply arrived within 120 seconds. The worker is closed.
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
        """Load the accepted AES bundle in the child process.

        Args:
            path (Any): Path to the AES bundle directory.

        Returns:
            Any: AESProxy bound to the CPU worker.

        Raises:
            ValueError: The worker reports a device other than CPU.
        """
        if self.exchange("load_aes", str(path))["device"] != "cpu":
            raise ValueError("AES must remain on CPU")
        return AESProxy(self)

    def load_encoder(self, profile: Any) -> Any:
        """Load the pinned embedding encoder in the child process.

        Args:
            profile (Any): Verified embedding profile.

        Returns:
            Any: EncoderProxy bound to the CPU worker and profile.

        Raises:
            ValueError: The worker reports a device other than CPU.
        """
        if self.exchange("load_encoder", profile)["device"] != "cpu":
            raise ValueError("Retrieval encoder must remain on CPU")
        return EncoderProxy(self, profile)

    def close(self) -> None:
        """Close the pipe and stop the child process from its owning process."""
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
