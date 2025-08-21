class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0]; // chỉ lấy channel 0
    if (input) {
      // gửi Float32Array về main thread
      this.port.postMessage(input);
    }
    return true; // tiếp tục process
  }
}

registerProcessor('recorder-processor', RecorderProcessor);
