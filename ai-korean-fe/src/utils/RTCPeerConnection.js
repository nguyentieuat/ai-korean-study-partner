export default class WebRTCClient {
  constructor(backendUrl, onFeedback) {
    this.backendUrl = backendUrl;
    this.onFeedback = onFeedback;
    this.pc = null;
    this.dc = null;
    this.ready = false;
  }

  async start() {
    this.pc = new RTCPeerConnection({
      iceServers: [{ urls: ["stun:stun.l.google.com:19302"] }],
    });

    // Feedback channel
    const feedbackChannel = this.pc.createDataChannel("feedback");
    feedbackChannel.onopen = () => console.log("📡 Feedback ready");
    feedbackChannel.onmessage = (e) => {
      if (this.onFeedback) this.onFeedback(JSON.parse(e.data));
    };
    this.feedbackChannel = feedbackChannel;

    // Chunks channel
    this.dc = this.pc.createDataChannel("chunks");
    this.dc.onopen = () => {
      console.log("📡 Chunks channel ready");
      this.ready = true;
    };
    this.dc.onclose = () => {
      console.log("📡 Chunks channel closed");
      this.ready = false;
    };

    // Tạo offer
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    // Gửi offer lên server
    const res = await fetch(`${this.backendUrl}/api/webrtc/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: this.pc.localDescription.sdp,
        type: this.pc.localDescription.type,
      }),
    });
    const answer = await res.json();
    await this.pc.setRemoteDescription(answer);

    // ICE logging
    this.pc.oniceconnectionstatechange = () => {
      console.log("ICE state:", this.pc.iceConnectionState);
    };
  }

  send(data) {
    if (this.ready && this.dc.readyState === "open") {
      this.dc.send(data);
    }
  }

  sendSentence(sentence) {
    if (this.ready && this.dc.readyState === "open") {
      this.dc.send(JSON.stringify({ sentence }));
    }
  }

  async stop() {
    if (this.pc) await this.pc.close();
    this.pc = null;
    this.dc = null;
    this.ready = false;
  }
}
