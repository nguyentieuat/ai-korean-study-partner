import { useEffect, useRef, useState } from "react";

const SAMPLE_RATE = 16000; // Hz
const CHUNK_MS = 2000;      // mỗi chunk ~2000ms
const BUFFER_MS = 200;      // overlap  ~200ms
const CHUNK_SIZE = Math.floor((CHUNK_MS / 1000) * SAMPLE_RATE);
const BUFFER_SIZE = Math.floor((BUFFER_MS / 1000) * SAMPLE_RATE);

export default function useRecorderRTC(serverUrl) {
    const [recording, setRecording] = useState(false);
    const [feedback, setFeedback] = useState([]);
    const [audioUrl, setAudioUrl] = useState(null);

    const pcRef = useRef(null);
    const chunkChannelRef = useRef(null);
    const feedbackChannelRef = useRef(null);

    const audioChunksRef = useRef([]);
    const bufferRef = useRef([]);
    const audioCtxRef = useRef(null);
    const workletNodeRef = useRef(null);
    const localStreamRef = useRef(null);
    const sentenceRef = useRef(null);

    // --- Init PeerConnection + DataChannels ---
    useEffect(() => {
        let canceled = false;
        const initPC = async () => {
            try {
                const pc = new RTCPeerConnection();
                pcRef.current = pc;
                console.log("🔹 PeerConnection created");

                // Chunk channel
                const chunkChannel = pc.createDataChannel("chunks");
                chunkChannelRef.current = chunkChannel;
                chunkChannel.onopen = () => {
                    console.log("📡 Chunk channel opened");
                    if (sentenceRef.current) {
                        chunkChannel.send(JSON.stringify({ sentence: sentenceRef.current }));
                        console.log("📤 Sentence sent:", sentenceRef.current);
                    }
                };

                // Feedback channel
                pc.ondatachannel = (event) => {
                    if (event.channel.label === "feedback") {
                        const ch = event.channel;
                        feedbackChannelRef.current = ch;
                        ch.onmessage = (evt) => {
                            const fb = JSON.parse(evt.data);
                            setFeedback((prev) => [...prev, fb]);
                        };
                    }
                };

                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                if (canceled) {
                    stream.getTracks().forEach((t) => t.stop());
                    pc.close();
                    return;
                }
                localStreamRef.current = stream;
                stream.getTracks().forEach((track) => pc.signalingState !== "closed" && pc.addTrack(track, stream));

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                const res = await fetch(`${serverUrl}/api/webrtc/offer`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(pc.localDescription),
                });
                const answer = await res.json();
                if (pc.signalingState !== "closed") {
                    await pc.setRemoteDescription(answer);
                }
            } catch (err) {
                console.error("❌ initPC error:", err);
            }
        };

        initPC();
        return () => {
            canceled = true;
            pcRef.current?.close();
            pcRef.current = null;
        };
    }, [serverUrl]);

    // --- Helpers ---
    const sendChunkRealtime = (chunk, startSample) => {
        const bufData = new Float32Array(chunk);
        const header = new ArrayBuffer(12);
        const view = new DataView(header);
        view.setUint32(0, startSample, true);
        view.setUint32(4, startSample + chunk.length, true);
        view.setUint32(8, bufData.byteLength, true);

        const payload = new Uint8Array(12 + bufData.byteLength);
        payload.set(new Uint8Array(header), 0);
        payload.set(new Uint8Array(bufData.buffer), 12);

        if (chunkChannelRef.current?.readyState === "open") {
            chunkChannelRef.current.send(payload.buffer);
        }
    };

    // Float32 -> PCM16 ArrayBuffer
    const float32ToArrayBuffer = (f32) => {
        const buf = new ArrayBuffer(f32.length * 2);    // PCM16
        const view = new DataView(buf);
        for (let i = 0; i < f32.length; i++) {
            const s = Math.max(-1, Math.min(1, f32[i]));
            view.setInt16(i * 2, s * 0x7fff, true);
        }
        return buf;
    };

    const encodeWAV16kMono = (samples) => {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        const writeString = (view, offset, str) => {
            for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
        };

        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, SAMPLE_RATE, true);
        view.setUint32(28, SAMPLE_RATE * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, "data");
        view.setUint32(40, samples.length * 2, true);

        let offset = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s * 0x7fff, true);
            offset += 2;
        }

        return new Blob([view], { type: "audio/wav" });
    };

    // --- Start recording ---
    const startRecording = async (sentence) => {
        if (recording) return;
        sentenceRef.current = sentence;
        if (chunkChannelRef.current?.readyState === "open") {
            chunkChannelRef.current.send(JSON.stringify({ sentence }));
        }

        const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        audioCtxRef.current = audioCtx;

        await audioCtx.audioWorklet.addModule("recorder-processor.js");
        const workletNode = new AudioWorkletNode(audioCtx, "recorder-processor");
        workletNodeRef.current = workletNode;

        let totalSamplesSent = 0;

        workletNode.port.onmessage = (evt) => {
            const chunk = evt.data; // Float32Array
            bufferRef.current.push(...chunk);

            while (bufferRef.current.length >= CHUNK_SIZE) {
                // Lấy chunk hiện tại: CHUNK_SIZE + BUFFER_SIZE
                const sendChunk = bufferRef.current.slice(0, CHUNK_SIZE + BUFFER_SIZE);

                // Gửi **toàn bộ CHUNK_SIZE + BUFFER_SIZE** để chunk sau còn overlap
                sendChunkRealtime(sendChunk, totalSamplesSent);

                // Lưu CHUNK_SIZE đầu vào audioChunksRef (để merge)
                audioChunksRef.current.push(new Float32Array(sendChunk.slice(0, CHUNK_SIZE)));

                // Cập nhật totalSamplesSent chỉ tăng CHUNK_SIZE (không tăng BUFFER_SIZE)
                totalSamplesSent += CHUNK_SIZE;

                // Giữ lại BUFFER_SIZE cuối để overlap với chunk sau
                bufferRef.current = bufferRef.current.slice(CHUNK_SIZE);
            }
        };

        const source = audioCtx.createMediaStreamSource(localStreamRef.current);
        source.connect(workletNode);
        workletNode.connect(audioCtx.destination);

        setRecording(true);
        setAudioUrl(null);
        setFeedback([]);
    };

    // --- Stop recording ---
    const stopRecording = () => {
    if (!recording) return null;

    let totalSamplesSent = audioChunksRef.current.reduce((sum, c) => sum + c.length, 0);

    // --- Gửi nốt buffer còn lại ---
    if (bufferRef.current.length > 0) {
        sendChunkRealtime(bufferRef.current, totalSamplesSent);
        audioChunksRef.current.push(new Float32Array(bufferRef.current));
        console.log(`📤 Sent final buffer [${totalSamplesSent}:${totalSamplesSent + bufferRef.current.length}]`);
        bufferRef.current = [];
    }

    // --- Tắt audio sau khi gửi hết ---
    workletNodeRef.current.disconnect();
    audioCtxRef.current.close();

    setRecording(false);

    if (audioChunksRef.current.length === 0) return null;

    // Merge tất cả audio chunks để tạo WAV
    const totalLength = audioChunksRef.current.reduce((sum, c) => sum + c.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    audioChunksRef.current.forEach((chunk) => {
        merged.set(chunk, offset);
        offset += chunk.length;
    });

    const wavBlob = encodeWAV16kMono(merged);
    const url = URL.createObjectURL(wavBlob);
    setAudioUrl(url);

    audioChunksRef.current = [];
    return url;
};


    return { recording, feedback, audioUrl, startRecording, stopRecording };
}
