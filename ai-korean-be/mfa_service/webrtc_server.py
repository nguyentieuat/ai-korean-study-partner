import asyncio
import json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
import aiohttp_cors
import numpy as np
from g2p_needleman_wunsch_alignment.g2p_needleman_alignment_chuck import evaluate_chunk
from g2p_needleman_wunsch_alignment.g2p_needleman_alignment import evaluate
from data_model_first.mapping_model_first import get_paths_by_text

pcs = set()
peer_buffers = {}

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    peer_buffers[pc] = {"sentence": None, "chunks": []}

    @pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        print("ICE state:", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            peer_buffers.pop(pc, None)

    @pc.on("datachannel")
    def on_datachannel(channel):
        print("📡 DataChannel opened:", channel.label)

        @channel.on("message")
        def on_message(message):
            # --- string message ---
            if isinstance(message, str):
                data = json.loads(message)
                if data.get("sentence"):
                    peer_buffers[pc]["sentence"] = data["sentence"]
                    print("📝 Sentence received:", data["sentence"])
            # --- bytes message ---
            elif isinstance(message, bytes):
                if len(message) < 12:
                    print("⚠️ Invalid chunk, too small")
                    return

                header = message[:12]
                payload = message[12:]

                # parse header
                view = memoryview(header)
                start_sample = int.from_bytes(view[0:4], "little")
                end_sample = int.from_bytes(view[4:8], "little")
                chunk_bytes_len = int.from_bytes(view[8:12], "little")

                print(f"📦 Chunk received: start={start_sample}, end={end_sample}, length={chunk_bytes_len}")
                # parse Float32
                chunk_float32 = np.frombuffer(payload, dtype='<f4').copy()
                if np.all(chunk_float32 == 0):
                    print("⚠️ Chunk empty, skipping ASR")
                else:
                    import soundfile as sf
                    chunk_filename = f"chunk_{start_sample}_{end_sample}.wav"
                    sf.write(chunk_filename, chunk_float32, 16000)  # sample rate phải khớp Whisper

                sentence = peer_buffers[pc]["sentence"]
                if sentence:
                    try:
                        _, textgrid_file = get_paths_by_text(sentence)
                        result = evaluate(chunk_filename, sentence, textgrid_file)
                        print("✅ Chunk evaluated:", result)
                    except Exception as e:
                        print("[ERROR] evaluate_chunk failed:", e)

                # lưu vào buffer với start/end
                peer_buffers[pc]["chunks"].append({
                    "data": chunk_float32,
                    "startSample": start_sample,
                    "endSample": end_sample
                })

    # --- set remote description ---
    await pc.setRemoteDescription(offer)

    # tạo feedback channel
    feedback_channel = pc.createDataChannel("feedback")
    pc.feedback_channel = feedback_channel

    @feedback_channel.on("open")
    def on_feedback_open():
        print("📡 Feedback channel ready")

    # chỉ tạo answer khi signalingState đúng
    if pc.signalingState == "have-remote-offer":
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    elif pc.signalingState == "stable":
        # đôi khi stable vì offer đã được set trước, chỉ log
        print("[WARN] Signaling state stable, using existing localDescription")
    else:
        print("[WARN] Unexpected signalingState:", pc.signalingState)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    peer_buffers.clear()


app = web.Application()
app.router.add_post("/api/webrtc/offer", offer)
app.on_shutdown.append(on_shutdown)

# bật CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods=["GET", "POST", "OPTIONS"],
    )
})
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=5004)
