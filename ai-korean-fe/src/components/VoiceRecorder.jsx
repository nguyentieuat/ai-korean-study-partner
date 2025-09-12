import React, { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
const backendUrl = import.meta.env.VITE_API_URL;

const VoiceRecorder = ({
  conversation_id,
  onConversationId, // optional: parent có thể truyền để nhận id mới từ BE
  onComplete,
  history,
  setHistory,
  getFormattedHistoryForServer,
  voice = 1, // 1 = Nam (default theo yêu cầu), 0 = Nữ
  // Track
  sessionId: sessionIdProp,
  userIdHash: userIdHashProp,
  onTrackEvent,
}) => {
  const [recording, setRecording] = useState(false);
  const audioChunksRef = useRef([]);
  const mediaRecorderRef = useRef(null);
  const audioBlobRef = useRef(null);
  const waveformContainerRef = useRef(null);
  const waveSurferRef = useRef(null);
  const objectUrlRef = useRef(null);

  // === Tracking endpoint (khớp FastAPI /api/track/event) ===
  const TRACK_URL = `${backendUrl}/track/event`;
  // === Tracking identities ===
  const [sessionId, setSessionId] = useState(sessionIdProp || "");
  const [userIdHash, setUserIdHash] = useState(userIdHashProp || "");

  // Nếu cha không truyền, tự tạo id ẩn danh + sessionId cho phiên hiện tại
  useEffect(() => {
    if (!userIdHashProp) {
      let uid = localStorage.getItem("uid_hash");
      if (!uid) {
        uid = crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
        localStorage.setItem("uid_hash", uid);
      }
      setUserIdHash(uid);
    } else {
      setUserIdHash(userIdHashProp);
    }

    if (!sessionIdProp) {
      const day = new Date().toISOString().slice(0, 10).replace(/-/g, "");
      const rand = Math.random().toString(36).slice(2, 6).toUpperCase();
      setSessionId(`S-${day}-${rand}`);
    } else {
      setSessionId(sessionIdProp);
    }
  }, [sessionIdProp, userIdHashProp]);

  async function postTrack(payload) {
    try {
      if (typeof onTrackEvent === "function") {
        // cho phép cha tự xử lý nếu muốn
        return onTrackEvent(payload);
      }
      const event_id =
        crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
      await fetch(TRACK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_id, ...payload }),
        keepalive: true,
      });
    } catch {
      // nuốt lỗi để không ảnh hưởng UX
    }
  }

  useEffect(() => {
    return () => {
      try {
        waveSurferRef.current?.destroy();
      } catch {}
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setRecording(true);
      audioChunksRef.current = [];

      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        audioBlobRef.current = blob;

        // Waveform preview
        const audioUrl = URL.createObjectURL(blob);
        objectUrlRef.current = audioUrl;

        try {
          waveSurferRef.current?.destroy();
        } catch {}
        waveSurferRef.current = WaveSurfer.create({
          container: waveformContainerRef.current,
          height: 60,
          waveColor: "violet",
          progressColor: "purple",
        });
        waveSurferRef.current.load(audioUrl);

        // User voice placeholder
        const userVoiceMsgTemp = {
          role: "user",
          transcript: "⏳ Đang nhận diện giọng nói...",
          waitingReply: true,
          id: Date.now(),
          isVoice: true,
        };
        setHistory((prev) => [...prev, userVoiceMsgTemp]);

        // --- 1) TRANSCRIBE ---
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", file);

        try {
          const res1 = await fetch(`${backendUrl}/transcribe`, {
            method: "POST",
            body: formData,
          });
          const data1 = await res1.json();
          if (!res1.ok) throw new Error(data1?.error || "ASR error");

          const transcript = data1?.transcript || "";
          const audio_url_goc = data1?.audio_url_goc || ""; // data URL base64 từ BE

          // bỏ placeholder user voice temp
          setHistory((prev) =>
            prev.filter((m) => m.id !== userVoiceMsgTemp.id)
          );

          // User message thực sự
          const userMessage = {
            role: "user",
            transcript,
            audioUrl: audio_url_goc, // FE có thể play bằng <audio src={dataURL} />
          };

          // --- 2) ONCOMPLETE: tạo AI placeholder & gọi talking ---
          onComplete(async (updateAiMessage) => {
            try {
              const aiPlaceholder = {
                role: "ai",
                reply: "⏳ Đang phản hồi...",
                typing: true,
                id: Date.now(),
              };
              setHistory((prev) => [...prev, userMessage, aiPlaceholder]);

              const formattedHistory = getFormattedHistoryForServer([
                ...history,
                userMessage,
              ]);

              const payload = {
                transcript,
                history: formattedHistory,
                voice, // gửi giọng (0/1)
                ...(conversation_id ? { conversation_id } : {}),
              };

              // === đo latency FE cho voice chat_turn
              const t0 = performance.now();

              const res2 = await fetch(
                `${backendUrl}/korean-speaking-talking`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(payload),
                }
              );
              const data2 = await res2.json();

              const latencyMs = Math.round(performance.now() - t0);

              // nếu lần đầu chưa có conversation_id → nhận từ BE
              if (
                !conversation_id &&
                data2?.conversation_id &&
                typeof onConversationId === "function"
              ) {
                onConversationId(data2.conversation_id);
              }

              updateAiMessage({
                role: "ai",
                replyTTS: data2.ai_reply_tts || "",
                aiVoiceUrl: data2.tts_audio_url || "",
                highlighted: data2.highlighted || "",
                typing: false,
              });

              // === TRACK: chat_turn (voice)
              postTrack({
                user_id_hash: userIdHash,
                session_id: sessionId,
                event_type: "chat_turn",
                duration_ms: latencyMs,
                meta: { mode: "voice" },
              });
            } catch (error) {
              console.error("Lỗi khi lấy phản hồi AI:", error);
              updateAiMessage({
                role: "ai",
                replyTTS: "❌ Lỗi phản hồi từ AI.",
                typing: false,
              });

              // (tuỳ chọn) log lỗi cũng bằng chat_turn với cờ error
              const latencyMsErr = 0; // không đo được sau exception? để 0 hoặc tự lưu lần gần nhất
              postTrack({
                user_id_hash: userIdHash,
                session_id: sessionId,
                event_type: "chat_turn",
                duration_ms: latencyMsErr,
                meta: { mode: "voice", error: true },
              });
            }
          });
        } catch (err) {
          // bỏ placeholder nếu transcribe lỗi
          setHistory((prev) =>
            prev.filter((m) => m.id !== userVoiceMsgTemp.id)
          );
          console.error("Lỗi xử lý audio:", err);
          // đẩy một tin lỗi nhẹ (tuỳ chọn)
          setHistory((prev) => [
            ...prev,
            {
              role: "ai",
              reply: "⚠️ Không nhận diện được giọng nói.",
              typing: false,
            },
          ]);
        }
      };

      mediaRecorderRef.current.start();
    } catch (error) {
      console.error("Không thể truy cập microphone:", error);
    }
  };

  const handleStop = () => {
    setRecording(false);
    try {
      mediaRecorderRef.current?.stop();
      // dừng tracks để nhả mic
      mediaRecorderRef.current?.stream?.getTracks()?.forEach((t) => t.stop());
    } catch {}
  };

  return (
    <div>
      <button onClick={recording ? handleStop : handleStart}>
        {recording ? "⏹ Dừng ghi âm" : "🎙 Bắt đầu ghi âm"}
      </button>
      <div ref={waveformContainerRef} style={{ marginTop: 10 }} />
    </div>
  );
};

export default VoiceRecorder;
