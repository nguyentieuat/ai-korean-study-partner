// src/components/InlineMic.jsx
import React, { useRef, useState, useEffect } from "react";

const backendUrl = import.meta.env.VITE_API_URL;

export default function InlineMic({
  conversationId,
  onConversationId,         // optional: nhận id mới từ BE lần đầu
  history,
  setHistory,
  getFormattedHistoryForServer,
  voice = 1,
  // tracking (optional)
  postTrackEvent,
  sessionId,
  userIdHash,
}) {
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  useEffect(() => {
    return () => {
      // cleanup stream khi unmount
      try {
        mediaRecorderRef.current?.stream?.getTracks()?.forEach((t) => t.stop());
      } catch {}
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });

        // 1) Hiển thị “đang nhận diện…”
        const tempUser = {
          role: "user",
          transcript: "⏳ Đang nhận diện giọng nói...",
          waitingReply: true,
          id: Date.now(),
          isVoice: true,
        };
        setHistory((prev) => [...prev, tempUser]);

        // 2) Gửi ASR
        try {
          const file = new File([blob], "recording.webm", { type: "audio/webm" });
          const form = new FormData();
          form.append("audio", file);

          const r1 = await fetch(`${backendUrl}/transcribe`, { method: "POST", body: form });
          const d1 = await r1.json();
          if (!r1.ok) throw new Error(d1?.error || "ASR error");

          const transcript = d1?.transcript || "";
          const audioUrl = d1?.audio_url_goc || ""; // data:… từ BE

          // bỏ placeholder
          setHistory((prev) => prev.filter((m) => m.id !== tempUser.id));

          // user voice thật
          const userMsg = { role: "user", transcript, audioUrl };

          // 3) Placeholder AI + gọi talking
          const aiPlaceholder = {
            role: "ai",
            reply: "⏳ Đang phản hồi...",
            typing: true,
            id: Date.now(),
          };
          setHistory((prev) => [...prev, userMsg, aiPlaceholder]);

          const formatted = getFormattedHistoryForServer([...history, userMsg]);

          const payload = {
            transcript,
            history: formatted,
            voice,
            ...(conversationId ? { conversation_id: conversationId } : {}),
          };

          const t0 = performance.now();
          const r2 = await fetch(`${backendUrl}/korean-speaking-talking`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const d2 = await r2.json();
          const latencyMs = Math.round(performance.now() - t0);

          if (!conversationId && d2?.conversation_id && typeof onConversationId === "function") {
            onConversationId(d2.conversation_id);
          }

          setHistory((prev) => {
            // thay placeholder cuối cùng bằng câu trả lời thật
            const last = [...prev];
            const idx = last.findIndex((m) => m.id === aiPlaceholder.id);
            if (idx !== -1) {
              last[idx] = {
                role: "ai",
                replyTTS: d2.ai_reply_tts || "",
                aiVoiceUrl: d2.tts_audio_url || "",
                highlighted: d2.highlighted || "",
                typing: false,
              };
            }
            return last;
          });

          postTrackEvent?.({
            user_id_hash: userIdHash,
            session_id: sessionId,
            event_type: "chat_turn",
            duration_ms: latencyMs,
            meta: { mode: "voice" },
          });
        } catch (e) {
          // bỏ placeholder nếu lỗi
          setHistory((prev) => prev.filter((m) => m.id !== tempUser.id));
          setHistory((prev) => [
            ...prev,
            { role: "ai", reply: "⚠️ Không nhận diện được giọng nói.", typing: false },
          ]);
          postTrackEvent?.({
            user_id_hash: userIdHash,
            session_id: sessionId,
            event_type: "chat_turn",
            duration_ms: 0,
            meta: { mode: "voice", error: true },
          });
        } finally {
          try {
            mediaRecorderRef.current?.stream?.getTracks()?.forEach((t) => t.stop());
          } catch {}
        }
      };

      mediaRecorderRef.current.start();
      setRecording(true);
    } catch (err) {
      console.error("Không thể truy cập microphone:", err);
    }
  };

  const stopRecording = () => {
    setRecording(false);
    try {
      mediaRecorderRef.current?.stop();
    } catch {}
  };

  return (
    <button
      type="button"
      onClick={recording ? stopRecording : startRecording}
      className={`mic-btn ${recording ? "recording" : ""}`}
      title={recording ? "Dừng ghi âm" : "Bắt đầu ghi âm"}
    >
      {recording ? "⏹" : "🎙"}
      {recording && <span className="recording-dot" />}
    </button>
  );
}
