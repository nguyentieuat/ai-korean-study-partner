import React, { useState, useRef, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
const backendUrl = import.meta.env.VITE_API_URL;

const VoiceRecorder = ({
  conversation_id,
  onComplete,
  history,
  setHistory,
  getFormattedHistoryForServer,
}) => {
  const [recording, setRecording] = useState(false);
  const audioChunksRef = useRef([]);
  const mediaRecorderRef = useRef(null);
  const audioBlobRef = useRef(null);
  const waveformContainerRef = useRef(null);
  const waveSurferRef = useRef(null);

  useEffect(() => {
    return () => {
      waveSurferRef.current?.destroy();
    };
  }, []);

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setRecording(true);
      audioChunksRef.current = [];

      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const audioUrl = URL.createObjectURL(blob);
        audioBlobRef.current = blob;

        if (waveSurferRef.current) {
          waveSurferRef.current.destroy();
        }

        waveSurferRef.current = WaveSurfer.create({
          container: waveformContainerRef.current,
          waveColor: "violet",
          progressColor: "purple",
          height: 60,
        });

        waveSurferRef.current.load(audioUrl);

        // 1️⃣ Tạo user message voice tạm thời ngay khi dừng mic
        const userVoiceMsgTemp = {
          role: "user",
          transcript: "⏳ Đang nhận diện giọng nói...",
          waitingReply: true,
          id: Date.now(),
          isVoice: true, // đánh dấu để phân biệt với text
        };

        // 2️⃣ Cập nhật history ngay → TypingIndicator hiện
        setHistory((prev) => [...prev, userVoiceMsgTemp]);

        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", file);

        try {
          // 1. Gửi audio để lấy transcript
          const res1 = await fetch(`${backendUrl}/api/transcribe`, {
            method: "POST",
            body: formData,
          });
          const data1 = await res1.json();
          const transcript = data1.transcript;
          const audio_url_goc = data1.audio_url_goc;

          // 2. Tạo tin nhắn người dùng
          const userMessage = {
            role: "user",
            transcript,
            audioUrl: audio_url_goc,
          };

          // Xóa waitingReply ở user message vừa gửi
          setHistory((prev) =>
            prev.filter((msg) => msg.id !== userVoiceMsgTemp.id)
          );

          // 3. Bắn userMessage ra UI và đợi phản hồi AI
          onComplete(async (updateAiMessage) => {
            try {
              // 3.1 Thêm AI placeholder ngay
              const aiPlaceholder = {
                role: "ai",
                reply: "⏳ Đang phản hồi...",
                typing: true,
                id: Date.now(),
              };
              setHistory((prev) => [...prev, userMessage, aiPlaceholder]);

              // Chuẩn bị history gửi server
              const formattedHistory = getFormattedHistoryForServer([
                ...history,
                userMessage,
              ]);
              // 5. Gửi đến AI server
              const res2 = await fetch(
                `${backendUrl}/api/korean-speaking-talking`,
                {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    conversation_id: conversation_id,
                    transcript: transcript,
                    history: formattedHistory,
                  }),
                }
              );

              const data2 = await res2.json();

              // 6. Khi có kết quả → cập nhật lại message AI
              updateAiMessage({
                role: "ai",
                replyTTS: data2.ai_reply_tts || "",
                aiVoiceUrl: data2.tts_audio_url || "",
                typing: false,
                highlighted: data2.highlighted || "",
              });
            } catch (error) {
              console.error("Lỗi khi lấy phản hồi AI:", error);
              updateAiMessage({
                role: "ai",
                replyTTS: "❌ Lỗi phản hồi từ AI.",
                typing: false,
              });
            }
          });
        } catch (err) {
          setHistory((prev) =>
            prev.filter((msg) => msg.id !== userVoiceMsgTemp.id)
          );
          console.error("Lỗi xử lý audio:", err);
        }
      };

      mediaRecorderRef.current.start();
    } catch (error) {
      console.error("Không thể truy cập microphone:", error);
    }
  };

  const handleStop = () => {
    setRecording(false);
    mediaRecorderRef.current?.stop();
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
