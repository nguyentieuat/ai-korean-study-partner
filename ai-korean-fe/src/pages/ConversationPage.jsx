import React, { useState, useRef } from "react";
import "./ConversationPage.css";
import VoiceRecorder from "../components/VoiceRecorder";
import { v4 as uuidv4 } from "uuid";

const backendUrl = import.meta.env.VITE_API_URL;

const TypingIndicator = () => (
  <div className="typing-indicator">
    <div className="typing-dot"></div>
    <div className="typing-dot"></div>
    <div className="typing-dot"></div>
  </div>
);

const ConversationPage = () => {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [mode, setMode] = useState("text"); // "text" | "voice"
  const [recordTime, setRecordTime] = useState(0);
  const [playingUrl, setPlayingUrl] = useState(null);
  const [conversationId] = useState(uuidv4());

  // Hàm gửi tin nhắn dạng text
  const handleSendText = async () => {
    if (!input.trim()) return;

    // Tạo message user với waitingReply = true (chờ phản hồi)
    const newUserMessage = { role: "user", content: input, waitingReply: true };
    // Tạo message AI tạm thời (loading)
    const placeholderReply = {
      role: "ai",
      reply: "⏳ Đang phản hồi...",
      typing: true,
    };

    // Cập nhật lịch sử để hiển thị ngay
    const updatedHistory = [...history, newUserMessage, placeholderReply];
    setHistory(updatedHistory);
    setInput("");
    setIsTyping(true);

    // Chuẩn bị dữ liệu gửi server
    const formattedHistory = getFormattedHistoryForServer(updatedHistory);

    try {
      const response = await fetch(
        `${backendUrl}/api/korean-speaking-chatting`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            conversation_id: conversationId,
            message: input,
            conversation_history: formattedHistory,
          }),
        }
      );

      const data = await response.json();
      const reply = data.reply || "죄송합니다. 다시 말씀해 주세요.";
      const highlighted = data.highlighted || "";

      setHistory((prev) => {
        // Xóa flag waitingReply ở user message cuối cùng
        const clearedWaiting = prev.map((msg) =>
          msg.waitingReply ? { ...msg, waitingReply: false } : msg
        );

        // Cập nhật message AI cuối cùng (thay message loading)
        const updated = [...clearedWaiting];
        updated[updated.length - 1] = {
          role: "ai",
          reply,
          highlighted,
          typing: false,
        };
        return updated;
      });
    } catch (error) {
      setHistory((prev) => {
        const clearedWaiting = prev.map((msg) =>
          msg.waitingReply ? { ...msg, waitingReply: false } : msg
        );
        const updated = [...clearedWaiting];
        updated[updated.length - 1] = {
          role: "ai",
          reply: "⚠️ 오류가 발생했습니다. 다시 시도해 주세요.",
          typing: false,
        };
        return updated;
      });
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendText();
    }
  };

  const playAudio = (audioUrl) => {
    if (!audioUrl) return;
    setPlayingUrl(audioUrl);
    const audio = new Audio(`${backendUrl}${audioUrl}`);
    audio.onended = () => {
      setPlayingUrl(null); // Cho phép phát lại
    };

    audio.play().catch(() => {
      setPlayingUrl(null); // Nếu lỗi cũng enable lại
    });
  };

  const getFormattedHistoryForServer = (history) => {
    return history
      .filter((msg) => {
        if (msg.role === "user" && !msg.waitingReply) return true; // user bình thường
        if (msg.role === "ai") {
          const content = msg.reply || msg.replyTTS || "";
          return !(
            content.includes("❌") ||
            content.includes("⚠️ 오류") ||
            content.includes("죄송합니다. 다시 말씀해 주세요.")
          );
        }
        return false;
      })
      .map((msg) => ({
        role: msg.role === "ai" ? "assistant" : "user",
        message:
          msg.content || msg.transcript || msg.reply || msg.replyTTS || "",
      }))
      .filter((item) => item.message.trim() !== "");
  };

  return (
    <div className="speaking-container">
      <h2>🗣️ AI Korean Conversation Partner</h2>
      <div className="mode-switch">
        <label>
          <input
            type="radio"
            name="mode"
            value="text"
            checked={mode === "text"}
            onChange={() => setMode("text")}
          />
          ✍️ Text
        </label>
        <label style={{ marginLeft: "1rem" }}>
          <input
            type="radio"
            name="mode"
            value="voice"
            checked={mode === "voice"}
            onChange={() => setMode("voice")}
          />
          🎤 Voice
        </label>
      </div>

      <div className="chat-box">
        {history.map((item, index) => {
          if (item.role === "user") {
            const aiReply = history[index + 1];
            return (
              <div key={index} className="chat-message user">
                <strong>👤 Bạn:</strong>{" "}
                {item.content && <div>{item.content}</div>}
                {/* Hiển thị TypingIndicator nếu user message đang chờ reply */}
                {item.waitingReply && <TypingIndicator />}
                {item.audioUrl && (
                  <div className="mt-2">
                    <button
                      className="btn btn-primary me-2"
                      onClick={() => playAudio(item.audioUrl)}
                      disabled={playingUrl === item.audioUrl}
                    >
                      🔊
                    </button>
                  </div>
                )}
                {item.transcript && (
                  <div className="mt-1 ms-3 fst-italic text-secondary">
                    📝 {item.transcript}
                  </div>
                )}
                {aiReply?.highlighted && (
                  <div
                    className="highlighted-reply"
                    dangerouslySetInnerHTML={{
                      __html: `<strong>✍️ AI chỉnh sửa:</strong> ${aiReply.highlighted}`,
                    }}
                  />
                )}
              </div>
            );
          }

          if (item.role === "ai") {
            return (
              <div key={index} className="chat-message ai">
                <strong>🤖 AI:</strong>{" "}
                {item.typing ? (
                  <TypingIndicator />
                ) : (
                  <>
                    {item.reply && <div>{item.reply}</div>}

                    {item.aiVoiceUrl && (
                      <div className="mt-2">
                        <button
                          className="btn btn-primary me-2"
                          onClick={() => playAudio(item.aiVoiceUrl)}
                          disabled={playingUrl === item.aiVoiceUrl}
                        >
                          🔊
                        </button>
                      </div>
                    )}
                    {item.replyTTS && (
                      <div className="mt-1 ms-3 fst-italic text-secondary">
                        📝 {item.replyTTS}
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          }

          return null;
        })}
      </div>

      <div className="input-box">
        {mode === "text" ? (
          <>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Nhập tin nhắn của bạn..."
            />
            <button
              onClick={handleSendText}
              className="bg-blue-500 hover:bg-blue-600 text-white font-medium px-4 py-2 rounded-xl transition"
            >
              Gửi
            </button>
          </>
        ) : (
          <VoiceRecorder
            conversation_id={conversationId}
            history={history}
            setHistory={setHistory}
            getFormattedHistoryForServer={getFormattedHistoryForServer}
            onComplete={(updateFn) => {
              // 4. Gọi updateFn để cập nhật message AI khi có phản hồi
              updateFn((aiMessage) => {
                setHistory((prev) => {
                  // Xóa message AI loading
                  const withoutTemp = prev.filter(
                    (msg) => !(msg.role === "ai" && msg.typing)
                  );
                  // Thêm message AI thật
                  return [...withoutTemp, aiMessage];
                });
              });
            }}
          />
        )}
      </div>
    </div>
  );
};

export default ConversationPage;
