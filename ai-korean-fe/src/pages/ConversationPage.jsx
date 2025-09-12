import React, { useState, useRef } from "react";
import "./ConversationPage.css";
import VoiceRecorder from "../components/VoiceRecorder";
// import { v4 as uuidv4 } from "uuid"; // không còn dùng

const backendUrl = import.meta.env.VITE_API_URL;

const TypingIndicator = () => (
  <div className="typing-indicator">
    <div className="typing-dot"></div>
    <div className="typing-dot"></div>
    <div className="typing-dot"></div>
  </div>
);

const LS_KEY_CONV_ID = "korean_conv_id";
const LS_KEY_HISTORY = "korean_conv_history";
const LS_KEY_VOICE = "korean_voice"; // lưu lựa chọn giọng

const ConversationPage = () => {
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const [mode, setMode] = useState("text"); // "text" | "voice"
  const [recordTime, setRecordTime] = useState(0);
  const [playingUrl, setPlayingUrl] = useState(null);

  const [conversationId, setConversationId] = useState(
    () => localStorage.getItem(LS_KEY_CONV_ID) || null
  );
  const [history, setHistory] = useState(() => {
    try {
      const raw = localStorage.getItem(LS_KEY_HISTORY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      if (!Array.isArray(arr)) return [];
      return arr.map((m) => ({ ...m, typing: false, waitingReply: false }));
    } catch {
      return [];
    }
  });

  // giọng AI: female/male
  const [voice, setVoice] = useState(() => {
    const raw = localStorage.getItem(LS_KEY_VOICE);
    // migrate giá trị cũ "male"/"female" nếu có
    if (raw === "male") return 1;
    if (raw === "female") return 0;
    return raw === null ? 1 : Number(raw);
  });

  React.useEffect(() => {
    try {
      localStorage.setItem(LS_KEY_HISTORY, JSON.stringify(history));
    } catch {}
  }, [history]);

  React.useEffect(() => {
    try {
      conversationId && localStorage.setItem(LS_KEY_CONV_ID, conversationId);
    } catch {}
  }, [conversationId]);

  React.useEffect(() => {
    try {
      localStorage.setItem(LS_KEY_VOICE, voice);
    } catch {}
  }, [voice]);

  const newChat = () => {
    setHistory([]);
    setConversationId(null);
    try {
      localStorage.removeItem(LS_KEY_HISTORY);
      localStorage.removeItem(LS_KEY_CONV_ID);
    } catch {}
  };

  // Gửi tin nhắn dạng text
  const handleSendText = async () => {
    if (!input.trim()) return;

    const newUserMessage = { role: "user", content: input, waitingReply: true };
    const placeholderReply = {
      role: "ai",
      reply: "⏳ Đang phản hồi...",
      typing: true,
    };

    const updatedHistory = [...history, newUserMessage, placeholderReply];
    setHistory(updatedHistory);
    setInput("");
    setIsTyping(true);

    const formattedHistory = getFormattedHistoryForServer(updatedHistory);

    try {
      const response = await fetch(
        `${backendUrl}/korean-speaking-chatting`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: input,
            conversation_history: formattedHistory,
            voice, // gửi kèm lựa chọn giọng (BE có thể bỏ qua nếu chưa dùng)
            ...(conversationId ? { conversation_id: conversationId } : {}),
          }),
        }
      );

      const data = await response.json();
      if (!conversationId && data.conversation_id)
        setConversationId(data.conversation_id);

      const reply = data.reply || "죄송합니다. 다시 말씀해 주세요.";
      const highlighted = data.highlighted || "";

      setHistory((prev) => {
        const clearedWaiting = prev.map((msg) =>
          msg.waitingReply ? { ...msg, waitingReply: false } : msg
        );
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
    const src = audioUrl.startsWith("data:")
      ? audioUrl
      : `${backendUrl}${audioUrl}`;
    const audio = new Audio(src);
    audio.onended = () => setPlayingUrl(null);
    audio.play().catch(() => setPlayingUrl(null));
  };

  const getFormattedHistoryForServer = (history) => {
    return history
      .filter((msg) => {
        if (msg.role === "user" && !msg.waitingReply) return true;
        if (msg.role === "ai") {
          const content = msg.reply || msg.replyTTS || "";
          return !(
            content.includes("❌") ||
            content.includes("⚠️ 오류") ||
            content.includes("죄송합니다. 다시 말씀해 주세요.") ||
            content.includes("⏳ Đang phản hồi...")
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
      <div className="header-row">
        <h2>🗣️ AI Korean Conversation Partner</h2>
        <div className="controls-row">
          <div className="voice-chooser">
            <span>Giọng AI:</span>
            <label style={{ marginLeft: 8 }}>
              <input
                type="radio"
                name="voice"
                value={0}
                checked={voice === 0}
                onChange={() => setVoice(0)}
              />{" "}
              Nữ
            </label>
            <label style={{ marginLeft: 12 }}>
              <input
                type="radio"
                name="voice"
                value={1}
                checked={voice === 1}
                onChange={() => setVoice(1)}
              />{" "}
              Nam
            </label>
          </div>
        </div>
      </div>

      <div className="topbar">
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

          <label
            style={{ marginLeft: "1rem" }}
            onClick={newChat}
            className="clear-history-btn"
          >
            🗑 Clean History
          </label>
        </div>

        {/* <button
          onClick={newChat}
          className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-medium px-3 py-2 rounded-lg transition"
          title="Xóa lịch sử & bắt đầu cuộc mới"
        >
          🗑 Xóa lịch sử
        </button> */}
      </div>

      <div className="chat-box">
        {history.map((item, index) => {
          if (item.role === "user") {
            const aiReply = history[index + 1];
            return (
              <div key={index} className="chat-message user">
                <strong>👤 Bạn:</strong>{" "}
                {item.content && <div>{item.content}</div>}
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
            voice={voice} // truyền giọng cho recorder (để backend TTS dùng)
            onComplete={(updateFn) => {
              updateFn((aiMessage) => {
                setHistory((prev) => {
                  const withoutTemp = prev.filter(
                    (msg) => !(msg.role === "ai" && msg.typing)
                  );
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
