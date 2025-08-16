import React, { useState, useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
const backendUrl = import.meta.env.VITE_API_URL;

const PronunciationPage = () => {
  const [mode, setMode] = useState("level");
  const [level, setLevel] = useState(1);
  const [topic, setTopic] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [sampleData, setSampleData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [recording, setRecording] = useState(false);
  const [userAudio, setUserAudio] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [evaluating, setEvaluating] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const waveformContainerRef = useRef(null);
  const waveSurferRef = useRef(null);
  const [hoveredWordIndex, setHoveredWordIndex] = useState(null);

  // Chủ đề tĩnh
  const dataByTopic = {
    "Sinh hoạt": [
      {
        text: "밥 먹었어요?",
        notes: "Bạn đã ăn cơm chưa?",
        pronunciation: "bap meo-geo-sseo-yo",
      },
      {
        text: "집에 가요.",
        notes: "Tôi về nhà.",
        pronunciation: "jib-e ga-yo",
      },
    ],
    "Du lịch": [
      {
        text: "이 호텔 예약했어요.",
        notes: "Tôi đã đặt phòng khách sạn.",
        pronunciation: "i hotel ye-yak-hae-sseo-yo",
      },
      {
        text: "공항으로 가 주세요.",
        notes: "Làm ơn đưa tôi đến sân bay.",
        pronunciation: "gong-hang-eu-ro ga ju-se-yo",
      },
    ],
    "Nhà hàng": [
      {
        text: "메뉴 좀 보여 주세요.",
        notes: "Cho tôi xem thực đơn.",
        pronunciation: "me-nyu jom bo-yeo ju-se-yo",
      },
      {
        text: "계산서 부탁해요.",
        notes: "Làm ơn tính tiền.",
        pronunciation: "gye-san-seo bu-tak-hae-yo",
      },
    ],
    "Công xưởng": [
      {
        text: "안전모를 착용하세요.",
        notes: "Hãy đội mũ bảo hộ.",
        pronunciation: "an-jeon-mo-reul chak-yong-ha-se-yo",
      },
      {
        text: "작업을 시작합니다.",
        notes: "Bắt đầu làm việc.",
        pronunciation: "jak-eob-eul si-jak-ham-ni-da",
      },
    ],
  };

  useEffect(() => {
    if (mode === "level") {
      setLoading(true);
      setError(null);
      setUserAudio(null);
      setEvaluation(null);
      setSelectedIndex(0);
      fetch(`${backendUrl}/api/pronunciation/${level}`)
        .then((res) => {
          if (!res.ok) throw new Error(`Lỗi tải dữ liệu cấp độ ${level}`);
          return res.json();
        })
        .then((data) => {
          setSampleData(data.items.items || []);
          setLoading(false);
        })
        .catch((err) => {
          setError(err.message);
          setSampleData([]);
          setLoading(false);
        });
    }
  }, [level, mode]);

  useEffect(() => {
    if (mode === "topic" && topic) {
      setSampleData(dataByTopic[topic] || []);
      setSelectedIndex(0);
      setUserAudio(null);
      setEvaluation(null);
    }
  }, [topic, mode]);

  const current = sampleData[selectedIndex] || {};

  const playSample = () => {
    if (!current.audioUrl || isPlaying) return;
    const audio = new Audio(`${backendUrl}${current.audioUrl}`);
    setIsPlaying(true);
    audio.play();
    audio.onended = () => setIsPlaying(false);
    audio.onerror = () => setIsPlaying(false);
  };

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setRecording(true);
      audioChunksRef.current = [];
      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (e) => {
        audioChunksRef.current.push(e.data);
      };
      mediaRecorderRef.current.start();
    } catch (err) {
      alert("Không thể truy cập microphone.");
      console.error(err);
    }
  };

  const handleStop = () => {
    setRecording(false);
    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.onstop = async () => {
      const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const audioUrl = URL.createObjectURL(blob);
      setUserAudio(audioUrl);

      // Hiển thị waveform
      if (waveSurferRef.current) waveSurferRef.current.destroy();
      waveSurferRef.current = WaveSurfer.create({
        container: waveformContainerRef.current,
        waveColor: "violet",
        progressColor: "purple",
        height: 60,
      });
      waveSurferRef.current.load(audioUrl);

      // Bắt đầu loading đánh giá
      setEvaluating(true);
      setEvaluation(null);

      // Gửi audio lên server để chấm điểm
      const file = new File([blob], "recording.webm", { type: "audio/webm" });
      const formData = new FormData();
      formData.append("audio", file);
      formData.append("text", current.text || "");

      try {
        const res = await fetch(`${backendUrl}/api/pronunciation/evaluate`, {
          method: "POST",
          body: formData,
        });
        const result = await res.json();
        setEvaluation(result); // { score, feedback }
      } catch (err) {
        console.error("Lỗi chấm điểm:", err);
      } finally {
        setEvaluating(false);
      }
    };
  };

  const handleChangeLevel = (lv) => {
    setMode("level");
    setLevel(lv);
  };

  const handleSelectTopic = (t) => {
    setMode("topic");
    setTopic(t);
  };

  // Hàm tách syllable Hangeul thành phoneme
  function splitHangulSyllable(syllable) {
    const CHOSEONG = 0x1100;
    const JUNGSEONG = 0x1161;
    const JONGSEONG = 0x11a7;
    const SBase = 0xac00;
    const LCount = 19;
    const VCount = 21;
    const TCount = 28;
    const NCount = VCount * TCount;

    const code = syllable.charCodeAt(0);
    if (code < SBase || code > 0xd7a3) return [syllable];

    const SIndex = code - SBase;
    const L = Math.floor(SIndex / NCount);
    const V = Math.floor((SIndex % NCount) / TCount);
    const T = SIndex % TCount;

    const chars = [
      String.fromCharCode(CHOSEONG + L),
      String.fromCharCode(JUNGSEONG + V),
    ];
    if (T !== 0) chars.push(String.fromCharCode(JONGSEONG + T));
    return chars;
  }

  // Map syllable → phoneme MFA
  function mapSyllableToPhonemeScore(text, phonemes) {
    const syllables = text.split("");
    const phonemesCopy = [...phonemes]; // tránh mutate gốc

    return syllables.map((syllable) => {
      const syllablePhonemes = splitHangulSyllable(syllable); // ký tự trong syllable
      const mapped = [];

      for (let i = 0; i < syllablePhonemes.length; i++) {
        if (phonemesCopy.length === 0) break;
        mapped.push(phonemesCopy.shift());
      }

      const avgScore =
        mapped.reduce((sum, p) => sum + (p.score || 0), 0) /
        (mapped.length || 1);

      return { syllable, phonemes: mapped, avgScore };
    });
  }

  // Map syllable -> phoneme + avgScore, bỏ qua space
  function mapSyllableToPhonemeScoreWithHangul(text, phonemes) {
    const result = [];
    const phonemesCopy = [...phonemes];

    text.split("").forEach((syllable) => {
      if (syllable === " ") {
        result.push({ hangul: " ", phoneme: [], avgScore: 1 });
        return;
      }

      const letters = splitHangulSyllable(syllable); // [초성, 중성, 종성]
      const mappedLetters = letters.map((letter) => {
        const p = phonemesCopy.shift();
        return {
          hangul: letter,
          phoneme: p?.phoneme || "?",
          score: p?.score || 0,
        };
      });

      const avgScore =
        mappedLetters.reduce((sum, l) => sum + l.score, 0) /
        (mappedLetters.length || 1);

      result.push({ hangulLetters: mappedLetters, avgScore });
    });

    return result;
  }

  return (
    <div className="container mt-4">
      <h2>
        Luyện phát âm -{" "}
        {mode === "level" ? `Cấp độ ${level}` : `Chủ đề: ${topic || "..."}`}
      </h2>

      {/* Menu cấp độ / chủ đề */}
      <div className="mb-4">
        {[1, 2, 3, 4, 5].map((lv) => (
          <button
            key={lv}
            className={`btn me-2 ${
              mode === "level" && level === lv
                ? "btn-primary"
                : "btn-outline-primary"
            }`}
            onClick={() => handleChangeLevel(lv)}
          >
            Cấp độ {lv}
          </button>
        ))}
        <button
          className={`btn ${
            mode === "topic" ? "btn-primary" : "btn-outline-primary"
          }`}
          onClick={() => {
            setMode("topic");
            setTopic(null);
          }}
        >
          Tùy chọn theo chủ đề
        </button>
      </div>

      {mode === "topic" && !topic && (
        <div className="mb-4">
          <h5>Chọn chủ đề:</h5>
          {Object.keys(dataByTopic).map((t) => (
            <button
              key={t}
              className="btn btn-outline-secondary me-2 mb-2"
              onClick={() => handleSelectTopic(t)}
            >
              {t}
            </button>
          ))}
        </div>
      )}

      {loading && <p>Đang tải dữ liệu...</p>}
      {error && <p className="text-danger">Lỗi: {error}</p>}

      {current.text && (
        <div className="card p-3">
          <h4>{current.text}</h4>
          <p>
            <strong>Chú thích:</strong> {current.notes}
          </p>
          <p>
            <strong>Phiên âm:</strong> {current.ipa || current.pronunciation}
          </p>

          <div className="mt-3">
            <button
              className="btn btn-primary me-2"
              onClick={playSample}
              disabled={isPlaying}
            >
              🔊 Nghe mẫu
            </button>
            <button
              className={`btn ${recording ? "btn-danger" : "btn-warning"}`}
              onClick={recording ? handleStop : handleStart}
              disabled={loading || !current.text || evaluating}
            >
              {recording ? "⏹ Dừng ghi âm" : "🎙️ Ghi âm"}
            </button>
          </div>

          <div ref={waveformContainerRef} style={{ marginTop: 10 }} />

          {userAudio && (
            <div className="mt-3">
              <h6>🔁 So sánh với bạn:</h6>
              <audio controls src={userAudio}></audio>
            </div>
          )}

          {evaluating && (
            <div className="mt-3 p-3 border rounded text-center">
              <div className="spinner-border text-primary" role="status" />
              <p className="mt-2">Đang chấm điểm...</p>
            </div>
          )}

          {evaluation && evaluation.detail && (
            <div className="mt-3 p-3 border rounded">
              <h6>📊 Kết quả đánh giá:</h6>
              <p>
                <strong>Điểm trung bình:</strong>{" "}
                {((evaluation.avg_score ?? 0) * 100).toFixed(0)}%
              </p>

              <div
                style={{
                  fontSize: "1.5rem",
                  lineHeight: "2rem",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    fontSize: "1.5rem",
                    lineHeight: "2rem",
                    position: "relative",
                  }}
                >
                  {mapSyllableToPhonemeScoreWithHangul(
                    current.text,
                    evaluation.detail
                  ).map((s, idx) => {
                    const { hangulLetters, avgScore } = s;

                    let color = "black";
                    if (hangulLetters.length > 0) {
                      if (avgScore >= 0.8) color = "green";
                      else if (avgScore >= 0.6) color = "orange";
                      else color = "red";
                    }

                    return (
                      <span
                        key={idx}
                        style={{
                          color,
                          marginRight: "0.05rem",
                          position: "relative",
                        }}
                      >
                        {hangulLetters.map((l, i) => (
                          <span
                            key={i}
                            style={{ cursor: "pointer" }}
                            onMouseEnter={() =>
                              setHoveredWordIndex(`${idx}-${i}`)
                            }
                            onMouseLeave={() => setHoveredWordIndex(null)}
                          >
                            {l.hangul}
                            {hoveredWordIndex === `${idx}-${i}` && (
                              <div
                                style={{
                                  position: "absolute",
                                  top: "-2rem",
                                  left: 0,
                                  background: "white",
                                  border: "1px solid #ccc",
                                  padding: "4px 8px",
                                  borderRadius: "4px",
                                  fontSize: "0.9rem",
                                  zIndex: 10,
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {`${l.hangul} (${l.phoneme} (${(
                                  l.score * 100
                                ).toFixed(0)}%))`}
                              </div>
                            )}
                          </span>
                        ))}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          <div className="mt-4">
            <button
              className="btn btn-outline-secondary me-2"
              onClick={() => setSelectedIndex((prev) => Math.max(0, prev - 1))}
              disabled={selectedIndex === 0}
            >
              ⬅ Câu trước
            </button>
            <button
              className="btn btn-outline-secondary"
              onClick={() =>
                setSelectedIndex((prev) =>
                  Math.min(sampleData.length - 1, prev + 1)
                )
              }
              disabled={selectedIndex === sampleData.length - 1}
            >
              Câu sau ➡
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default PronunciationPage;
