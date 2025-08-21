import React, { useState, useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
import useRecorderRTC from "@/hooks/useRecorderRTC";
const backendMainUrl = import.meta.env.VITE_API_MAIN_URL;
const backendMFAUrl = import.meta.env.VITE_API_MFA_URL;

const PronunciationPage = () => {
  const [mode, setMode] = useState("level");
  const [level, setLevel] = useState(1);
  const [topic, setTopic] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [sampleData, setSampleData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [hoveredWordIndex, setHoveredWordIndex] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const waveformContainerRef = useRef(null);
  const waveSurferRef = useRef(null);

  // Lưu audio và evaluation theo từng index
  const [userAudioByIndex, setUserAudioByIndex] = useState({});
  const [evaluationByIndex, setEvaluationByIndex] = useState({});
  const [rtcClient, setRtcClient] = useState(null);

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

  // Load data theo level
  useEffect(() => {
    if (mode === "level") {
      setLoading(true);
      setError(null);
      setSelectedIndex(0);
      fetch(`/api/pronunciation/${level}`)
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

  // Load data theo topic
  useEffect(() => {
    if (mode === "topic" && topic) {
      setSampleData(dataByTopic[topic] || []);
      setSelectedIndex(0);
    }
  }, [topic, mode]);

  const current = sampleData[selectedIndex] || {};
  const currentAudio = userAudioByIndex[selectedIndex] || null;
  const currentEvaluation = evaluationByIndex[selectedIndex] || null;

  const playSample = () => {
    if (!current.audioUrl || isPlaying) return;
    const audio = new Audio(`${backendMainUrl}${current.audioUrl}`);
    setIsPlaying(true);
    audio.play();
    audio.onended = () => setIsPlaying(false);
    audio.onerror = () => setIsPlaying(false);
  };

  const { recording, feedback, audioUrl, startRecording, stopRecording } =
    useRecorderRTC(backendMFAUrl);

  const handleStart = () => {
    startRecording(current.text); 
    setEvaluating(false);
  };

  const handleStop = () => {
    const url = stopRecording(); // url từ hook
    if (url) {
      setUserAudioByIndex((prev) => ({
        ...prev,
        [selectedIndex]: url, // lưu theo câu hiện tại
      }));
    }
    setEvaluating(true); // bật loading chấm điểm
  };

  const handleChangeLevel = (lv) => {
    setMode("level");
    setLevel(lv);
    setUserAudioByIndex({});
    setEvaluationByIndex({});
  };

  const handleSelectTopic = (t) => {
    setMode("topic");
    setTopic(t);
    setUserAudioByIndex({});
    setEvaluationByIndex({});
  };

  // Hàm tách syllable Hangeul thành phoneme
  function splitHangulSyllable(syllable) {
    const CHOSEONG = 0x1100,
      JUNGSEONG = 0x1161,
      JONGSEONG = 0x11a7;
    const SBase = 0xac00,
      LCount = 19,
      VCount = 21,
      TCount = 28,
      NCount = VCount * TCount;
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

  // Map syllable -> phoneme + avgScore, bỏ qua space
  function mapSyllableToPhonemeScoreWithHangul(text, phonemes) {
    const phonemesCopy = [...phonemes];
    const result = [];
    for (const syllable of text) {
      if (syllable === " ") {
        result.push({ hangulLetters: [], phonemes: [], avgScore: 1 });
        continue;
      }
      const letters = splitHangulSyllable(syllable);
      const syllablePhonemes = [];
      for (let i = 0; i < letters.length; i++) {
        if (phonemesCopy.length === 0) break;
        syllablePhonemes.push(phonemesCopy.shift());
      }
      const avgScore =
        syllablePhonemes.reduce((sum, p) => sum + (p.score || 0), 0) /
        (syllablePhonemes.length || 1);
      const hangulLetters = letters.map((l) => ({ hangul: l }));
      result.push({ hangulLetters, phonemes: syllablePhonemes, avgScore });
    }
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
        {/* <button
          className={`btn ${mode === "topic" ? "btn-primary" : "btn-outline-primary"}`}
          onClick={() => { setMode("topic"); setTopic(null); }}
        >
          Tùy chọn theo chủ đề
        </button> */}
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
            <pre>
              {feedback ? JSON.stringify(feedback, null, 2) : "No feedback yet"}
            </pre>
          </div>

          <div ref={waveformContainerRef} style={{ marginTop: 10 }} />

          {currentAudio && (
            <div className="mt-3">
              <h6>🔁 So sánh với bạn:</h6>
              <audio controls src={currentAudio}></audio>
            </div>
          )}

          {evaluating && (
            <div className="mt-3 p-3 border rounded text-center">
              <div className="spinner-border text-primary" role="status" />
              <p className="mt-2">Đang chấm điểm...</p>
            </div>
          )}

          {currentEvaluation?.detail && (
            <div className="mt-3 p-3 border rounded">
              <h6>📊 Kết quả đánh giá:</h6>
              <p>
                <strong>Điểm trung bình:</strong>{" "}
                {((currentEvaluation.score ?? 0) * 100).toFixed(0)}%
              </p>
              <div
                style={{
                  fontSize: "1.5rem",
                  lineHeight: "2rem",
                  position: "relative",
                }}
              >
                {mapSyllableToPhonemeScoreWithHangul(
                  current.text,
                  currentEvaluation.detail
                ).map((s, idx) => {
                  const { hangulLetters, phonemes, avgScore } = s;
                  let color =
                    avgScore >= 0.8
                      ? "green"
                      : avgScore >= 0.6
                      ? "orange"
                      : "red";
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
                          onMouseEnter={() => setHoveredWordIndex(idx)}
                          onMouseLeave={() => setHoveredWordIndex(null)}
                        >
                          {l.hangul}
                        </span>
                      ))}
                      {hoveredWordIndex === idx && phonemes.length > 0 && (
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
                          {phonemes
                            .map(
                              (p, i) =>
                                `${hangulLetters[i]?.hangul || "?"} (${
                                  p.phoneme || "?"
                                } (${((p.score ?? 0) * 100).toFixed(0)}%))`
                            )
                            .join(" ")}
                        </div>
                      )}
                    </span>
                  );
                })}
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
