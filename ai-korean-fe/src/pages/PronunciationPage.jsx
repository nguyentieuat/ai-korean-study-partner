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
  const [evaluating, setEvaluating] = useState(false);
  const [hoveredWordIndex, setHoveredWordIndex] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const waveformContainerRef = useRef(null);
  const waveSurferRef = useRef(null);

  // Phát hiện tiếng nói
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const rafIdRef = useRef(null);
  const soundDetectedRef = useRef(false);
  const silenceTimerRef = useRef(null);

  const VOLUME_THRESHOLD = 0.015; // ngưỡng RMS 0..1
  const NO_SOUND_MS = 3000; // 3s

  // Lưu audio & evaluation theo từng index
  const [userAudioByIndex, setUserAudioByIndex] = useState({});
  const [evaluationByIndex, setEvaluationByIndex] = useState({}); // { [index]: { ...response } }

  const [levelsMeta, setLevelsMeta] = useState([]);
  const [levelsLoading, setLevelsLoading] = useState(false);
  const [levelsError, setLevelsError] = useState(null);

  // === Tracking identities ===
  const [sessionId, setSessionId] = useState("");
  const [userIdHash, setUserIdHash] = useState("");
  // === Tracking endpoint ===
  const TRACK_URL = `${backendUrl}/track/event`;

  // Init userIdHash (ẩn danh) + sessionId để ghép với Google Form
  useEffect(() => {
    // userIdHash lưu localStorage
    let uid = localStorage.getItem("uid_hash");
    if (!uid) {
      uid = crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
      localStorage.setItem("uid_hash", uid);
    }
    setUserIdHash(uid);

    // sessionId mới cho mỗi phiên
    const day = new Date().toISOString().slice(0, 10).replace(/-/g, "");
    const rand = Math.random().toString(36).slice(2, 6).toUpperCase();
    setSessionId(`S-${day}-${rand}`);
  }, []);

  // Gửi sự kiện an toàn (idempotent bằng event_id random)
  async function postTrackEvent(payload) {
    try {
      const event_id =
        crypto?.randomUUID?.() || Math.random().toString(36).slice(2);
      const bodyObj = { event_id, ...payload };
      Object.keys(bodyObj).forEach(
        (k) => bodyObj[k] === undefined && delete bodyObj[k]
      );
      const body = JSON.stringify(bodyObj);
      const res = await fetch(TRACK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      return await res.json().catch(() => ({ ok: false }));
    } catch {
      return { ok: false };
    }
  }

  // Trích các phoneme điểm thấp từ kết quả chấm (threshold 0.8) — dùng cho tracking
  function extractProblemPhones(result, threshold = 0.8) {
    const out = new Set();
    if (Array.isArray(result?.details_collapsed)) {
      for (const syl of result.details_collapsed) {
        const ph = syl?.phonemes || [];
        const sc = (syl?.scores || []).map(Number);
        for (let i = 0; i < Math.min(ph.length, sc.length); i++) {
          if (sc[i] < threshold && ph[i]) out.add(ph[i]);
        }
      }
    }
    return Array.from(out);
  }

  // Gom lời khuyên/tip (nếu backend trả về) — dùng cho tracking
  function extractTipsShown(result) {
    const tips = new Set();
    if (Array.isArray(result?.notes)) {
      result.notes.forEach((n) => tips.add(String(n)));
    }
    const dc = result?.details_collapsed;
    if (Array.isArray(dc)) {
      dc.forEach((syl) => {
        if (Array.isArray(syl?.advice)) {
          syl.advice.forEach((a) => {
            if (Array.isArray(a)) a.forEach((x) => tips.add(String(x)));
            else if (typeof a === "string") tips.add(a);
          });
        }
      });
    }
    return Array.from(tips).slice(0, 10);
  }

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

  // ====== Helpers: volume monitor ======
  function startVolumeMonitor(stream) {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioCtx();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      src.connect(analyser);

      audioContextRef.current = ctx;
      analyserRef.current = analyser;
      soundDetectedRef.current = false;

      const data = new Uint8Array(analyser.fftSize);
      const tick = () => {
        analyser.getByteTimeDomainData(data);
        let sum = 0;
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] - 128) / 128; // -1..1
          sum += v * v;
        }
        const rms = Math.sqrt(sum / data.length); // 0..1
        if (rms > VOLUME_THRESHOLD) {
          soundDetectedRef.current = true;
        }
        rafIdRef.current = requestAnimationFrame(tick);
      };
      tick();

      // cảnh báo nếu 3s chưa có âm thanh
      silenceTimerRef.current = setTimeout(() => {
        if (!soundDetectedRef.current) {
          window.alert(
            "Mình chưa nghe thấy âm thanh nào. Hãy kiểm tra mic hoặc nói gần hơn nhé!"
          );
        }
      }, NO_SOUND_MS);
    } catch (e) {
      console.error("startVolumeMonitor error:", e);
    }
  }

  function stopVolumeMonitor() {
    try {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
      analyserRef.current = null;
    } catch {}
  }

  useEffect(() => {
    let ignore = false;
    (async () => {
      try {
        setLevelsLoading(true);
        setLevelsError(null);
        const res = await fetch(`${backendUrl}/pronunciation/levels`);
        if (!res.ok) throw new Error("Không tải được danh sách cấp độ");
        const data = await res.json();

        const list = Array.isArray(data?.levels) ? data.levels : [];
        list.sort((a, b) => (a.level ?? 0) - (b.level ?? 0));
        if (!ignore) {
          setLevelsMeta(list);
          if (!list.find((x) => x.level === level) && list.length > 0) {
            setLevel(list[0].level);
          }
        }
      } catch (e) {
        if (!ignore) setLevelsError(String(e?.message || e));
      } finally {
        if (!ignore) setLevelsLoading(false);
      }
    })();
    return () => {
      ignore = true;
    };
  }, []);

  // dọn dẹp khi unmount
  useEffect(() => {
    return () => {
      stopVolumeMonitor();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, []);

  // Load data theo level
  useEffect(() => {
    if (mode === "level") {
      setLoading(true);
      setError(null);
      setSelectedIndex(0);
      fetch(`${backendUrl}/pronunciation/level/${level}`)
        .then((res) => {
          if (!res.ok) throw new Error(`Lỗi tải dữ liệu cấp độ ${level}`);
          return res.json();
        })
        .then((data) => {
          setSampleData(data.items || []);
          setLoading(false);
        })
        .catch((err) => {
          setError(err.message);
          setSampleData([]);
          setLoading(false);
        });
    }
  }, [level, mode]);

  // --- thêm state/refs ở đầu component ---
  const [visitedOrder, setVisitedOrder] = useState([]); // các index đã xem theo thứ tự
  const [cursor, setCursor] = useState(-1); // vị trí hiện tại trong visitedOrder
  const unseenRef = useRef(new Set()); // tập index chưa xem

  // --- sau khi sampleData được load xong, khởi tạo random start ---
  useEffect(() => {
    if (sampleData.length > 0) {
      const all = new Set(sampleData.map((_, i) => i));
      const first = Math.floor(Math.random() * sampleData.length);
      all.delete(first);
      unseenRef.current = all;

      setVisitedOrder([first]);
      setCursor(0);
      setSelectedIndex(first);
    } else {
      unseenRef.current = new Set();
      setVisitedOrder([]);
      setCursor(-1);
      setSelectedIndex(0);
    }
  }, [sampleData]);

  // --- helper chọn random từ unseen ---
  const pickRandomFromUnseen = () => {
    const arr = Array.from(unseenRef.current);
    if (arr.length === 0) return null;
    const k = Math.floor(Math.random() * arr.length);
    return arr[k];
  };

  // --- nút Câu sau ---
  const goNext = () => {
    if (sampleData.length === 0) return;

    if (cursor < visitedOrder.length - 1) {
      const nextIdx = visitedOrder[cursor + 1];
      setCursor((c) => c + 1);
      setSelectedIndex(nextIdx);
      return;
    }

    let next = pickRandomFromUnseen();
    if (next === null) {
      const cur = visitedOrder[cursor];
      const all = new Set(sampleData.map((_, i) => i));
      all.delete(cur);
      unseenRef.current = all;
      next = pickRandomFromUnseen();
      if (next === null) return;
    }

    unseenRef.current.delete(next);
    setVisitedOrder((v) => [...v, next]);
    setCursor((c) => c + 1);
    setSelectedIndex(next);
  };

  // --- nút Câu trước ---
  const goPrev = () => {
    if (cursor <= 0) return;
    const prevIdx = visitedOrder[cursor - 1];
    setCursor((c) => c - 1);
    setSelectedIndex(prevIdx);
  };

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
    if (isPlaying) return;
    const src =
      current.audioBase64 ||
      current.audioDataUrl ||
      (current.audioUrl ? `${backendUrl}${current.audioUrl}` : null);
    if (!src) return;
    const audio = new Audio(src);
    setIsPlaying(true);
    audio.play();
    audio.onended = () => setIsPlaying(false);
    audio.onerror = () => setIsPlaying(false);
  };

  const handleStart = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      setRecording(true);
      audioChunksRef.current = [];
      mediaRecorderRef.current = new MediaRecorder(stream);
      mediaRecorderRef.current.ondataavailable = (e) =>
        audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.start();

      // bắt đầu đo âm lượng
      startVolumeMonitor(stream);
    } catch (err) {
      alert("Không thể truy cập microphone.");
      console.error(err);
    }
  };

  const handleStop = () => {
    setRecording(false);

    // dừng monitor + stream
    stopVolumeMonitor();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    mediaRecorderRef.current.stop();
    mediaRecorderRef.current.onstop = async () => {
      const hasSpeech = !!soundDetectedRef.current;

      // nếu không phát hiện tiếng → cảnh báo & không gửi
      if (!hasSpeech) {
        window.alert("Chưa phát hiện giọng nói. Vui lòng ghi âm lại.");
        return;
      }

      const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
      const audioUrl = URL.createObjectURL(blob);

      // Hiển thị waveform
      if (waveSurferRef.current) waveSurferRef.current.destroy();
      waveSurferRef.current = WaveSurfer.create({
        container: waveformContainerRef.current,
        waveColor: "violet",
        progressColor: "purple",
        height: 60,
      });
      waveSurferRef.current.load(audioUrl);

      setEvaluating(true);

      // Gửi audio lên server để chấm điểm
      const file = new File([blob], "recording.webm", { type: "audio/webm" });
      const formData = new FormData();
      formData.append("audio", file);
      formData.append("text", current.text || "");

      const tReqStartRef = performance.now();
      try {
        const res = await fetch(`${backendUrl}/pronunciation/evaluate`, {
          method: "POST",
          body: formData,
        });

        // Có thể status != 200 nhưng vẫn trả json message
        const result = await res.json().catch(() => ({
          ok: false,
          message: "Không phân tích được phản hồi.",
        }));

        console.log(result);

        // Lưu audio & evaluation theo index (kể cả khi chỉ có message)
        setUserAudioByIndex((prev) => ({ ...prev, [selectedIndex]: audioUrl }));
        setEvaluationByIndex((prev) => ({ ...prev, [selectedIndex]: result }));

        const tResponse = Math.round(performance.now() - tReqStartRef);

        // ======= TRACK SỰ KIỆN PRONUN =======
        const itemId =
          current.item_id ||
          current.id ||
          `PRON-lev${level}-${String(selectedIndex).padStart(4, "0")}`;

        const score_overall = Number(result?.avg_score ?? result?.score ?? 0);
        const cerNum = Number(result?.cer ?? result?.cer_rate ?? NaN);
        const problem_phones = extractProblemPhones(result);
        const tips_shown = extractTipsShown(result);

        await postTrackEvent({
          user_id_hash: userIdHash,
          session_id: sessionId,
          event_type: "pronun_scored",
          item_id: itemId,
          duration_ms: tResponse,
          score_overall,
          cer: Number.isFinite(cerNum) ? cerNum : undefined,
          meta: { problem_phones, tips_shown },
        });
      } catch (err) {
        console.error("Lỗi chấm điểm:", err);
        // vẫn lưu message để hiển thị
        setEvaluationByIndex((prev) => ({
          ...prev,
          [selectedIndex]: { ok: false, message: "Lỗi kết nối khi chấm điểm." },
        }));
      } finally {
        setEvaluating(false);
      }
    };
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

  /** ====== PHẦN RÚT GỌN DÙNG details_collapsed ====== */
  const ADVICE_THRESHOLD = 0.75;

  /** Lấy details_collapsed nếu có */
  function getCollapsed(evaluation) {
    return Array.isArray(evaluation?.details_collapsed)
      ? evaluation.details_collapsed
      : null;
  }

  /** Màu theo điểm */
  function scoreColor(s) {
    if (s >= 0.8) return "green";
    if (s >= 0.6) return "orange";
    return "red";
  }

  /**
   * Map text -> mảng ký tự, gắn từng ký tự Hangul với 1 item ở details_collapsed theo thứ tự.
   * Non-Hangul giữ nguyên, không tooltip.
   * Nếu thiếu/thừa phần tử, các ký tự ngoài phạm vi collapsed sẽ không có tooltip.
   */
  const to01 = (x) => {
    const n = Number(x);
    if (!Number.isFinite(n)) return 0;
    return n > 0 ? n / 100 : n;
  };

  // 0..1 hay 0..100 đều nhận; trả về % (0..100, int)
  const toPct = (x) => {
    const n = Number(x);
    if (!Number.isFinite(n)) return 0;
    return n > 1 ? Math.round(n) : Math.round(n * 100);
  };

  function mapForTooltipCollapsed(text, collapsed) {
    const chars = Array.from(text);
    const out = [];
    let k = 0; // index âm tiết trong collapsed

    for (let i = 0; i < chars.length; i++) {
      const ch = chars[i];
      const cp = ch.codePointAt(0);
      const isHangul = cp >= 0xac00 && cp <= 0xd7a3;

      if (isHangul && k < collapsed.length) {
        const syl = collapsed[k++];
        const avg = to01(syl?.score ?? syl?.avg_score); // 0..1
        const jamoArr = Array.isArray(syl?.jamo) ? syl.jamo : [];

        out.push({
          ch,
          isHangul: true,
          label: syl?.text ?? ch,
          // chuẩn hoá từng jamo
          jamo: jamoArr.map((j) => ({
            jamo: String(j?.jamo ?? ""),
            conf01: to01(j?.score),
            pct: toPct(j?.score),
          })),
          avg, // 0..1 cho tô màu
        });
      } else {
        out.push({
          ch,
          isHangul: false,
          label: ch,
          jamo: [],
          avg: 1,
        });
      }
    }
    return out;
  }

  /** Gom lỗi (score<threshold) trực tiếp từ details_collapsed */
  const lowGroups = React.useMemo(() => {
    const collapsed = getCollapsed(currentEvaluation);
    if (!Array.isArray(collapsed) || !current?.text) return [];

    const isRealSyllable = (syl) =>
      !!(syl?.text && String(syl.text).trim()) &&
      syl?.note !== "spillover_extra_ctc_chars";

    const jamoScore01 = (j) => {
      // hỗ trợ cả j.score (0..100) và j.conf (0..1)
      const v = j?.score ?? 0;
      return to01(v);
    };

    const sylAvg01 = (syl) => {
      const v = syl?.score ?? syl?.avg_score?? 0;
      return to01(v);
    };

    const result = [];
    collapsed.forEach((syl, i) => {
      if (!isRealSyllable(syl)) return;

      const jamoArr = Array.isArray(syl?.jamo) ? syl.jamo : [];
      const items = [];

      jamoArr.forEach((j, jidx) => {
        // bỏ jamo rỗng/placeholder
        const jstr = String(j?.jamo ?? "");
        if (!jstr.trim()) return;

        const s01 = jamoScore01(j);
        if (s01 < ADVICE_THRESHOLD) {
          const jamoAdvice = Array.isArray(j?.advice) ? j.advice : [];
          items.push({
            key: `${i}:${jidx}`,
            jamo: jstr,
            score: s01, // 0..1
            color: scoreColor(s01),
            advice: jamoAdvice,
          });
        }
      });

      const avg = sylAvg01(syl);
      if (items.length > 0 || avg < ADVICE_THRESHOLD) {
        result.push({
          groupKey: `g${i}`,
          label: syl?.text ?? "",
          syllableIndex: i,
          avgScore: avg, // 0..1
          items,
        });
      }
    });

    return result;
  }, [currentEvaluation?.details_collapsed, current?.text]);

  // Khoá UI khi đang ghi
  const uiDisabled = recording;

  // ====== UI ======
  return (
    <div className="container mt-4">
      <h2>
        Luyện phát âm{" "}
        {mode === "level" ? `- Cấp độ ${level}` : `- Chủ đề: ${topic || "..."}`}
      </h2>

      {/* Menu cấp độ / chủ đề */}
      <div className="mb-4 d-flex align-items-center flex-wrap gap-2">
        {levelsLoading && <span>Đang tải cấp độ…</span>}
        {levelsError && <span className="text-danger">Lỗi: {levelsError}</span>}

        {!levelsLoading && !levelsError && levelsMeta.length === 0 && (
          <span className="text-muted">Chưa có dữ liệu cấp độ</span>
        )}

        {!levelsLoading &&
          !levelsError &&
          levelsMeta.map((lvObj) => (
            <button
              key={lvObj.level}
              title={lvObj.focus ? `Focus: ${lvObj.focus}` : undefined}
              className={`btn me-2 ${
                mode === "level" && level === lvObj.level
                  ? "btn-primary"
                  : "btn-outline-primary"
              }`}
              onClick={() => handleChangeLevel(lvObj.level)}
              disabled={uiDisabled}
            >
              Cấp độ {lvObj.level}
            </button>
          ))}
      </div>

      {mode === "topic" && !topic && (
        <div className="mb-4">
          <h5>Chọn chủ đề:</h5>
          {Object.keys(dataByTopic).map((t) => (
            <button
              key={t}
              className="btn btn-outline-secondary me-2 mb-2"
              onClick={() => handleSelectTopic(t)}
              disabled={uiDisabled}
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
              disabled={uiDisabled || isPlaying}
            >
              🔊 Nghe mẫu
            </button>
            <button
              className={`btn ${recording ? "btn-danger" : "btn-warning"}`}
              onClick={recording ? handleStop : handleStart}
              disabled={
                uiDisabled ? false : loading || !current.text || evaluating
              }
            >
              {recording ? "⏹ Dừng ghi âm" : "🎙️ Ghi âm"}
            </button>
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

          {/* === KẾT QUẢ / THÔNG ĐIỆP TỪ BACKEND === */}
          {(() => {
            const collapsed = getCollapsed(currentEvaluation);
            const hasScores =
              Array.isArray(collapsed) &&
              collapsed.length > 0 &&
              Array.isArray(collapsed[0]?.jamo);

            // Nếu không có dữ liệu chấm → hiển thị message (nếu có)
            if (!hasScores) {
              const msg =
                currentEvaluation?.message || currentEvaluation?.error || null;
              return msg ? (
                <div className="mt-3 p-3 border rounded bg-light">
                  <h6>ℹ️ Thông báo</h6>
                  <p className="mb-0">{String(msg)}</p>
                </div>
              ) : null;
            }

            // Có dữ liệu chấm: hiển thị điểm + dòng chữ tô màu + issues
            const overallRaw =
              Number(
                currentEvaluation?.score ?? currentEvaluation?.avg_score ?? 0
              ) || 0;
            const overallPct = toPct(overallRaw);
            const mapped = mapForTooltipCollapsed(current.text, collapsed);

            return (
              <div className="mt-3 p-3 border rounded">
                <h6>📊 Kết quả đánh giá:</h6>
                <p>
                  <strong>Điểm trung bình:</strong> {overallPct}%
                </p>

                {/* Dòng chữ có tô màu theo từng âm tiết (simple by details_collapsed) */}
                <div
                  style={{
                    fontSize: "1.5rem",
                    lineHeight: "2rem",
                    position: "relative",
                  }}
                >
                  {mapped.map((m, idx) => {
                    const color = m.isHangul ? scoreColor(m.avg) : undefined;
                    const isHoverable = m.isHangul && m.jamo.length > 0;

                    return (
                      <span
                        key={idx}
                        style={{
                          color,
                          marginRight: "0.05rem",
                          position: "relative",
                          cursor: isHoverable ? "pointer" : "default",
                        }}
                        onMouseEnter={() =>
                          isHoverable && setHoveredWordIndex(idx)
                        }
                        onMouseLeave={() =>
                          isHoverable && setHoveredWordIndex(null)
                        }
                      >
                        {m.ch}
                        {hoveredWordIndex === idx && isHoverable && (
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
                            {m.jamo.map((j) => `${j.jamo}_${j.pct}%`).join(" ")}
                          </div>
                        )}
                      </span>
                    );
                  })}
                </div>

                {/* Notes */}
                {(currentEvaluation?.notes || current?.notes) && (
                  <div className="mt-3">
                    <h6>📝 Notes</h6>
                    {Array.isArray(currentEvaluation?.notes) ? (
                      <ul className="mt-1">
                        {currentEvaluation.notes.map((n, i) => (
                          <li key={i}>{n}</li>
                        ))}
                      </ul>
                    ) : (
                      <p className="mb-0">
                        {currentEvaluation?.notes ?? current?.notes}
                      </p>
                    )}
                  </div>
                )}

                {/* Danh sách âm/từ phát âm kém + advice */}
                <div className="mt-3">
                  <h6>⚠️ Âm/từ cần cải thiện</h6>
                  {lowGroups.length === 0 ? (
                    <div className="text-success">
                      Tuyệt vời! Không có âm/từ nào dưới 80%.
                    </div>
                  ) : (
                    <div className="d-flex flex-column gap-3">
                      {lowGroups.map((g) => (
                        <div key={g.groupKey} className="p-2 border rounded">
                          <div className="d-flex align-items-center justify-content-between">
                            <div>
                              <strong style={{ fontSize: "1.1rem" }}>
                                {g.label || "(âm tiết)"}
                              </strong>
                            </div>
                            <div
                              title="Điểm trung bình của âm tiết"
                              style={{ minWidth: 90, textAlign: "right" }}
                            >
                              {(g.avgScore * 100).toFixed(0)}%
                            </div>
                          </div>

                          <div className="mt-2 d-flex flex-column gap-2">
                            {g.items.map((it) => (
                              <div
                                key={it.key}
                                className="p-2 border rounded"
                                style={{ background: "#fafafa" }}
                              >
                                <div className="d-flex align-items-center justify-content-between">
                                  <div>
                                    <code style={{ fontSize: "0.95rem" }}>
                                      {it.jamo}
                                    </code>
                                  </div>
                                  <div
                                    style={{ minWidth: 70, textAlign: "right" }}
                                  >
                                    {(it.score * 100).toFixed(0)}%
                                  </div>
                                </div>
                                {Array.isArray(it.advice) &&
                                  it.advice.length > 0 && (
                                    <ul className="mt-2 mb-0 ps-3">
                                      {it.advice.map((a, i) => (
                                        <li key={i}>{a}</li>
                                      ))}
                                    </ul>
                                  )}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })()}

          <div className="mt-4">
            <button
              className="btn btn-outline-secondary me-2"
              onClick={goPrev}
              disabled={uiDisabled || cursor <= 0}
            >
              ⬅ Câu trước
            </button>
            <button
              className="btn btn-outline-secondary"
              onClick={goNext}
              disabled={uiDisabled || sampleData.length === 0}
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
