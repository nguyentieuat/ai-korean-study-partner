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
  const [evaluationByIndex, setEvaluationByIndex] = useState({});

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
      const body = JSON.stringify({ event_id, ...payload });
      const res = await fetch(TRACK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      // không chặn UI nếu lỗi
      return await res.json().catch(() => ({ ok: false }));
    } catch {
      return { ok: false };
    }
  }

  // Trích các phoneme điểm thấp từ kết quả chấm (threshold 0.8)
  function extractProblemPhones(result, threshold = 0.8) {
    const out = new Set();
    if (Array.isArray(result?.details_collapsed)) {
      for (const syl of result.details_collapsed) {
        const ph = syl?.phonemes || [];
        const sc = syl?.scores || [];
        for (let i = 0; i < Math.min(ph.length, sc.length); i++) {
          const s = Number(sc[i] ?? 0);
          if (s < threshold && ph[i]) out.add(ph[i]);
        }
      }
    } else if (Array.isArray(result?.details)) {
      for (const d of result.details) {
        const s = Number(d?.score ?? d?.conf ?? 0);
        const lab = d?.phoneme_surface || d?.phoneme;
        if (lab && s < threshold) out.add(lab);
      }
    }
    return Array.from(out);
  }

  // Gom lời khuyên/tip (nếu backend trả về)
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

      // cảnh báo nếu 1s chưa có âm thanh
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

      try {
        const res = await fetch(`${backendUrl}/pronunciation/evaluate`, {
          method: "POST",
          body: formData,
        });
        const result = await res.json();
        console.log("Kết quả chấm điểm:", result);

        // Lưu audio & evaluation theo index
        setUserAudioByIndex((prev) => ({ ...prev, [selectedIndex]: audioUrl }));
        setEvaluationByIndex((prev) => ({ ...prev, [selectedIndex]: result }));

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

  // --- Hangeul utils & constants ---
  const CHOSEONG = 0x1100,
    JUNGSEONG = 0x1161,
    JONGSEONG = 0x11a7;
  const SBase = 0xac00,
    LCount = 19,
    VCount = 21,
    TCount = 28,
    NCount = VCount * TCount;

  function isHangulSyllable(ch) {
    const code = ch.codePointAt(0);
    return code >= SBase && code <= 0xd7a3;
  }
  function splitHangulSyllable(syllable) {
    const code = syllable.codePointAt(0);
    if (!isHangulSyllable(syllable)) return [syllable];
    const SIndex = code - SBase;
    const L = Math.floor(SIndex / NCount);
    const V = Math.floor((SIndex % NCount) / TCount);
    const T = SIndex % TCount;
    const chars = [
      String.fromCharCode(CHOSEONG + L),
      String.fromCharCode(JUNGSEONG + V),
    ];
    if (T !== 0) chars.push(String.fromCharCode(JONGSEONG + T));
    return chars; // [L, V, (T?)]
  }

  const SILENT_ONSET = "\u110B"; // ᄋ
  const COMPLEX_VOWELS = new Set([
    "\u1163",
    "\u1164",
    "\u1167",
    "\u1168",
    "\u116D",
    "\u1172",
    "\u116A",
    "\u116B",
    "\u116C",
    "\u116F",
    "\u1170",
    "\u1171",
    "\u1174",
  ]);
  const PUNCTS = "?.!,;:~…—-()[]{}";

  // Bảng tách 받침 kép (Jongseong -> [trái, phải])
  const JONG_SPLIT = {
    "\u11AA": ["\u11A8", "\u11BA"], // ᆪ = ㄱ + ㅅ
    "\u11AC": ["\u11AB", "\u11BD"], // ᆬ = ㄴ + ㅈ
    "\u11AD": ["\u11AB", "\u11C2"], // ᆭ = ㄴ + ㅎ
    "\u11B0": ["\u11AF", "\u11A8"], // ᆰ = ㄹ + ㄱ
    "\u11B1": ["\u11AF", "\u11B7"], // ᆱ = ㄹ + ㅁ
    "\u11B2": ["\u11AF", "\u11B8"], // ᆲ = ㄹ + ㅂ
    "\u11B3": ["\u11AF", "\u11BA"], // ᆳ = ㄹ + ㅅ
    "\u11B4": ["\u11AF", "\u11C0"], // ᆴ = ㄹ + ㅌ
    "\u11B5": ["\u11AF", "\u11C1"], // ᆵ = ㄹ + ㅍ
    "\u11B6": ["\u11AF", "\u11C2"], // ᆶ = ㄹ + ㅎ
    "\u11B9": ["\u11B8", "\u11BA"], // ᆹ = ㅂ + ㅅ
  };

  // ㅎ ở phần phải (ᆭ, ᆶ) gặp ᄋ phía sau → KHÔNG đẩy
  function rightPartMoves(rightJong) {
    return rightJong !== "\u11C2"; // ᇂ
  }

  /** Lập kế hoạch resyllabify: đẩy đúng phần phải sang âm tiết sau nếu sau là ᄋ */
  function buildResyllabifiedPlan(text) {
    const sylls = []; // [{L,V,T, nonHangul, hangulLetters, moveRight:null|{rightJong, toSylIdx, whole,leftJong}}]
    const order = [];
    for (const ch of text) {
      if (ch === " " || PUNCTS.includes(ch)) {
        order.push({ type: "space" });
        continue;
      }
      const letters = splitHangulSyllable(ch);
      if (letters.length === 1) {
        const idx = sylls.length;
        sylls.push({
          L: null,
          V: null,
          T: null,
          nonHangul: true,
          hangulLetters: [{ hangul: ch }],
          moveRight: null,
        });
        order.push({ type: "syl", idx });
      } else {
        const [L, V, T] = letters;
        const idx = sylls.length;
        const hangulLetters = [{ hangul: L }, { hangul: V }];
        if (T) hangulLetters.push({ hangul: T });
        sylls.push({
          L,
          V,
          T,
          nonHangul: false,
          hangulLetters,
          moveRight: null,
        });
        order.push({ type: "syl", idx });
      }
    }

    // Xác định phần đẩy
    for (let i = 0; i < sylls.length - 1; i++) {
      const cur = sylls[i],
        nxt = sylls[i + 1];
      if (cur.nonHangul || nxt.nonHangul) continue;
      if (!cur.T) continue;
      if (nxt.L !== SILENT_ONSET) continue;

      // T đơn: đẩy hết
      if (!JONG_SPLIT[cur.hangulLetters[2]?.hangul]) {
        cur.moveRight = {
          rightJong: cur.hangulLetters[2]?.hangul,
          toSylIdx: i + 1,
          whole: true,
        };
        continue;
      }
      // T kép: đẩy phần phải (trừ ㅎ)
      const [left, right] = JONG_SPLIT[cur.hangulLetters[2].hangul];
      if (rightPartMoves(right)) {
        cur.moveRight = {
          rightJong: right,
          toSylIdx: i + 1,
          leftJong: left,
          whole: false,
        };
      } else {
        cur.moveRight = null; // ㅎ rơi → không đẩy
      }
    }

    // Sinh tokens theo hiển thị (onset → preOnset → nucleus → coda còn lại)
    const tokens = []; // {sylIdx, char, slot}
    const preOnset = new Map(); // sylIdx -> tokens onset chèn trước nucleus

    // Chuẩn bị preOnset từ moveRight (neo vào nguyên âm âm tiết nhận)
    sylls.forEach((s, i) => {
      if (!s.moveRight) return;
      const { rightJong, toSylIdx } = s.moveRight;
      if (!preOnset.has(toSylIdx)) preOnset.set(toSylIdx, []);
      const hostV =
        sylls[toSylIdx]?.V ||
        sylls[toSylIdx]?.hangulLetters?.[1]?.hangul ||
        null;
      preOnset.get(toSylIdx).push({
        sylIdx: toSylIdx,
        char: hostV,
        slot: "onset",
        fromRightJong: rightJong,
      });
    });

    order.forEach((it) => {
      if (it.type === "space") return;
      const i = it.idx,
        s = sylls[i];

      if (s.nonHangul) {
        tokens.push({
          sylIdx: i,
          char: s.hangulLetters[0].hangul,
          slot: "nucleus",
        });
        return;
      }

      // onset (bỏ ᄋ)
      if (s.L && s.L !== SILENT_ONSET) {
        tokens.push({ sylIdx: i, char: s.L, slot: "onset" });
      }

      // chèn liaison onset (nếu có) TRƯỚC nucleus
      if (preOnset.has(i)) preOnset.get(i).forEach((tk) => tokens.push(tk));

      // === NUCLEUS ===
      const hasRealOnset =
        (s.L && s.L !== SILENT_ONSET) || preOnset.get(i)?.length > 0;

      if (s.V) {
        const isComplex = COMPLEX_VOWELS.has(s.V);
        const nucleusCount = isComplex && !hasRealOnset ? 2 : 1;
        for (let k = 0; k < nucleusCount; k++) {
          tokens.push({ sylIdx: i, char: s.V, slot: "nucleus" });
        }
      }

      // === CODA === (phần còn lại sau khi đẩy)
      if (s.T) {
        const Tchar = s.hangulLetters[2].hangul;
        if (!s.moveRight) {
          tokens.push({ sylIdx: i, char: Tchar, slot: "coda" });
        } else {
          if (s.moveRight.whole) {
            // đẩy hết → không còn coda
          } else {
            tokens.push({ sylIdx: i, char: Tchar, slot: "coda" });
          }
        }
      }
    });

    return { sylls, order, tokens };
  }

  /** mapForTooltip: dùng plan sau resyllabify để hiển thị tooltip đúng chỗ */
  function mapForTooltip(text, detailsOrCollapsed) {
    const dets = Array.isArray(detailsOrCollapsed) ? detailsOrCollapsed : [];
    const isCollapsed =
      dets.length > 0 &&
      Array.isArray(dets[0]?.phonemes) &&
      Array.isArray(dets[0]?.scores);

    const { sylls, order, tokens } = buildResyllabifiedPlan(text);
    const syllables = sylls.map((s) => ({
      hangulLetters: s.hangulLetters,
      phonemes: [],
    }));

    if (isCollapsed) {
      // ----- dữ liệu theo âm tiết (details_collapsed) -----
      const sylIdxSeq = order.filter((x) => x.type === "syl").map((x) => x.idx);
      const K = Math.min(dets.length, sylIdxSeq.length);

      for (let i = 0; i < K; i++) {
        const sylIdx = sylIdxSeq[i];
        const sylTokens = tokens.filter((t) => t.sylIdx === sylIdx);
        const phs = dets[i].phonemes || [];
        const scs = dets[i].scores || [];
        const n = Math.min(sylTokens.length, phs.length);

        for (let j = 0; j < n; j++) {
          const tk = sylTokens[j];
          const ph = phs[j];
          const sc = Number(scs[j] ?? 0);
          syllables[sylIdx].phonemes.push({
            phoneme: ph,
            phoneme_surface: ph,
            score: sc,
            slot: tk.slot,
            hangulChar: tk.char,
            position: tk.slot,
            color: sc >= 0.8 ? "green" : sc >= 0.6 ? "yellow" : "red",
          });
        }
      }
    } else {
      // ----- dữ liệu phẳng cũ (details) — fallback giữ nguyên hành vi -----
      let p = 0;
      for (const tk of tokens) {
        if (p >= dets.length) break;
        const d = dets[p++];
        const lab = d.phoneme_surface || d.phoneme || "?";
        syllables[tk.sylIdx].phonemes.push({
          ...d,
          phoneme: lab,
          slot: d.slot || d.position || tk.slot,
          hangulChar: d.hangulChar || tk.char,
        });
      }
      if (p < dets.length && syllables.length > 0) {
        const last = syllables.length - 1;
        while (p < dets.length) {
          const d = dets[p++],
            lab = d.phoneme_surface || d.phoneme || "?";
          syllables[last].phonemes.push({
            ...d,
            phoneme: lab,
            slot: d.slot || d.position || "nucleus",
            hangulChar: syllables[last]?.hangulLetters?.[1]?.hangul || null,
          });
        }
      }
    }

    const result = [];
    order.forEach((it) => {
      if (it.type === "space") {
        result.push({ hangulLetters: [], phonemes: [], avgScore: 1 });
      } else {
        const s = syllables[it.idx] || { hangulLetters: [], phonemes: [] };
        const phs = s.phonemes || [];
        const avg = phs.length
          ? phs.reduce((a, x) => a + Number(x?.score ?? 0), 0) / phs.length
          : 1;
        result.push({
          hangulLetters: s.hangulLetters,
          phonemes: phs,
          avgScore: avg,
        });
      }
    });
    return result;
  }

  /** Nhóm theo jamo trong âm tiết (để hiện lỗi đúng chỗ) */
  function groupPhonemesByJamo(hangulLetters, phonemes) {
    const groups = (hangulLetters || []).map((l) => ({
      jamo: l.hangul,
      items: [],
    }));
    (phonemes || []).forEach((p) => {
      const idx = groups.findIndex((g) => g.jamo === p.hangulChar);
      if (idx >= 0) groups[idx].items.push(p);
    });
    return groups;
  }

  /** mapForIssues: trả list lỗi (score<threshold) sau khi đẩy 받침 */
  function mapForIssues(text, detailsOrCollapsed, threshold = 0.8) {
    const dets = Array.isArray(detailsOrCollapsed) ? detailsOrCollapsed : [];
    const isCollapsed =
      dets.length > 0 &&
      Array.isArray(dets[0]?.phonemes) &&
      Array.isArray(dets[0]?.scores);

    // ----------- CASE 1: dùng details_collapsed -----------
    if (isCollapsed) {
      const { order, tokens } = buildResyllabifiedPlan(text);
      const sylIdxSeq = order.filter((x) => x.type === "syl").map((x) => x.idx);
      const K = Math.min(dets.length, sylIdxSeq.length);

      const groups = [];
      for (let i = 0; i < K; i++) {
        const sylIdx = sylIdxSeq[i];
        const sylTokens = tokens.filter((t) => t.sylIdx === sylIdx);

        const label = dets[i]?.label ?? "";
        const phs = dets[i]?.phonemes || [];
        const scs = dets[i]?.scores || [];
        const advLists = Array.isArray(dets[i]?.advice) ? dets[i].advice : [];
        const n = Math.min(sylTokens.length, phs.length);

        const getAdviceFor = (j) => {
          if (Array.isArray(advLists[j])) return advLists[j];
          if (
            Array.isArray(advLists) &&
            typeof advLists[0] === "string" &&
            phs.length === 1
          )
            return advLists;
          return [];
        };

        const avgScore =
          typeof dets[i]?.avg_score === "number"
            ? dets[i].avg_score
            : scs.length
            ? scs.reduce((a, b) => a + Number(b || 0), 0) / scs.length
            : 1;

        const items = [];
        for (let j = 0; j < n; j++) {
          const sc = Number(scs[j] ?? 0);
          if (sc < threshold) {
            const tk = sylTokens[j];
            items.push({
              key: `${i}:${j}`,
              phoneme: phs[j],
              score: sc,
              advice: getAdviceFor(j),
              position: tk.slot,
              hangulChar: tk.char,
              color: sc >= 0.8 ? "green" : sc >= 0.6 ? "yellow" : "red",
            });
          }
        }

        if (avgScore < threshold || items.length > 0) {
          groups.push({
            groupKey: `g${i}`,
            label,
            syllableIndex: sylIdx,
            avgScore,
            items,
          });
        }
      }
      return groups;
    }

    // Fallback
    const mapped = mapForTooltip(text, dets);
    const groups = [];
    mapped.forEach((syl, i) => {
      const phs = syl.phonemes || [];
      const items = [];
      phs.forEach((p, j) => {
        const sc = Number(p?.score ?? 0);
        if (sc < threshold) {
          items.push({
            key: `${i}:${j}`,
            phoneme: p?.phoneme || p?.phoneme_surface || "?",
            score: sc,
            advice: Array.isArray(p?.advice) ? p.advice : [],
            position: p?.slot ?? p?.position ?? null,
            hangulChar: p?.hangulChar ?? null,
            color: p?.color || undefined,
          });
        }
      });
      if (items.length > 0) {
        const label = (syl.hangulLetters || []).map((x) => x.hangul).join("");
        groups.push({
          groupKey: `g${i}`,
          label,
          syllableIndex: i,
          avgScore: syl.avgScore ?? 1,
          items,
        });
      }
    });
    return groups;
  }

  // 2) Gom lỗi (score<threshold)
  const ADVICE_THRESHOLD = 0.8;
  const lowGroups = React.useMemo(() => {
    const src =
      currentEvaluation?.details_collapsed ?? currentEvaluation?.details;
    if (!src || !current?.text) return [];
    return mapForIssues(current.text, src, ADVICE_THRESHOLD);
  }, [
    currentEvaluation?.details_collapsed,
    currentEvaluation?.details,
    current?.text,
  ]);

  // Khoá UI khi đang ghi
  const uiDisabled = recording;

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

          {currentEvaluation?.details && (
            <div className="mt-3 p-3 border rounded">
              <h6>📊 Kết quả đánh giá:</h6>

              <p>
                <strong>Điểm trung bình:</strong>{" "}
                {((currentEvaluation.avg_score ?? 0) * 100).toFixed(0)}%
              </p>

              {/* Dòng chữ có tô màu theo từng âm tiết */}
              <div
                style={{
                  fontSize: "1.5rem",
                  lineHeight: "2rem",
                  position: "relative",
                }}
              >
                {mapForTooltip(
                  current.text,
                  currentEvaluation?.details_collapsed
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
                            .map((p) => {
                              const pct = ((p.score ?? 0) * 100).toFixed(0);
                              const lab = p.phoneme_surface || p.phoneme || "?";
                              return `${lab}_${pct}%`;
                            })
                            .join(" ")}
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

                        {/* phoneme issues trong label */}
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
                                    {it.phoneme}
                                  </code>
                                  {it.position && (
                                    <span className="badge text-bg-light ms-2">
                                      {it.position}
                                    </span>
                                  )}
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
          )}

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
