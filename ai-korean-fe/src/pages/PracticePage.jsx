// src/pages/PracticePage.jsx
import React, { useState, useRef, useEffect } from "react";
import axios from "axios";

const PracticePage = () => {
  const [level, setLevel] = useState("topik1");
  const [type, setType] = useState(null);
  const [selectedQuestion, setSelectedQuestion] = useState(null);
  const [questionContent, setQuestionContent] = useState(null);
  const [loadingQuestion, setLoadingQuestion] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [feedback, setFeedback] = useState(null); // "like" | "dislike" | null

  const backendUrl = import.meta.env.VITE_API_URL;

  // ---- Audio states: play once -> auto 2 plays (gap 5s)
  const audioRef = useRef(null);
  const [audioLocked, setAudioLocked] = useState(false); // ẩn controls trong lượt phát đầu
  const [sequenceActive, setSequenceActive] = useState(false);
  const [replaysRemaining, setReplaysRemaining] = useState(0);
  const replayTimerRef = useRef(null);

  // ---- Countdown (starts on first play)
  const [remainingSec, setRemainingSec] = useState(null); // null until started
  const [timeUp, setTimeUp] = useState(false);
  const timerRef = useRef(null);

  // ---- Auto-scroll target (answers/dialog section)
  const answerSectionRef = useRef(null);

  // ---- Dislike feedback details
  const [showDislikePanel, setShowDislikePanel] = useState(false);
  const [reasonTags, setReasonTags] = useState([]);
  const [freeText, setFreeText] = useState("");

  const REASON_OPTIONS = [
    { code: "audio_bad", label: "Âm thanh kém/khó nghe" },
    { code: "off_topic", label: "Lệch chủ đề" },
    { code: "too_easy", label: "Quá dễ" },
    { code: "too_hard", label: "Quá khó" },
    { code: "typo", label: "Lỗi chính tả" },
    { code: "ambiguous", label: "Mơ hồ/khó hiểu" },
  ];

  // ---------- Helpers ----------
  const formatTime = (s) => {
    if (s == null) return "02:00";
    const mm = String(Math.floor(s / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  };

  const startCountdownOnce = () => {
    if (timerRef.current || remainingSec !== null) return;
    setRemainingSec(120);
    timerRef.current = setInterval(() => {
      setRemainingSec((prev) => {
        if (prev === null) return prev;
        if (prev <= 1) {
          clearInterval(timerRef.current);
          timerRef.current = null;
          setTimeUp(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const resetTimer = () => {
    stopTimer();
    setRemainingSec(null);
    setTimeUp(false);
  };

  const resetAllPlayback = () => {
    setAudioLocked(false);
    setSequenceActive(false);
    setReplaysRemaining(0);
    if (replayTimerRef.current) {
      clearTimeout(replayTimerRef.current);
      replayTimerRef.current = null;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const resetFeedbackUI = () => {
    setFeedback(null);
    setShowDislikePanel(false);
    setReasonTags([]);
    setFreeText("");
  };

  // Reset khi đổi câu hỏi hoặc audio
  useEffect(() => {
    resetAllPlayback();
    resetTimer();
    resetFeedbackUI();
    return () => {
      if (replayTimerRef.current) {
        clearTimeout(replayTimerRef.current);
        replayTimerRef.current = null;
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [questionContent?.question_audio, selectedQuestion]);

  // Khi hết giờ: dừng audio + dọn lịch replay, và mở lại controls để có thể play lại
  useEffect(() => {
    if (timeUp) {
      if (audioRef.current) audioRef.current.pause();
      if (replayTimerRef.current) {
        clearTimeout(replayTimerRef.current);
        replayTimerRef.current = null;
      }
      setSequenceActive(false);
      setReplaysRemaining(0);
      setAudioLocked(false); // cho phép play lại sau khi hết giờ
    }
  }, [timeUp]);

  // Auto-scroll khi HẾT GIỜ hoặc khi CHỌN ĐÁP ÁN
  useEffect(() => {
    if ((timeUp || selectedAnswer) && answerSectionRef.current) {
      answerSectionRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [timeUp, selectedAnswer]);

  const handleAudioPlay = () => {
    // Nếu đã chọn đáp án hoặc đã hết giờ -> cho phép nghe lại, KHÔNG khởi động countdown/lock
    if (selectedAnswer || timeUp) return;

    // Lần đầu: khóa controls, bắt đầu sequence và countdown
    if (!sequenceActive && !audioLocked) {
      setAudioLocked(true);      // khóa controls sau lần click đầu (không bấm lại)
      setSequenceActive(true);
      setReplaysRemaining(1);    // 2 lượt: lượt này + 1 replay
      startCountdownOnce();      // bắt đầu đếm ngược 120s
    }
  };

  const handleAudioEnded = () => {
    if (sequenceActive && replaysRemaining > 0 && audioRef.current) {
      // chờ 5s rồi phát lại
      replayTimerRef.current = setTimeout(() => {
        if (!audioRef.current) return;
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
        setReplaysRemaining((n) => Math.max(0, n - 1));
      }, 5000);
    } else {
      // chuỗi kết thúc (vẫn giữ audioLocked = true cho lượt đầu)
      setSequenceActive(false);
      setReplaysRemaining(0);
      // mở controls chỉ khi timeUp hoặc đã chọn đáp án
    }
  };

  // ---------- Question list ----------
  const getQuestions = () => {
    let start = 1;
    let count = 0;

    if (level === "topik1") {
      if (type === "listening") {
        start = 1;
        count = 30;
      } else if (type === "reading") {
        start = 31;
        count = 40; // 40..70 => 40 câu
      }
    } else if (level === "topik2") {
      if (type === "listening" || type === "reading") {
        start = 1;
        count = 50;
      }
    }

    return Array.from({ length: count }, (_, i) => start + i);
  };

  const questions = type ? getQuestions() : [];

  // ---------- API ----------
  const fetchQuestion = async (q) => {
    // Kiểm tra câu chưa có dữ liệu
    const isUnderDevelopment =
      (level === "topik1" && type === "reading") ||
      level === "topik2" ||
      (level === "topik1" && type === "listening" && q > 4);

    if (isUnderDevelopment) {
      setQuestionContent({
        title: "Thông báo",
        message: "Đang phát triển, vui lòng quay lại sau",
      });
      return;
    }

    setLoadingQuestion(true);
    setSelectedQuestion(q);
    setSelectedAnswer(null);
    resetFeedbackUI();
    resetAllPlayback();
    resetTimer();

    const categoryParam = type === "listening" ? "listening" : "reading";

    try {
      const res = await axios.post(`${backendUrl}/api/generate_question`, {
        level: level,
        category: categoryParam,
        cau: parseInt(q.toString(), 10),
      });
      setQuestionContent(res.data);
    } catch (err) {
      console.error("Error fetching question:", err);
      setQuestionContent({ error: "Không thể lấy câu hỏi" });
    } finally {
      setLoadingQuestion(false);
    }
  };

  const buildFeedbackPayload = (reaction, extra = {}) => {
    // snapshot câu hỏi để bạn có thể phục hồi/đánh giá về sau
    const snapshot = {
      id: questionContent?.id ?? null,
      title: questionContent?.title ?? null,
      question: questionContent?.question ?? null,
      dialog: questionContent?.dialog ?? [],
      choices: questionContent?.choices ?? {},
      answer: questionContent?.answer ?? null,
      score: questionContent?.score ?? null,
      level,
      type,
      cau: selectedQuestion,
    };

    return {
      level,
      type,
      cau: selectedQuestion,
      questionId: questionContent?.id ?? null,
      reaction,                                // "like" | "dislike"
      reason_tags: extra.reason_tags ?? null,  // array hoặc null
      free_text: extra.free_text ?? null,
      answer_selected: selectedAnswer ?? null,
      is_correct:
        selectedAnswer != null && questionContent?.answer != null
          ? selectedAnswer === questionContent.answer
          : null,
      timer_seconds_left: remainingSec,        // có thể null nếu chưa bắt đầu
      time_limit: 120,
      question_snapshot: snapshot,
    };
  };

  const sendFeedbackLike = async () => {
    try {
      setFeedback("like");
      setShowDislikePanel(false);
      const payload = buildFeedbackPayload("like");
      await axios.post(`${backendUrl}/api/feedback`, payload);
    } catch (e) {
      console.error("Feedback error:", e);
    }
  };

  const submitDislike = async () => {
    try {
      setFeedback("dislike");
      const payload = buildFeedbackPayload("dislike", {
        reason_tags: reasonTags.length ? reasonTags : null,
        free_text: freeText?.trim() ? freeText.trim() : null,
      });
      await axios.post(`${backendUrl}/api/feedback`, payload);
      setShowDislikePanel(false);
    } catch (e) {
      console.error("Feedback error:", e);
    }
  };

  const toggleReason = (code) => {
    setReasonTags((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]
    );
  };

  // ---------- Choice rendering ----------
  const disableChoices = !!selectedAnswer || timeUp;
  const choiceVariant = (key) => {
    const isCorrect = questionContent?.answer === key;

    if (selectedAnswer) {
      const isSelected = selectedAnswer === key;
      return isSelected ? (isCorrect ? "btn-success" : "btn-danger") : (isCorrect ? "btn-success" : "btn-outline-secondary");
    }

    if (timeUp) {
      // Hết giờ: highlight đáp án đúng
      return isCorrect ? "btn-success" : "btn-outline-secondary";
    }

    return "btn-outline-secondary";
  };

  const showDialog = !!selectedAnswer || timeUp;

  // Khi người dùng chọn đáp án: dừng timer, mở controls để có thể nghe lại, dừng chuỗi replay
  const onChooseAnswer = (key) => {
    if (disableChoices) return;
    setSelectedAnswer(key);
    // dừng countdown & playback
    stopTimer();
    if (replayTimerRef.current) {
      clearTimeout(replayTimerRef.current);
      replayTimerRef.current = null;
    }
    setSequenceActive(false);
    setReplaysRemaining(0);
    setAudioLocked(false); // cho phép play lại
    if (audioRef.current) {
      audioRef.current.pause();
    }
  };

  return (
    <div className="container py-5">
      <h1 className="text-center mb-4">Practice TOPIK</h1>

      {/* Tabs TOPIK */}
      <div className="d-flex justify-content-center mb-4 gap-2">
        {["topik1", "topik2"].map((lvl) => (
          <button
            key={lvl}
            className={`btn btn-lg ${level === lvl ? "btn-primary shadow" : "btn-outline-secondary"}`}
            onClick={() => {
              setLevel(lvl);
              setType(null);
              setSelectedQuestion(null);
              setQuestionContent(null);
              setSelectedAnswer(null);
              resetFeedbackUI();
              resetAllPlayback();
              resetTimer();
            }}
          >
            {lvl.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Chọn dạng đề */}
      <div className="d-flex justify-content-center mb-4 gap-2">
        {["reading", "listening"].map((t) => (
          <button
            key={t}
            className={`btn btn-md ${type === t ? "btn-success shadow" : "btn-outline-secondary"}`}
            onClick={() => {
              setType(t);
              setSelectedQuestion(null);
              setQuestionContent(null);
              setSelectedAnswer(null);
              resetFeedbackUI();
              resetAllPlayback();
              resetTimer();
            }}
          >
            {t === "reading" ? "Read" : "Listen"}
          </button>
        ))}
      </div>

      {/* Chọn câu hỏi */}
      {type && questions.length > 0 && (
        <div className="mb-4">
          <h5 className="text-center mb-2">Chọn câu hỏi ({questions.length} câu)</h5>
          <div
            className="d-flex flex-wrap justify-content-center gap-2 border rounded p-2 bg-light"
            style={{ maxHeight: "400px", overflowY: "auto" }}
          >
            {questions.map((q) => (
              <button
                key={q}
                className={`btn btn-sm ${selectedQuestion === q ? "btn-primary" : "btn-outline-secondary"}`}
                style={{ width: "40px", height: "40px", position: "relative" }}
                disabled={loadingQuestion}
                onClick={() => fetchQuestion(q)}
              >
                {q}
                {loadingQuestion && selectedQuestion === q && (
                  <span
                    className="spinner-border spinner-border-sm position-absolute top-50 start-50 translate-middle"
                    role="status"
                    aria-hidden="true"
                  ></span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Hiển thị thông báo câu đang phát triển */}
      {questionContent?.message ? (
        <div className="card shadow p-4 mt-4 mx-auto text-center" style={{ maxWidth: "600px" }}>
          <h5 className="card-title mb-3">{questionContent.title}</h5>
          <p className="text-warning fst-italic">{questionContent.message}</p>
        </div>
      ) : (
        selectedQuestion &&
        questionContent &&
        !questionContent.error && (
          <div className="card shadow p-4 mt-4 mx-auto" style={{ maxWidth: "600px" }}>
            {/* Title row with countdown at right */}
            <div className="d-flex align-items-center justify-content-center position-relative mb-3">
              <h5 className="card-title text-center m-0">
                {level.toUpperCase()} - {type === "reading" ? "reading" : "listening"} - Câu {selectedQuestion}
              </h5>
              {type === "listening" && remainingSec !== null && (
                <span
                  className="badge bg-dark position-absolute end-0"
                  title="Thời gian còn lại"
                  style={{ fontSize: "0.9rem" }}
                >
                  {formatTime(remainingSec)}
                </span>
              )}
            </div>

            {/* Title + feedback */}
            <div className="d-flex align-items-start justify-content-between">
              <p className="fw-bold mb-2 me-3">{questionContent.title}</p>
              <div className="position-relative">
                <div className="btn-group" role="group" aria-label="Feedback">
                  <button
                    type="button"
                    className={`btn btn-sm ${feedback === "like" ? "btn-success" : "btn-outline-success"}`}
                    onClick={sendFeedbackLike}
                    disabled={!selectedQuestion}
                    title="Hữu ích"
                  >
                    👍
                  </button>
                  <button
                    type="button"
                    className={`btn btn-sm ${feedback === "dislike" ? "btn-danger" : "btn-outline-danger"}`}
                    onClick={() => setShowDislikePanel((v) => !v)}
                    disabled={!selectedQuestion}
                    title="Chưa ổn"
                  >
                    👎
                  </button>
                </div>

                {/* Dislike panel */}
                {showDislikePanel && (
                  <div
                    className="card shadow-sm p-2 mt-2"
                    style={{
                      position: "absolute",
                      right: 0,
                      zIndex: 20,
                      width: "280px",
                    }}
                  >
                    <div className="small fw-semibold mb-2">Vì sao bạn không thích?</div>
                    <div className="d-flex flex-column gap-1 mb-2">
                      {REASON_OPTIONS.map((opt) => (
                        <label key={opt.code} className="form-check d-flex align-items-center gap-2">
                          <input
                            type="checkbox"
                            className="form-check-input"
                            checked={reasonTags.includes(opt.code)}
                            onChange={() => toggleReason(opt.code)}
                          />
                          <span className="form-check-label">{opt.label}</span>
                        </label>
                      ))}
                    </div>
                    <div className="mb-2">
                      <textarea
                        className="form-control form-control-sm"
                        rows={2}
                        placeholder="Phản ánh thêm (tuỳ chọn)"
                        value={freeText}
                        onChange={(e) => setFreeText(e.target.value)}
                      />
                    </div>
                    <div className="d-flex justify-content-end gap-2">
                      <button
                        className="btn btn-sm btn-light"
                        onClick={() => {
                          setShowDislikePanel(false);
                          // không reset reason để user mở lại vẫn còn
                        }}
                      >
                        Đóng
                      </button>
                      <button className="btn btn-sm btn-danger" onClick={submitDislike}>
                        Gửi
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Nội dung câu hỏi (nếu có) */}
            <p>
              {questionContent?.cau ? `${questionContent.cau}. ` : ""}
              {questionContent?.question ?? ""}
            </p>

            {/* Audio: lần đầu bấm -> auto 2 lượt (gap 5s), khóa controls;
                Sau khi hết giờ hoặc đã chọn đáp án -> controls mở, có thể play lại */}
            {questionContent?.question_audio && (
              <audio
                ref={audioRef}
                controls={!audioLocked || timeUp || !!selectedAnswer}
                className="my-2 w-100"
                src={questionContent.question_audio} // data URI
                onPlay={handleAudioPlay}
                onEnded={handleAudioEnded}
                preload="auto"
              />
            )}

            {/* Phần đáp án / dialog */}
            <div ref={answerSectionRef}>
              {/* Dialog: hiển thị khi đã chọn đáp án HOẶC hết giờ */}
              {(!!selectedAnswer || timeUp) &&
                Array.isArray(questionContent?.dialog) &&
                questionContent.dialog.length > 0 && (
                  <div className="mb-2 p-2 bg-light border rounded">
                    {questionContent.dialog.map((d, idx) => (
                      <p key={idx} className="mb-1">
                        <strong>{d.speaker}:</strong> {d.text}
                      </p>
                    ))}
                  </div>
                )}

              {/* Choices */}
              {selectedQuestion && questionContent?.choices && (
                <div className="d-flex flex-column gap-2 mt-3">
                  {Object.entries(questionContent.choices).map(([key, text]) => (
                    <button
                      key={key}
                      className={`btn ${choiceVariant(key)}`}
                      onClick={() => onChooseAnswer(key)}
                      disabled={disableChoices}
                    >
                      {key}. {text}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Nút thử lại */}
            {(selectedAnswer || timeUp) && (
              <div className="d-flex justify-content-center mt-3">
                <button
                  className="btn btn-outline-primary"
                  onClick={() => fetchQuestion(selectedQuestion)}
                >
                  <span className="loop-icon me-2" aria-hidden="true">↻</span>
                  Thử lại câu hỏi
                </button>
              </div>
            )}
          </div>
        )
      )}

      {/* Loading spinner overlay */}
      {loadingQuestion && (
        <div
          className="position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center"
          style={{ background: "rgba(0,0,0,0.2)", zIndex: 1000 }}
        >
          <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default PracticePage;
