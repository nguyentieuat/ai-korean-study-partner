// src/pages/PracticePage.jsx
import React, { useState } from "react";
import axios from "axios";

const PracticePage = () => {
  const [level, setLevel] = useState("topik1");
  const [type, setType] = useState(null);
  const [selectedQuestion, setSelectedQuestion] = useState(null);
  const [questionContent, setQuestionContent] = useState(null);
  const [loadingQuestion, setLoadingQuestion] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const backendUrl = import.meta.env.VITE_API_URL;

  // Hàm trả về mảng câu hỏi dựa trên level và type
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

  // Call API để lấy câu hỏi
  const fetchQuestion = async (q) => {
     // Kiểm tra câu chưa có dữ liệu
    const isUnderDevelopment =
      (level === "topik1" && type === "reading") ||
      level === "topik2" ||
      (level === "topik1" && type === "listening" && q >= 8);

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
   
    const categoryParam = type === "listening" ? "Nghe" : "Đọc";

    try {
      const res = await axios.post(`${backendUrl}/api/generate_question`, {
        level: level,
        category: categoryParam,
        cau: q.toString(),
      });
      console.log("Fetched question:", res.data);
      setQuestionContent(res.data);
    } catch (err) {
      console.error("Error fetching question:", err);
      setQuestionContent({ error: "Không thể lấy câu hỏi" });
    } finally {
      setLoadingQuestion(false);
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
            className={`btn btn-lg ${
              level === lvl ? "btn-primary shadow" : "btn-outline-secondary"
            }`}
            onClick={() => {
              setLevel(lvl);
              setType(null);
              setSelectedQuestion(null);
              setQuestionContent(null);
              setSelectedAnswer(null);
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
            className={`btn btn-md ${
              type === t ? "btn-success shadow" : "btn-outline-secondary"
            }`}
            onClick={() => {
              setType(t);
              setSelectedQuestion(null);
              setQuestionContent(null);
              setSelectedAnswer(null);
            }}
          >
            {t === "reading" ? "Đọc" : "Nghe"}
          </button>
        ))}
      </div>

      {/* Chọn câu hỏi */}
      {type && questions.length > 0 && (
        <div className="mb-4">
          <h5 className="text-center mb-2">
            Chọn câu hỏi ({questions.length} câu)
          </h5>
          <div
            className="d-flex flex-wrap justify-content-center gap-2 border rounded p-2 bg-light"
            style={{ maxHeight: "400px", overflowY: "auto" }}
          >
            {questions.map((q) => (
              <button
                key={q}
                className={`btn btn-sm ${
                  selectedQuestion === q
                    ? "btn-primary"
                    : "btn-outline-secondary"
                }`}
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
        <div
          className="card shadow p-4 mt-4 mx-auto text-center"
          style={{ maxWidth: "600px" }}
        >
          <h5 className="card-title mb-3">{questionContent.title}</h5>
          <p className="text-warning fst-italic">{questionContent.message}</p>
        </div>
      ) : (
        /* Hiển thị câu hỏi bình thường */
        selectedQuestion &&
        questionContent &&
        !questionContent.error && (
          <div
            className="card shadow p-4 mt-4 mx-auto"
            style={{ maxWidth: "600px" }}
          >
            <h5 className="card-title text-center mb-3">
              {level.toUpperCase()} - {type === "reading" ? "Đọc" : "Nghe"} -
              Câu {selectedQuestion}
            </h5>

            <p className="fw-bold">{questionContent.title}</p>
            {selectedAnswer && (
              <p>{questionContent?.question?.question || ""}</p>
            )}

            {questionContent.question_audio_url && (
              <audio
                controls
                className="my-2 w-100"
                src={`${backendUrl}${questionContent.question_audio_url}`}
              ></audio>
            )}

            {questionContent.dialogue &&
              questionContent.dialogue.length > 0 && (
                <div className="mb-2 p-2 bg-light border rounded">
                  {questionContent.dialogue.map((d, idx) => (
                    <p key={idx}>
                      <strong>{d.speaker}:</strong> {d.text}
                    </p>
                  ))}
                </div>
              )}

            {selectedQuestion && questionContent?.question?.options && (
              <div className="d-flex flex-column gap-2 mt-3">
                {Object.entries(questionContent.question.options).map(
                  ([key, text]) => {
                    const isSelected = selectedAnswer === key;
                    const isCorrect = questionContent.question.answer === key;
                    let bgColor = "btn-outline-secondary";

                    if (selectedAnswer) {
                      bgColor = isSelected
                        ? isCorrect
                          ? "btn-success"
                          : "btn-danger"
                        : isCorrect
                        ? "btn-success"
                        : "btn-outline-secondary";
                    }

                    return (
                      <button
                        key={key}
                        className={`btn ${bgColor}`}
                        onClick={() =>
                          !selectedAnswer && setSelectedAnswer(key)
                        }
                        disabled={!!selectedAnswer}
                      >
                        {key}. {text}
                      </button>
                    );
                  }
                )}
              </div>
            )}

            {selectedAnswer && (
              <div className="d-flex justify-content-center mt-3">
                <button
                  className="btn btn-outline-primary"
                  onClick={() => fetchQuestion(selectedQuestion)}
                >
                  🔄 Thử lại câu hỏi
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
