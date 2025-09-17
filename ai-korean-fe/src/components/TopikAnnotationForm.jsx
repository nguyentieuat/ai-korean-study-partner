// src/components/TopikAnnotationForm.jsx
import React, { useState } from "react";
const backendUrl = import.meta.env.VITE_API_URL;

const typeMappingTopik1 = [
  // Nghe
  {
    display: "1~4번: Chọn câu trả lời phù hợp (Nghe)",
    jsonType: "Nghe_Almat_Dap",
  },
  {
    display: "5~6번: Chọn lời nói tiếp theo (Nghe)",
    jsonType: "Nghe_Loi_Tiep",
  },
  { display: "7~10번: Chọn địa điểm (Nghe)", jsonType: "Nghe_DiaDiem" },
  { display: "11~14번: Chọn chủ đề (Nghe)", jsonType: "Nghe_ChuDe" },
  { display: "15~16번: Chọn tranh (Nghe)", jsonType: "Nghe_Tranh" },
  {
    display: "17~21번: Chọn nội dung đúng (Nghe)",
    jsonType: "Nghe_NoidungDung",
  },
  {
    display: "22~24번: Chọn suy nghĩ trọng tâm (Nghe)",
    jsonType: "Nghe_TamTu",
  },
  { display: "25~30번: Trả lời câu hỏi (Nghe)", jsonType: "Nghe_TraLoi" },
  // Đọc
  { display: "31~33번: Chủ đề (Đọc)", jsonType: "Doc_ChuDe" },
  { display: "34~39번: Điền chỗ trống (Đọc)", jsonType: "Doc_DienTrong" },
  {
    display: "40~42번: Chọn nội dung không đúng (Đọc)",
    jsonType: "Doc_NoidungKhongDung",
  },
  { display: "43~45번: Chọn nội dung đúng (Đọc)", jsonType: "Doc_NoidungDung" },
  { display: "46~48번: Chọn ý chính (Đọc)", jsonType: "Doc_YChinh" },
  { display: "57~58번: Sắp xếp thứ tự (Đọc)", jsonType: "Doc_SapXep" },
  { display: "59~62번: Điền + chủ đề (Đọc)", jsonType: "Doc_Dien_ChuDe" },
  {
    display: "63~64번: Mục đích viết + nội dung (Đọc)",
    jsonType: "Doc_MucDich",
  },
  { display: "65~70번: Điền + suy luận (Đọc)", jsonType: "Doc_Dien_SuyLuan" },
];

const typeMappingTopik2 = [
  // NGHE
  {
    display: "1~3번: Nhìn hình, nghe và chọn đáp án đúng (Nghe)",
    jsonType: "Nghe_HinhAnh",
  },
  {
    display: "4~8번: Nghe và chọn câu trả lời tiếp theo (Nghe)",
    jsonType: "Nghe_LoiTiep",
  },
  {
    display: "9~12번: Nghe và chọn hành động tiếp theo (Nghe)",
    jsonType: "Nghe_HanhDong",
  },
  {
    display: "13~16번: Nghe và chọn nội dung giống (Nghe)",
    jsonType: "Nghe_GiongNoiDung",
  },
  {
    display: "17~21번: Nghe và chọn suy nghĩ nhân vật (Nghe)",
    jsonType: "Nghe_SuyNghi",
  },
  {
    display: "22번: Nghe và chọn nội dung giống (Nghe)",
    jsonType: "Nghe_GiongDung",
  },
  {
    display: "23번: Nghe và trả lời hành động nhân vật (Nghe)",
    jsonType: "Nghe_HanhDongNV",
  },
  {
    display: "24~25번: Nghe và chọn đáp án đúng (Nghe)",
    jsonType: "Nghe_NoidungDung",
  },
  {
    display: "26~27번: Nghe và chọn ý đồ nhân vật (Nghe)",
    jsonType: "Nghe_YDo",
  },
  {
    display: "28~30번: Nghe và chọn đáp án đúng (Nghe)",
    jsonType: "Nghe_Noidung",
  },
  {
    display: "31~32번: Nghe và chọn suy nghĩ/thái độ nhân vật (Nghe)",
    jsonType: "Nghe_ThaiDo",
  },
  {
    display: "33~36번: Nghe và chọn nội dung đúng nhất (Nghe)",
    jsonType: "Nghe_NoidungChinh",
  },
  {
    display: "37~38번: Nghe chương trình giáo dục (Nghe)",
    jsonType: "Nghe_ChuongTrinhGD",
  },
  {
    display: "39~40번: Nghe hội thoại và chọn đáp án đúng (Nghe)",
    jsonType: "Nghe_HoiThoai",
  },
  {
    display: "41~42번: Nghe diễn thuyết và chọn suy nghĩ đúng (Nghe)",
    jsonType: "Nghe_DienThuyet_SuyNghi",
  },
  {
    display: "43~44번: Nghe tư liệu và chọn đáp án đúng (Nghe)",
    jsonType: "Nghe_TuLieu",
  },
  {
    display: "45~46번: Nghe diễn thuyết và chọn đáp án đúng nhất (Nghe)",
    jsonType: "Nghe_DienThuyet",
  },
  {
    display: "47~48번: Nghe tọa đàm và chọn đáp án đúng nhất (Nghe)",
    jsonType: "Nghe_ToaDam",
  },
  {
    display: "49~50번: Nghe diễn thuyết và chọn đáp án đúng nhất (Nghe)",
    jsonType: "Nghe_DienThuyet2",
  },

  // ĐỌC
  { display: "1~4번: Ngữ pháp (Đọc)", jsonType: "Doc_NguPhap" },
  { display: "5~8번: Poster quảng cáo (Đọc)", jsonType: "Doc_Poster" },
  { display: "9~12번: Nội dung đoạn văn (Đọc)", jsonType: "Doc_NoidungDung" },
  { display: "13~15번: Sắp xếp thứ tự câu (Đọc)", jsonType: "Doc_SapXep" },
  { display: "16~22번: Điền vào chỗ trống (Đọc)", jsonType: "Doc_DienTrong" },
  {
    display: "23~27번: Gạch chân hoặc nội dung đoạn văn (Đọc)",
    jsonType: "Doc_GachChan",
  },
  { display: "28~31번: Điền chỗ trống (Đọc)", jsonType: "Doc_DienTrong2" },
  { display: "32~38번: Nội dung đoạn văn (Đọc)", jsonType: "Doc_Noidung2" },
  {
    display: "39~43번: Gạch chân hoặc điền đoạn văn (Đọc)",
    jsonType: "Doc_GachChan2",
  },
  { display: "44~45번: Chủ đề đoạn văn (Đọc)", jsonType: "Doc_ChuDe" },
  {
    display: "46~50번: Điền chỗ trống / mục đích đoạn văn (Đọc)",
    jsonType: "Doc_Dien_MucDich",
  },
];

const TopikAnnotationForm = ({ userInfo }) => {
  const [level, setLevel] = useState("1"); // TOPIK level
  const [type, setType] = useState("");
  const [title, setTitle] = useState("");
  const [question, setQuestion] = useState("");
  const [dialogue, setDialogue] = useState([{ speaker: "", text: "" }]);
  const [passage, setPassage] = useState("");
  const [options, setOptions] = useState({ A: "", B: "", C: "", D: "" });
  const [answer, setAnswer] = useState("");
  const [explanation, setExplanation] = useState("");
  const [dataset, setDataset] = useState([]);
  const [editIndex, setEditIndex] = useState(null);
  const [loading, setLoading] = useState(false);

  const currentTypeMapping =
    level === "1" ? typeMappingTopik1 : typeMappingTopik2;
  const isNghe = type.startsWith("Nghe_");

  const handleDialogueChange = (index, field, value) => {
    const newDialogue = [...dialogue];
    newDialogue[index][field] = value;
    setDialogue(newDialogue);
  };

  const addDialogueLine = () =>
    setDialogue([...dialogue, { speaker: "", text: "" }]);

  const handleAddOrUpdate = () => {
    const selectedType = currentTypeMapping.find((t) => t.jsonType === type);
    const section = type.startsWith("Nghe") ? "Nghe" : type.startsWith("Doc") ? "Đọc" : "";
    const item = {
      type,
      section: section,
      level,
      title,
      question,
      explanation,
      options: Object.values(options).some((v) => v) ? options : undefined,
      answer,
      ...(isNghe ? { dialogue } : { passage }),
    };

    if (editIndex !== null) {
      const newDataset = [...dataset];
      newDataset[editIndex] = item;
      setDataset(newDataset);
      setEditIndex(null);
    } else {
      setDataset([...dataset, item]);
    }

    // reset form
    setTitle("");
    setQuestion("");
    setDialogue([{ speaker: "", text: "" }]);
    setPassage("");
    setOptions({ A: "", B: "", C: "", D: "" });
    setAnswer("");
    setExplanation("");
  };

  const handleRowClick = (index) => {
    const item = dataset[index];
    setType(item.type);
    setTitle(item.title);
    setQuestion(item.question);
    setDialogue(item.dialogue || [{ speaker: "", text: "" }]);
    setPassage(item.passage || "");
    setOptions(item.options || { A: "", B: "", C: "", D: "" });
    setAnswer(item.answer || "");
    setExplanation(item.explanation || "");
    setLevel(item.level || "1");
    setEditIndex(index);

    const section = item.type.startsWith("Nghe")
    ? "Nghe"
    : item.type.startsWith("Doc")
    ? "Đọc"
    : "";
    item.section = section;
  };

  const handleSubmitAll = async () => {
    if (!userInfo.phone.trim()) {
      alert("Số điện thoại là bắt buộc!");
      return;
    }
    if (dataset.length === 0) {
      alert("Chưa có dữ liệu nào để gửi!");
      return;
    }

    setLoading(true);
    const payload = {
      annotator_name: userInfo.name,
      annotator_phone: userInfo.phone,
      annotations: dataset,
    };

    try {
      const response = await fetch(
        `${backendUrl}/cooperate_topik_annotator`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      if (!response.ok) throw new Error("Gửi thất bại!");

      const result = await response.json();
      console.log("✅ Server trả về:", result);
      alert("Gửi dữ liệu thành công!");

      // reset toàn bộ
      setTitle("");
      setQuestion("");
      setDialogue([{ speaker: "", text: "" }]);
      setPassage("");
      setOptions({ A: "", B: "", C: "", D: "" });
      setAnswer("");
      setExplanation("");
      setDataset([]);
    } catch (err) {
      console.error("❌ Lỗi khi gửi:", err);
      alert("Có lỗi khi gửi dữ liệu!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
      <h2>Topik Annotation Form</h2>
      <small style={{ color: "red" }}>
        ⚠️ Lưu ý: Vui lòng chỉ nhập nội dung bằng tiếng Hàn.
      </small>
      {/* chọn level */}
      <div style={{ display: "flex", flexDirection: "column" }}>
        <label>Chọn Level TOPIK:</label>
        <select
          value={level}
          onChange={(e) => {
            setLevel(e.target.value);
            setType(""); // reset type khi đổi level
          }}
          style={{ padding: 5, maxWidth: 200 }}
        >
          <option value="1">TOPIK I</option>
          <option value="2">TOPIK II</option>
        </select>
      </div>

      <div style={{ display: "flex", flexDirection: "column" }}>
        <label>Chọn loại câu hỏi:</label>
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          style={{ padding: 5, maxWidth: 400 }}
        >
          <option value="">-- Chọn --</option>
          {currentTypeMapping.map((t) => (
            <option key={t.jsonType} value={t.jsonType}>
              {t.display}
            </option>
          ))}
        </select>
      </div>

      <div style={{ display: "flex", flexDirection: "column" }}>
        <label>Title:</label>
        <small style={{ color: "#666", marginLeft: 5 }}>
          (Tên ngắn gọn cho câu hỏi)
        </small>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          style={{ padding: 5, width: "100%" }}
        />
      </div>

      {isNghe ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <label>Dialogue:</label>
          <small style={{ color: "#666", marginLeft: 5 }}>
            (Hội thoại: đối thoại A-B)
          </small>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 5,
              maxHeight: 200,
              overflowY: "auto",
            }}
          >
            {dialogue.map((d, i) => (
              <div key={i} style={{ display: "flex", gap: 5 }}>
                <input
                  placeholder="Speaker"
                  value={d.speaker}
                  onChange={(e) =>
                    handleDialogueChange(i, "speaker", e.target.value)
                  }
                  style={{ flex: 1, padding: 5 }}
                />
                <input
                  placeholder="Text"
                  value={d.text}
                  onChange={(e) =>
                    handleDialogueChange(i, "text", e.target.value)
                  }
                  style={{ flex: 3, padding: 5 }}
                />
              </div>
            ))}
          </div>
          <button onClick={addDialogueLine}>Thêm dòng dialogue</button>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column" }}>
          <label>Passage:</label>
          <small style={{ color: "#666", marginLeft: 5 }}>
            (Đoạn văn: nhập đoạn đọc hiểu)
          </small>
          <textarea
            value={passage}
            onChange={(e) => setPassage(e.target.value)}
            rows={5}
            style={{ width: "100%", padding: 5 }}
          />
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column" }}>
        <label>Question:</label>
        <small style={{ color: "#666", marginLeft: 5 }}>
          (Câu hỏi chính cần trả lời)
        </small>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{ padding: 5, width: "100%" }}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        <label>Options:</label>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          {["A", "B", "C", "D"].map((k) => (
            <input
              key={k}
              placeholder={k}
              value={options[k]}
              onChange={(e) => setOptions({ ...options, [k]: e.target.value })}
              style={{ flex: 1, minWidth: 80, padding: 5 }}
            />
          ))}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <label>Answer:</label>
        <small style={{ color: "#666", marginLeft: 5 }}>
          (nhập A/B/C/D hoặc câu trả lời)
        </small>
        <input
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          style={{ width: 60, padding: 5 }}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column" }}>
        <label>Explanation:</label>
        <small style={{ color: "#666", marginLeft: 5 }}>
          (Giải thích ngắn gọn tại sao đáp án đúng)
        </small>
        <textarea
          value={explanation}
          onChange={(e) => setExplanation(e.target.value)}
          rows={2}
          style={{ width: "100%", padding: 5 }}
        />
      </div>

      <div style={{ display: "flex", gap: 10 }}>
        <button onClick={handleAddOrUpdate}>
          {editIndex !== null ? "Cập nhật" : "Add"}
        </button>
        <button
          onClick={handleSubmitAll}
          disabled={loading}
          style={{ position: "relative" }}
        >
          {loading ? "Đang gửi..." : "Submit All"}
          {loading && (
            <span
              style={{
                marginLeft: 10,
                width: 16,
                height: 16,
                border: "2px solid #fff",
                borderTop: "2px solid transparent",
                borderRadius: "50%",
                display: "inline-block",
                animation: "spin 1s linear infinite",
              }}
            />
          )}
        </button>
      </div>
      <style>
        {`
          @keyframes spin {
            0% { transform: rotate(0deg);}
            100% { transform: rotate(360deg);}
          }
        `}
      </style>
      {dataset.length > 0 && (
        <table
          style={{ width: "100%", borderCollapse: "collapse", marginTop: 10 }}
        >
          <thead>
            <tr>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>#</th>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>Level</th>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>Section</th>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>Type</th>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>Title</th>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>Question</th>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>Answer</th>
            </tr>
          </thead>
          <tbody>
            {dataset.map((item, idx) => (
              <tr
                key={idx}
                onClick={() => handleRowClick(idx)}
                style={{
                  cursor: "pointer",
                  backgroundColor: editIndex === idx ? "#eef" : "white",
                }}
              >
                <td style={{ border: "1px solid #ccc", padding: 5 }}>
                  {idx + 1}
                </td>
                <td style={{ border: "1px solid #ccc", padding: 5 }}>
                  {item.level}
                </td>
                <td style={{ border: "1px solid #ccc", padding: 5 }}>
                  {item.section}
                </td>
                <td style={{ border: "1px solid #ccc", padding: 5 }}>
                  {item.type}
                </td>
                <td style={{ border: "1px solid #ccc", padding: 5 }}>
                  {item.title}
                </td>
                <td style={{ border: "1px solid #ccc", padding: 5 }}>
                  {item.question}
                </td>
                <td style={{ border: "1px solid #ccc", padding: 5 }}>
                  {item.answer}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default TopikAnnotationForm;
