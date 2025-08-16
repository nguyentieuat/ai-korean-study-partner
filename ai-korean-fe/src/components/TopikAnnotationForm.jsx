// src/components/TopikAnnotationForm.jsx
import React, { useState } from "react";
const backendUrl = import.meta.env.VITE_API_URL;

const typeMapping = [
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
  { display: "49~68번: Điền + chủ đề (Đọc)", jsonType: "Doc_Dien_ChuDe" },
  {
    display: "63~64번: Mục đích viết + nội dung (Đọc)",
    jsonType: "Doc_MucDich",
  },
  { display: "69~70번: Điền + suy luận (Đọc)", jsonType: "Doc_Dien_SuyLuan" },
];

const TopikAnnotationForm = ({ userInfo }) => {
  const [type, setType] = useState(typeMapping[0].jsonType);
  const [title, setTitle] = useState("");
  const [question, setQuestion] = useState("");
  const [dialogue, setDialogue] = useState([{ speaker: "", text: "" }]);
  const [passage, setPassage] = useState("");
  const [options, setOptions] = useState({ A: "", B: "", C: "", D: "" });
  const [answer, setAnswer] = useState("");
  const [dataset, setDataset] = useState([]);
  const [editIndex, setEditIndex] = useState(null);

  const isNghe = type.startsWith("Nghe_");

  const handleDialogueChange = (index, field, value) => {
    const newDialogue = [...dialogue];
    newDialogue[index][field] = value;
    setDialogue(newDialogue);
  };

  const addDialogueLine = () =>
    setDialogue([...dialogue, { speaker: "", text: "" }]);

  const handleAddOrUpdate = () => {
    const item = {
      type,
      title,
      question,
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
    setEditIndex(index);
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

    const payload = {
      annotator_name: userInfo.name,
      annotator_phone: userInfo.phone,
      annotations: dataset,
    };

    try {
      const response = await fetch(
        `${backendUrl}/api/cooperate_topik_annotator`,
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

      // reset form
      setTitle("");
      setQuestion("");
      setDialogue([{ speaker: "", text: "" }]);
      setPassage("");
      setOptions({ A: "", B: "", C: "", D: "" });
      setAnswer("");
      setDataset([]);
    } catch (err) {
      console.error("❌ Lỗi khi gửi:", err);
      alert("Có lỗi khi gửi dữ liệu!");
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 15 }}>
      <h2>Topik Annotation Form</h2>

      <div style={{ display: "flex", flexDirection: "column" }}>
        <label>Chọn loại câu hỏi:</label>
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          style={{ padding: 5, maxWidth: 400 }}
        >
          {typeMapping.map((t) => (
            <option key={t.jsonType} value={t.jsonType}>
              {t.display}
            </option>
          ))}
        </select>
      </div>

      <div style={{ display: "flex", flexDirection: "column" }}>
        <label>Title:</label>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          style={{ padding: 5, width: "100%" }}
        />
      </div>

      {isNghe ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <label>Dialogue:</label>
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
        <input
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          style={{ width: 60, padding: 5 }}
        />
      </div>

      <div style={{ display: "flex", gap: 10 }}>
        <button onClick={handleAddOrUpdate}>
          {editIndex !== null ? "Cập nhật" : "Add"}
        </button>
        <button onClick={handleSubmitAll}>Submit All</button>
      </div>

      {dataset.length > 0 && (
        <table
          style={{ width: "100%", borderCollapse: "collapse", marginTop: 10 }}
        >
          <thead>
            <tr>
              <th style={{ border: "1px solid #ccc", padding: 5 }}>#</th>
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
