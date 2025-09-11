// src/pages/MaterialsPage.jsx
import React from 'react';

const MaterialsPage = () => {
  // Ví dụ danh sách tài liệu
  const documents = [
    {
      title: "Ngữ pháp cơ bản TOPIK 1",
      description: "Tài liệu tổng hợp ngữ pháp TOPIK 1 kèm ví dụ minh họa.",
      link: "/materials/grammar-topik1.pdf",
      category: "Ngữ pháp",
    },
    {
      title: "Bài tập luyện nghe TOPIK 1",
      description: "Bộ bài tập luyện nghe theo các chủ đề TOPIK 1.",
      link: "/materials/listening-topik1.pdf",
      category: "Nghe hiểu",
    },
    {
      title: "Ngữ pháp nâng cao TOPIK 2",
      description: "Tài liệu tổng hợp ngữ pháp TOPIK 2 với ví dụ và bài tập.",
      link: "/materials/grammar-topik2.pdf",
      category: "Ngữ pháp",
    },
    {
      title: "Bài tập luyện hội thoại TOPIK 2",
      description: "Bài tập luyện hội thoại và phản xạ ngôn ngữ TOPIK 2.",
      link: "/materials/conversation-topik2.pdf",
      category: "Hội thoại",
    },
  ];

  return (
    <div className="container py-5">
      <h1 className="text-center mb-5">Tài liệu học TOPIK</h1>

      <div className="row">
        {documents.map((doc, index) => (
          <div key={index} className="col-md-6 col-lg-4 mb-4">
            <div className="card h-100 shadow-sm">
              <div className="card-body d-flex flex-column">
                <h5 className="card-title">{doc.title}</h5>
                <p className="card-text flex-grow-1">{doc.description}</p>
                <span className="badge bg-primary mb-2">{doc.category}</span>
                <a
                  href={doc.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn btn-outline-primary mt-auto"
                >
                  Xem / Tải xuống
                </a>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MaterialsPage;
