// src/pages/HomePage.jsx
import React from 'react';

const HomePage = () => {
  return (
    <>
      <section
        style={{
          background: 'radial-gradient(circle at top, rgb(223, 218, 218), rgb(223, 218, 218))',
          padding: '4rem 2rem',
          textAlign: 'center',
        }}
      >
        <h1 className="fw-bold display-5">AI Korean Study Partner</h1>
        <p className="lead">Cùng học tiếng Hàn với phản hồi thông minh từ trí tuệ nhân tạo.</p>
      </section>
    </>
  );
};

export default HomePage;
