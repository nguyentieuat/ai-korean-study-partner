import React from 'react';
import NavBar from '../components/NavBar';

const HomePage = () => {
  return (
    <>
      <div className="min-vh-100">
        <section
          style={{
            background: 'radial-gradient(circle at top, rgb(223, 218, 218), rgb(223, 218, 218))',
            padding: '4rem 2rem',
            textAlign: 'center',
          }}
        >
          <h1 className="fw-bold display-5">AI Korean Study Partner</h1>
          <p className="lead">
            Cùng học tiếng Hàn với phản hồi thông minh từ trí tuệ nhân tạo.
          </p>
        </section>
      </div>
    </>
  );
};

export default HomePage;
