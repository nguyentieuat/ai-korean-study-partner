import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

const NavBar = () => {
  const [isNavCollapsed, setIsNavCollapsed] = useState(true);
  const navRef = useRef(null);  // Tham chiếu vùng menu

  const handleToggle = () => setIsNavCollapsed(!isNavCollapsed);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        navRef.current &&
        !navRef.current.contains(event.target) &&
        !event.target.classList.contains('navbar-toggler') &&
        !event.target.closest('.navbar-toggler')
      ) {
        setIsNavCollapsed(true);
      }
    };

    if (!isNavCollapsed) {
      document.addEventListener('click', handleClickOutside);
    } else {
      document.removeEventListener('click', handleClickOutside);
    }

    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [isNavCollapsed]);

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark px-4" ref={navRef}>
      <Link className="navbar-brand fw-bold text-info" to="/">
        AI Korean <span className="text-success">Partner</span>
      </Link>
      <button
        className="navbar-toggler"
        type="button"
        aria-controls="navbarNav"
        aria-expanded={!isNavCollapsed}
        aria-label="Toggle navigation"
        onClick={handleToggle}
      >
        <span className="navbar-toggler-icon"></span>
      </button>
      <div
        className={`collapse navbar-collapse ${isNavCollapsed ? '' : 'show'}`}
        id="navbarNav"
      >
        <ul className="navbar-nav ms-auto">
          <li className="nav-item">
            <Link className="nav-link" to="/" onClick={() => setIsNavCollapsed(true)}>Trang chủ</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/pronunciation" onClick={() => setIsNavCollapsed(true)}>Luyện phát âm</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/conversation" onClick={() => setIsNavCollapsed(true)}>Luyện hội thoại</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/practice" onClick={() => setIsNavCollapsed(true)}>Luyện đề</Link>
          </li>
          <li className="nav-item">
            <Link className="nav-link" to="/materials" onClick={() => setIsNavCollapsed(true)}>Tài liệu</Link>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default NavBar;
