import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faHome, faMicrophone, faComments, faClipboard, faBook, faHandshake } from '@fortawesome/free-solid-svg-icons';

const NavBar = () => {
  const [isNavCollapsed, setIsNavCollapsed] = useState(true);
  const navRef = useRef(null);
  const location = useLocation();

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
    if (!isNavCollapsed) document.addEventListener('click', handleClickOutside);
    else document.removeEventListener('click', handleClickOutside);

    return () => document.removeEventListener('click', handleClickOutside);
  }, [isNavCollapsed]);

  const navItems = [
    { name: 'Trang chủ', path: '/', icon: faHome },
    { name: 'Luyện phát âm', path: '/pronunciation', icon: faMicrophone },
    { name: 'Luyện hội thoại', path: '/conversation', icon: faComments },
    { name: 'Luyện đề', path: '/practice', icon: faClipboard },
    // { name: 'Tài liệu', path: '/materials', icon: faBook },
    { name: 'Hợp tác Phát triển', path: '/cooperate', icon: faHandshake },
  ];

  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark shadow-lg" ref={navRef}>
      <div className="container-fluid">
        <Link className="navbar-brand fw-bold text-gradient" to="/">
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

        <div className={`collapse navbar-collapse ${isNavCollapsed ? '' : 'show'}`} id="navbarNav">
          <ul className="navbar-nav ms-auto">
            {navItems.map((item) => (
              <li className="nav-item mx-1" key={item.path}>
                <Link
                  className={`nav-link d-flex align-items-center px-3 py-2 rounded ${
                    location.pathname === item.path
                      ? 'bg-gradient text-white shadow-sm'
                      : 'text-light hover-bg-gradient'
                  }`}
                  to={item.path}
                  onClick={() => setIsNavCollapsed(true)}
                >
                  <FontAwesomeIcon icon={item.icon} className="me-2" />
                  {item.name}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <style>
        {`
          .text-gradient {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
          .bg-gradient {
            background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
          }
          .hover-bg-gradient:hover {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white !important;
            transition: 0.3s;
          }
        `}
      </style>
    </nav>
  );
};

export default NavBar;
