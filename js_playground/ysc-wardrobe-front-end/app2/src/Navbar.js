import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css'; // 我们会在下一步创建CSS文件

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-left">
        <img src="logo192.png" alt="logo" className="navbar-logo" />
        汉服租赁
      </div>
      <ul className="navbar-center nav-links">
        <li><Link to="/">主页</Link></li>
        <li><Link to="/return">归还界面</Link></li>
        <li><Link to="/reserve">预定取衣界面</Link></li>
        <li><Link to="/recommendations">智能推荐</Link></li>
      </ul>
      <div className="navbar-right">
        <img src="logo192.png" alt="logo" className="navbar-logo" />
        <input type="text" className="search-bar" placeholder="搜索..." />
      </div>
    </nav>
  );
}

export default Navbar;
