import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css'; // 我们会在下一步创建CSS文件

function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-left">
        <img src="static/ysc_search.png" alt="logo" className="navbar-left-logo" />
        汉服租赁
      </div>
      <ul className="navbar-center nav-links">
        <li><Link to="/">主页</Link></li>
        <li><Link to="/return">归还界面</Link></li>
        <li><Link to="/reserve">预定取衣界面</Link></li>
        <li><Link to="/recommendations">智能推荐</Link></li>
        <li><Link to="/tryout">虚拟试衣</Link></li>
      </ul>
      <div className="navbar-right">
        <img src="static/ysc_logo.png" alt="logo" className="navbar-right-logo" />
        <input type="text" className="search-bar" placeholder="搜索..." />
      </div>
    </nav>
  );
}

export default Navbar;
