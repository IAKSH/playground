// src/Home.js
import React from 'react';
import MainButtons from './MainButtons';
import './Home.css'

function Home() {
  return <div class='home-page'>
    <h2>主页</h2>
    <img src='static/wechat_app_qrcode.jpg' className='login-qr'/>
    <p>扫一扫微信小程序</p>
    <MainButtons />
  </div>
}

export default Home;