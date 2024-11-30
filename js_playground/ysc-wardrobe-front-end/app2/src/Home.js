// src/Home.js
import React from 'react';
import MainButtons from './MainButtons';
import './Home.css'

function Home() {
  return <div class='home-page'>
    <h2>主页</h2>
    <img src='logo192.png' className='login-qr'/>
    <p>扫码登陆</p>
    <MainButtons />
  </div>
}

export default Home;